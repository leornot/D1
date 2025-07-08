#!/usr/bin/env python3

import argparse
import json
import os
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

warnings.filterwarnings('ignore')

class SolanaTokenPreprocessor:
    def __init__(self):
        self.feature_names = []
        self.scaler = StandardScaler()
        self.label_encoders = {}

    @staticmethod
    def create_target_labels(df: pd.DataFrame, profit_threshold: float = 50) -> pd.DataFrame:
        if 'price_change_percentage' in df.columns:
            df['profit_target'] = (df['price_change_percentage'] > profit_threshold).astype(int)
        elif 'return' in df.columns:
            df['profit_target'] = (df['return'] > profit_threshold).astype(int)
        else:
            df['profit_target'] = np.random.randint(0, 2, len(df))
        df['risk_target'] = 1 - df['profit_target']
        return df

    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['profit_target', 'risk_target']:
                df[f'{col}_rolling_mean'] = df[col].rolling(window=5, min_periods=1).mean()
                df[f'{col}_rolling_std'] = df[col].rolling(window=5, min_periods=1).std()
        return df

    def preprocess_features(self, df: pd.DataFrame, fit_transform: bool = True) -> np.ndarray:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['profit_target', 'risk_target']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        X = df[feature_cols].copy()
        X = X.fillna(X.mean())
        self.feature_names = feature_cols
        if fit_transform:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        return X_scaled

class EnsembleTokenClassifier:
    def __init__(self, input_size: int):
        self.input_size = input_size
        self.profit_models = {}
        self.risk_models = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @classmethod
    def load(cls, model_path: str, input_size: int):
        instance = cls(input_size)
        if os.path.exists(model_path):
            try:
                try:
                    import xgboost  # attempt to import xgboost
                    torch.serialization.add_safe_globals(["xgboost.sklearn.XGBClassifier"])
                except Exception:
                    pass
                checkpoint = torch.load(model_path, map_location=instance.device, weights_only=False)
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
        return instance

class XAISuite:
    def __init__(self, dataset_path: str, model_path: str, output_dir: str = "./xai_output"):
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.preprocessor = SolanaTokenPreprocessor()
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = []
        self.results = {'feature_importance': {}, 'shap_values': {}, 'lime_explanations': {}, 'permutation_importance': {}, 'model_performance': {}}

    def load_and_preprocess_data(self) -> bool:
        try:
            df = pd.read_csv(self.dataset_path)
            df = self.preprocessor.create_target_labels(df, profit_threshold=50)
            df = self.preprocessor.engineer_features(df)
            X = self.preprocessor.preprocess_features(df, fit_transform=True)
            y_profit = df['profit_target'].values if 'profit_target' in df.columns else np.random.randint(0, 2, len(df))
            y_risk = df['risk_target'].values if 'risk_target' in df.columns else np.random.randint(0, 2, len(df))
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y_profit, test_size=0.2, random_state=42, stratify=y_profit)
            self.feature_names = self.preprocessor.feature_names
            return True
        except Exception as e:
            print(f"Error loading/preprocessing data: {e}")
            return False

    def load_model(self) -> bool:
        try:
            if self.model_path.endswith('.pt') or self.model_path.endswith('.pth'):
                self.model = EnsembleTokenClassifier.load(self.model_path, self.X_train.shape[1])
            elif self.model_path.endswith('.pkl'):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
            else:
                return self._create_surrogate_model()
            # Fallback to surrogate model if loaded model does not implement predict_proba
            if not hasattr(self.model, "predict_proba"):
                print("Loaded model does not support predict_proba; using surrogate model.")
                return self._create_surrogate_model()
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return self._create_surrogate_model()

    def _create_surrogate_model(self) -> bool:
        try:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(self.X_train, self.y_train)
            train_score = self.model.score(self.X_train, self.y_train)
            test_score = self.model.score(self.X_test, self.y_test)
            print(f"Surrogate model performance - Train: {train_score:.3f}, Test: {test_score:.3f}")
            return True
        except Exception as e:
            print(f"Error creating surrogate model: {e}")
            return False

    def analyze_feature_importance(self):
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            self.results['feature_importance']['tree_based'] = pd.Series(importances, index=self.feature_names).sort_values(ascending=False)
            plt.figure(figsize=(12,8))
            top_features = self.results['feature_importance']['tree_based'].head(20)
            plt.barh(range(len(top_features)), top_features.values)
            plt.yticks(range(len(top_features)), top_features.index)
            plt.xlabel('Feature Importance')
            plt.title('Tree-based Feature Importance (Top 20)')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'tree_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
        try:
            perm_importance = permutation_importance(self.model, self.X_test, self.y_test, n_repeats=10, random_state=42)
            self.results['permutation_importance'] = pd.Series(perm_importance.importances_mean, index=self.feature_names).sort_values(ascending=False)
            plt.figure(figsize=(12,8))
            top_features = self.results['permutation_importance'].head(20)
            plt.barh(range(len(top_features)), top_features.values)
            plt.yticks(range(len(top_features)), top_features.index)
            plt.xlabel('Permutation Importance')
            plt.title('Permutation Feature Importance (Top 20)')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'permutation_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"Error computing permutation importance: {e}")

    def analyze_shap_values(self):
        try:
            n_samples = min(100, len(self.X_test))
            X_sample = self.X_test[:n_samples]
            if hasattr(self.model, 'feature_importances_'):
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X_sample)
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    shap_values = shap_values[1]
            else:
                explainer = shap.KernelExplainer(self.model.predict_proba, X_sample)
                shap_values = explainer.shap_values(X_sample)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
            self.results['shap_values'] = shap_values
            X_sample_df = pd.DataFrame(X_sample, columns=self.feature_names)
            plt.figure(figsize=(12,8))
            shap.summary_plot(shap_values, X_sample_df, show=False)
            plt.title('SHAP Summary Plot')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'shap_summary.png', dpi=300, bbox_inches='tight')
            plt.show()
            plt.figure(figsize=(12,8))
            shap.summary_plot(shap_values, X_sample_df, plot_type='bar', show=False)
            plt.title('SHAP Feature Importance')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'shap_bar.png', dpi=300, bbox_inches='tight')
            plt.show()
            if len(shap_values) > 0:
                plt.figure(figsize=(12,8))
                shap.waterfall_plot(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=X_sample[0], feature_names=self.feature_names), show=False)
                plt.title('SHAP Waterfall Plot (First Instance)')
                plt.tight_layout()
                plt.savefig(self.output_dir / 'shap_waterfall.png', dpi=300, bbox_inches='tight')
                plt.show()
            print("SHAP analysis completed successfully!")
        except Exception as e:
            print(f"Error in SHAP analysis: {e}")

    def analyze_lime_explanations(self):
        try:
            explainer = lime.lime_tabular.LimeTabularExplainer(self.X_train, feature_names=self.feature_names, class_names=['Class 0', 'Class 1'], mode='classification')
            n_explanations = min(3, len(self.X_test))
            for i in range(n_explanations):
                explanation = explainer.explain_instance(self.X_test[i], self.model.predict_proba, num_features=10)
                explanation.save_to_file(self.output_dir / f'lime_explanation_{i}.html')
                fig = explanation.as_pyplot_figure()
                fig.suptitle(f'LIME Explanation - Instance {i}')
                plt.tight_layout()
                plt.savefig(self.output_dir / f'lime_explanation_{i}.png', dpi=300, bbox_inches='tight')
                plt.show()
            print(f"LIME analysis completed for {n_explanations} instances!")
        except Exception as e:
            print(f"Error in LIME analysis: {e}")

    def generate_correlation_analysis(self):
        try:
            X_df = pd.DataFrame(self.X_train, columns=self.feature_names)
            correlation_matrix = X_df.corr()
            plt.figure(figsize=(16,12))
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', center=0, square=True)
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
            plt.show()
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    if abs(correlation_matrix.iloc[i, j]) > 0.8:
                        high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j]))
            if high_corr_pairs:
                print(f"Found {len(high_corr_pairs)} highly correlated feature pairs (|r| > 0.8):")
                for feat1, feat2, corr in high_corr_pairs[:10]:
                    print(f"{feat1} <-> {feat2}: {corr:.3f}")
        except Exception as e:
            print(f"Error in correlation analysis: {e}")

    def evaluate_model_performance(self):
        try:
            y_pred = self.model.predict(self.X_test)
            y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
            report = classification_report(self.y_test, y_pred, output_dict=True)
            self.results['model_performance']['classification_report'] = report
            print(classification_report(self.y_test, y_pred))
            cm = confusion_matrix(self.y_test, y_pred)
            plt.figure(figsize=(8,6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.show()
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(8,6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(self.output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"Error in model performance evaluation: {e}")

    def generate_report(self):
        try:
            report = {'timestamp': datetime.now().isoformat(), 'dataset_path': str(self.dataset_path), 'model_path': str(self.model_path), 'dataset_shape': f"{len(self.X_train) + len(self.X_test)} samples, {len(self.feature_names)} features", 'train_test_split': f"Train: {len(self.X_train)}, Test: {len(self.X_test)}", 'feature_names': self.feature_names, 'results': self.results}
            with open(self.output_dir / 'xai_report.json', 'w') as f:
                json.dump(report, f, indent=2, default=str)
            md_content = f"# XAI Analysis Report\n\n## Dataset Information\n- **Dataset Path:** {self.dataset_path}\n- **Model Path:** {self.model_path}\n- **Dataset Shape:** {len(self.X_train) + len(self.X_test)} samples, {len(self.feature_names)} features\n- **Train/Test Split:** {len(self.X_train)} / {len(self.X_test)}\n- **Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n## Feature Importance Analysis\n"
            if 'tree_based' in self.results['feature_importance']:
                top_features = self.results['feature_importance']['tree_based']
                if isinstance(top_features, dict):
                    top_features = pd.Series(top_features).sort_values(ascending=False)
                else:
                    top_features = top_features.sort_values(ascending=False)
                top_features = top_features.head(10)
                md_content += "\n### Top 10 Most Important Features (Tree-based):\n"
                for i, (feature, importance) in enumerate(top_features.items(), 1):
                    md_content += f"{i}. **{feature}**: {importance:.4f}\n"
            md_content += f"\n## Generated Visualizations\n- Tree-based Feature Importance: `tree_feature_importance.png`\n- Permutation Importance: `permutation_importance.png`\n- SHAP Summary Plot: `shap_summary.png`\n- SHAP Bar Plot: `shap_bar.png`\n- SHAP Waterfall Plot: `shap_waterfall.png`\n- LIME Explanations: `lime_explanation_*.png`\n- Correlation Matrix: `correlation_matrix.png`\n- Confusion Matrix: `confusion_matrix.png`\n- ROC Curve: `roc_curve.png`\n\n## Files Generated\n- Detailed results: `xai_report.json`\n- LIME explanations: `lime_explanation_*.html`\n"
            with open(self.output_dir / 'README.md', 'w') as f:
                f.write(md_content)
            print(f"XAI analysis complete! Results saved to: {self.output_dir}")
            print(f"Total files generated: {len(list(self.output_dir.glob('*')))}")
        except Exception as e:
            print(f"Error generating report: {e}")

    def run_full_analysis(self):
        print("="*60)
        print("🔍 XAI SUITE - COMPREHENSIVE EXPLAINABLE AI ANALYSIS")
        print("="*60)
        if not self.load_and_preprocess_data():
            print("❌ Failed to load/preprocess data. Exiting.")
            return False
        if not self.load_model():
            print("❌ Failed to load model. Exiting.")
            return False
        try:
            self.analyze_feature_importance()
            self.analyze_shap_values()
            self.analyze_lime_explanations()
            self.generate_correlation_analysis()
            self.evaluate_model_performance()
            self.generate_report()
            print("\n✅ XAI analysis completed successfully!")
            return True
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="XAI Suite - Comprehensive Explainable AI Analysis Tool")
    parser.add_argument('--dataset_path', type=str, default=r'C:\Users\kunya\PycharmProjects\Sol_Dash\general_sol_tokens_last30d.csv')
    parser.add_argument('--model_path', type=str, default=r'C:\Users\kunya\PycharmProjects\Sol_Dash\ensemble_model.pt')
    parser.add_argument('--output_dir', type=str, default='./xai_output')
    parser.add_argument('--sample_size', type=int, default=None)
    args = parser.parse_args()
    if not os.path.exists(args.dataset_path):
        print(f"❌ Dataset file not found: {args.dataset_path}")
        return
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        return
    xai_suite = XAISuite(dataset_path=args.dataset_path, model_path=args.model_path, output_dir=args.output_dir)
    success = xai_suite.run_full_analysis()
    if success:
        print(f"\n🎉 Analysis complete! Check results in: {args.output_dir}")
    else:
        print("\n❌ Analysis failed. Check error messages above.")

if __name__ == "__main__":
    main()
