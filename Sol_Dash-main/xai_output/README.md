# XAI Analysis Report

## Dataset Information
- **Dataset Path:** C:\Users\kunya\PycharmProjects\Sol_Dash\general_sol_tokens_last30d.csv
- **Model Path:** C:\Users\kunya\PycharmProjects\Sol_Dash\ensemble_model.pt
- **Dataset Shape:** 52 samples, 54 features
- **Train/Test Split:** 41 / 11
- **Analysis Date:** 2025-07-05 20:12:00

## Feature Importance Analysis

### Top 10 Most Important Features (Tree-based):
1. **price_change_1h_rolling_mean**: 0.0755
2. **price_change_24h**: 0.0631
3. **price_change_1h**: 0.0500
4. **token_age_hours_rolling_std**: 0.0372
5. **price_change_6h**: 0.0323
6. **volume_change_1h**: 0.0281
7. **volume_change_6h**: 0.0278
8. **price_change_24h_rolling_mean**: 0.0256
9. **current_price_usd_rolling_mean**: 0.0253
10. **volume_change_6h_rolling_std**: 0.0249

## Generated Visualizations
- Tree-based Feature Importance: `tree_feature_importance.png`
- Permutation Importance: `permutation_importance.png`
- SHAP Summary Plot: `shap_summary.png`
- SHAP Bar Plot: `shap_bar.png`
- SHAP Waterfall Plot: `shap_waterfall.png`
- LIME Explanations: `lime_explanation_*.png`
- Correlation Matrix: `correlation_matrix.png`
- Confusion Matrix: `confusion_matrix.png`
- ROC Curve: `roc_curve.png`

## Files Generated
- Detailed results: `xai_report.json`
- LIME explanations: `lime_explanation_*.html`
