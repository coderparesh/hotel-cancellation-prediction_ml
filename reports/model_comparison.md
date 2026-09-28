# Baseline Model Comparison

**Dataset shape:** (86963, 59)

|                    |   accuracy |   precision |   recall |     f1 |   roc_auc |
|:-------------------|-----------:|------------:|---------:|-------:|----------:|
| XGBoost            |     0.8514 |      0.7503 |   0.6832 | 0.7152 |    0.9169 |
| RandomForest       |     0.8466 |      0.7646 |   0.6333 | 0.6928 |    0.9076 |
| GradientBoosting   |     0.8371 |      0.7541 |   0.5985 | 0.6674 |    0.8964 |
| DecisionTree       |     0.7989 |      0.6319 |   0.632  | 0.6319 |    0.7476 |
| LogisticRegression |     0.7852 |      0.6711 |   0.4187 | 0.5157 |    0.7954 |

> Models trained with default hyperparameters. Top candidates will proceed to Sprint 3 tuning.
