# Bank Marketing Prediction using Gradient Boosting

## Project Overview
This project uses a Gradient Boosting Classifier to predict whether a bank customer will subscribe to a term deposit based on customer information and previous marketing campaign data.

The project focuses on classification, data preprocessing, model evaluation, and hyperparameter tuning using Scikit-learn.

## Dataset
- **Dataset:** Bank Marketing
- **Rows:** 11,162
- **Columns:** 17
- **Target variable:** `deposit`
- **Classes:** `yes` and `no`

The dataset contains customer demographics, account balance, contact information, and previous campaign outcomes.

## Technologies Used
- Python
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## Project Workflow
1. Explored the dataset and checked for duplicate records.
2. Separated features and the target variable.
3. Excluded `duration` to predict subscription before the call.
4. Encoded categorical features using One-Hot Encoding.
5. Split the data into training and testing sets using stratification.
6. Built a preprocessing and classification pipeline.
7. Trained a Gradient Boosting Classifier.
8. Tuned hyperparameters using GridSearchCV with 5-fold cross-validation.
9. Evaluated the model using classification metrics and ROC-AUC.
10. Analyzed feature importance.

## Model
**Algorithm:** Gradient Boosting Classifier

The model was implemented using Scikit-learn's `GradientBoostingClassifier` inside a Pipeline.

### Best Hyperparameters
- `n_estimators`: 150
- `learning_rate`: 0.1
- `max_depth`: 3
- `min_samples_leaf`: 1

GridSearchCV selected these parameters based on the F1-score for the positive class (`yes`).

## Model Evaluation

| Metric | Result |
|---|---:|
| Training Accuracy | 75.90% |
| Testing Accuracy | 74.52% |
| F1-score (`yes`) | 0.70 |
| ROC-AUC | 0.795 |
| Best Cross-Validation F1-score (`yes`) | 0.687 |

The training and testing accuracies are close, suggesting no obvious sign of severe overfitting.

The ROC-AUC score indicates that the model can distinguish between subscribers and non-subscribers better than random ranking.

## Feature Importance
The most important features identified by the model included:
- Previous campaign outcome: success
- Contact method: unknown
- Age
- Account balance
- Day of the month

Feature importance represents the model's reliance on features for its tree-based decisions. It does not establish causation.

## Key Learning Outcomes
- Data preprocessing with `ColumnTransformer`
- Categorical encoding using `OneHotEncoder`
- Building Scikit-learn Pipelines
- Training Gradient Boosting models
- Hyperparameter tuning with GridSearchCV
- Evaluating classification models
- Understanding feature importance and ROC-AUC

## Future Improvements
- Compare Gradient Boosting with Random Forest and other boosting algorithms.
- Tune the classification threshold to improve subscriber recall.
- Use cross-validation to compare additional parameter combinations.
- Analyze model errors and evaluate performance on unseen data.

## Conclusion
This project demonstrates an end-to-end machine learning classification workflow for predicting bank term-deposit subscriptions. It covers preprocessing, model training, hyperparameter tuning, evaluation, and feature importance analysis.

The model achieved 74.52% test accuracy and a ROC-AUC score of 0.795.

