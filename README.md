# Bank Marketing Lead Conversion Prediction

## Summary
This project predicts whether a client will subscribe to a term deposit after a phone marketing campaign, using the Bank Marketing dataset from Kaggle. The dataset consists of 45,421 records and 16 features, including demographic and campaign-related attributes.

During the exploratory data analysis (EDA), we observed a strong class imbalance, with the majority of clients not subscribing ("no" class dominating). We also found that clients with certain jobs (for example, management or self-employed) and higher average balances were more likely to subscribe to the term deposit product.

For preprocessing, we confirmed that there were no missing values in the dataset. We performed one-hot encoding of categorical variables and applied a train-validation split (80/20) with stratification to maintain class proportions. This approach helps prevent data leakage and ensures reproducibility.

As a baseline, we trained a Logistic Regression model. The model achieved reasonable accuracy and an acceptable AUC score, but recall for the minority "yes" class was initially low due to the dataset imbalance. To improve this, we introduced class weighting in Logistic Regression, which increased recall for the positive class, making the model more effective for identifying potential subscribers.

If more time were available, we would explore advanced models such as Random Forest or XGBoost, perform additional feature engineering (e.g., creating interaction terms, binning numerical features), and conduct thorough hyperparameter tuning to further improve precision, recall, and overall robustness.

## Setup

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Run
Open and run all cells in bank_marketing.ipynb sequentially.

## Files
bank_marketing.ipynb — Main notebook containing all code: EDA, preprocessing pipeline, baseline modeling, improvements, and final evaluation.

README.md — Project instructions and concise summary (this file).

requirements.txt — (Optional) Only if additional non-standard Python libraries are added.

Notes
The current pipeline provides a strong foundation for predicting term deposit subscriptions and is designed to be reproducible and easy to extend.

All code is modular, clearly commented, and structured with section headings for clarity.
