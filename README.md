# Customer Churn Prediction – Model Development, Validation, and Deployment

### SRM University  
**Department of Computational Intelligence – School of Computing**  
**Course:** Inferential Statistics and Predictive Analytics (21AIC401T)  
**Assignment Type:** Case Study-Based Modeling Project  

---

## Student Details
| Field | Information |
|:------|:-------------|
| **Name** | CH.Karthikeya Sriram |
| **Reg. No** | RA2211047010089 |
| **Degree** | B.Tech – Artificial Intelligence |
| **Section** | AI-B |
| **Instructor** | Dr.Md shahnawaz  hussain |
| **Dataset Source** | [Kaggle – Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |

---

## Objective

The goal of this project is to **develop, validate, compare, and deploy** predictive models that can identify customers likely to **churn**.  
The project applies **inferential statistics** and **predictive modeling** techniques, including model validation, comparison, evaluation, and deployment.

---

## Case Background

Customer churn (loss of customers) is a major issue in telecom and subscription-based industries.  
By predicting which customers are at high risk of leaving, companies can proactively retain them through targeted offers, better customer experience, and personalized engagement.

---

## Dataset Overview

**Dataset Name:** Telco Customer Churn  
**Records:** 7,043 customers  
**Features:** 21 columns  

**Target Variable:** `Churn` (Yes = 1, No = 0)  

| Category | Example Columns |
|:----------|:----------------|
| **Demographics** | gender, SeniorCitizen, Partner, Dependents |
| **Account Information** | tenure, Contract, PaperlessBilling, PaymentMethod |
| **Services** | InternetService, OnlineSecurity, StreamingTV, TechSupport |
| **Charges** | MonthlyCharges, TotalCharges |

**Cleaning Steps:**
- Converted `TotalCharges` to numeric using `pd.to_numeric(errors='coerce')`
- Dropped rows with missing `TotalCharges`
- Removed duplicates using `drop_duplicates()`
- Handled outliers via IQR method
- Applied one-hot encoding for categorical variables

---

## Exploratory Data Analysis (EDA)

**Key Insights:**
- Around **26–27%** of customers churned.
- **Month-to-month** contracts have the highest churn rate.
- **Short-tenure** customers are most prone to churn.
- Customers paying via **Electronic Check** show the highest churn.
- **Fiber optic** internet users have above-average churn.

### Figures
| Figure | Description |
|:-------|:-------------|
| **Figure 1** | Churn Distribution |
| **Figure 2** | Churn by Contract Type |
| **Figure 3** | Churn by Tenure |
| **Figure 4** | Churn by Payment Method |
| **Figure 5** | Correlation Matrix (encoded features) |

---

## Model Development

Two models were developed and compared:

### CHAID (Decision Tree)
- Segments customers based on statistically significant predictors.
- Reveals **interpretable decision rules** useful for business strategy.

**Top Predictors:**
1. Tenure  
2. InternetService_Fiber optic  
3. TotalCharges  
4. MonthlyCharges  
5. Contract type

**Example Rules:**
- *Short tenure + Month-to-month contract → High churn probability*  
- *Two-year contract → Low churn probability*

---

### Logistic Regression
- Provides **probabilistic predictions** for churn.
- Highly interpretable coefficients for business explanation.
- Trained with 80/20 train-test split, scaled numeric features.

---

## Model Comparison and Evaluation

| Metric | CHAID | Logistic Regression |
|:--------|:------|:--------------------|
| **Accuracy** | 0.7754 | **0.7875** |
| **ROC-AUC** | 0.8130 | **0.8618** |

**Interpretation:**
- Logistic Regression performed slightly better in both Accuracy and ROC-AUC.
- CHAID provided valuable **decision rules** for actionable insights.

### Evaluation Plots
| Figure | Description |
|:--------|:-------------|
| **Figure 6** | Feature Importances (CHAID/Decision Tree) |
| **Figure 7** | ROC Curves (CHAID vs Logistic Regression) |
| **Figure 8** | Gains Chart (Logistic Regression) |
| **Figure 9** | Lift Chart (Logistic Regression) |

---

## Model Deployment and Updating

### Deployment Process:
1. **Export Model:**
   ```python
   import joblib
   joblib.dump(lr_model, "models/logistic_churn_model.joblib")
   joblib.dump(encoder, "models/preprocessing_encoder.joblib")
Predict on New Data:

python
Copy code
model = joblib.load("models/logistic_churn_model.joblib")
proba = model.predict_proba(new_data)[:,1]
Deployment Options:

Batch Scoring (nightly updates)

Real-Time API using Flask or FastAPI

Integration into CRM dashboards

Updating Strategy:
Periodic retraining using new customer data.

Monitor performance drift using Population Stability Index (PSI).

CI/CD automation with GitHub Actions or MLflow.

## Key Findings
Short tenure and month-to-month contracts are strong churn indicators.

Fiber optic users are more likely to leave, indicating possible service dissatisfaction.

Electronic check payments correlate with higher churn — auto-pay adoption could reduce churn.

## References
Kaggle – Telco Customer Churn Dataset

scikit-learn Documentation

IBM SPSS Modeler Guide – CHAID Algorithm

PyCHAID Documentation

## Conclusion
This project successfully applied inferential statistics and machine learning to predict customer churn.
The Logistic Regression model achieved the highest performance (ROC-AUC = 0.8618), while the CHAID model offered interpretable business rules.
A deployment pipeline and model updating strategy ensure scalability and ongoing relevance in real-world environments.
### “Predictive analytics is not about predicting the future perfectly — it’s about improving decisions today.”
