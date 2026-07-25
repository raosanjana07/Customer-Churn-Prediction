# Customer Churn Prediction Web App

![Python](https://img.shields.io/badge/Python-3.14-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Random_Forest-orange)
![Status](https://img.shields.io/badge/Status-Deployed-success)

An end-to-end machine learning web application that predicts whether a telecom customer is likely to leave the company.

The project uses a **Random Forest Classifier** and provides churn probability, customer risk classification, business recommendations, interactive analytics, model evaluation visualizations, and a downloadable prediction report.

---

## Live Application

**Try the deployed Streamlit app:**

[Open Customer Churn Prediction App](https://customer-churn-prediction-z8vtrr2kbxca4yfmeqihmj.streamlit.app/)

No installation is required to access the live version.

---

## Quick Start

### Option 1: Use the Live App — Recommended

No installation is required.

Open the application:

[Launch Customer Churn Prediction App](https://customer-churn-prediction-z8vtrr2kbxca4yfmeqihmj.streamlit.app/)

Then:

1. Enter the customer information in the sidebar.
2. Click **Predict Churn**.
3. View the prediction and churn probability.
4. Check the customer's risk level.
5. Review the recommended retention actions.
6. Download the prediction report as a CSV file.
7. Explore the analytics and model evaluation sections.

### Option 2: Run the Project Locally

Clone the repository:

```bash
git clone https://github.com/raosanjana07/Customer-Churn-Prediction.git
```

Open the project folder:

```bash
cd Customer-Churn-Prediction
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment on Windows:

```bash
venv\Scripts\activate
```

Activate the virtual environment on macOS or Linux:

```bash
source venv/bin/activate
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

The application will normally open at:

```text
http://localhost:8501
```

---

## Project Overview

Customer churn occurs when a customer stops using a company's services.

Predicting churn can help businesses identify customers who may leave and take action before the customer cancels their service.

This application analyzes customer information such as:

- Contract type
- Customer tenure
- Monthly charges
- Total charges
- Internet service
- Payment method
- Technical support
- Online security
- Streaming services
- Billing preferences
- Customer demographics

The trained model estimates the probability that a customer will churn and classifies the customer into one of three risk levels:

- **Low Risk**
- **Medium Risk**
- **High Risk**

The application also provides recommended business actions based on the predicted churn risk.

---

## Main Features

- Predicts whether a customer is likely to churn
- Displays churn probability
- Classifies customers by churn risk
- Provides retention recommendations
- Shows customer analytics
- Displays important model features
- Presents model evaluation metrics
- Includes an interactive confusion matrix
- Includes an ROC curve
- Allows users to download prediction results as a CSV report
- Provides an online deployed version using Streamlit Community Cloud

---

## Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 78.46% |
| Precision | 62.54% |
| Recall | 47.33% |
| F1 Score | 53.88% |
| ROC-AUC | 81.53% |

### Metric Interpretation

**Accuracy**

The model correctly classified approximately 78.46% of the test customers.

**Precision**

When the model predicted that a customer would churn, it was correct approximately 62.54% of the time.

**Recall**

The model identified approximately 47.33% of the customers who actually churned.

**F1 Score**

The F1 score provides a balance between precision and recall.

**ROC-AUC**

The ROC-AUC score of 81.53% shows that the model has a good ability to distinguish between customers who churn and customers who stay.

---

## Machine Learning Workflow

1. Loaded the telecom customer churn dataset
2. Cleaned and prepared the data
3. Removed the `customerID` identifier column
4. Converted categorical variables using one-hot encoding
5. Separated the features and target variable
6. Split the data into training and testing sets
7. Scaled the input features using `StandardScaler`
8. Trained a `RandomForestClassifier`
9. Evaluated the model using classification metrics
10. Saved the trained model and preprocessing files
11. Built an interactive Streamlit web application
12. Added visualizations using Plotly
13. Deployed the project using Streamlit Community Cloud

---

## Technology Stack

### Programming Language

- Python

### Data Processing

- Pandas
- NumPy

### Machine Learning

- Scikit-learn
- Random Forest Classifier
- StandardScaler
- One-Hot Encoding

### Visualization

- Plotly
- Streamlit charts

### Model Storage

- Joblib

### Development and Deployment

- VS Code
- Git
- GitHub
- Streamlit
- Streamlit Community Cloud

---

## Project Structure

```text
Customer-Churn-Prediction/
│
├── app.py
├── train.py
├── main.py
├── feature_list.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── cleaned_churn.csv
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/
│   ├── churn_model.pkl
│   ├── scaler.pkl
│   ├── feature_columns.pkl
│   └── model_metrics.pkl
│
└── utils/
    ├── model_loader.py
    └── prediction.py
```

---

## How to Access the Project

### Access the GitHub Repository

The source code is available in this repository:

```text
https://github.com/raosanjana07/Customer-Churn-Prediction
```

Visitors can inspect:

- Data preprocessing code
- Model training code
- Prediction logic
- Streamlit application code
- Saved model files
- Model evaluation results
- Project documentation

### Access the Live Application

Open:

[Customer Churn Prediction Web App](https://customer-churn-prediction-z8vtrr2kbxca4yfmeqihmj.streamlit.app/)

No local installation is required.

### Download the Source Code

GitHub users can download the project by:

1. Opening the repository
2. Clicking the green **Code** button
3. Selecting **Download ZIP**
4. Extracting the folder
5. Following the local setup instructions in this README

They can also clone the repository using:

```bash
git clone https://github.com/raosanjana07/Customer-Churn-Prediction.git
```

---

## How to Use the Application

1. Open the live application or run it locally.
2. Use the sidebar to enter customer details.
3. Provide personal information such as gender and senior citizen status.
4. Select the customer's services.
5. Select the contract and payment method.
6. Enter tenure, monthly charges, and total charges.
7. Click **Predict Churn**.
8. Review the prediction result.
9. Check the churn probability.
10. Check whether the customer is Low, Medium, or High Risk.
11. Review the suggested retention actions.
12. Download the prediction report if required.
13. Explore the dashboard analytics and model evaluation charts.

---

## Customer Information Used by the Model

The application collects the following customer information:

### Personal Information

- Gender
- Senior citizen status
- Partner status
- Dependent status

### Service Information

- Phone service
- Multiple lines
- Internet service
- Online security
- Online backup
- Device protection
- Technical support
- Streaming TV
- Streaming movies

### Contract and Billing Information

- Contract type
- Paperless billing
- Payment method

### Numerical Information

- Tenure
- Monthly charges
- Total charges

---

## Important Model Features

The Random Forest model evaluates features such as:

- Contract duration
- Customer tenure
- Monthly charges
- Total charges
- Internet service type
- Payment method
- Technical support
- Online security
- Paperless billing

These features help the model estimate the likelihood that a customer will churn.

---

## Prediction Risk Levels

### Low Risk

A customer is classified as Low Risk when the predicted churn probability is below 30%.

Recommended actions may include:

- Maintaining current service quality
- Offering loyalty rewards
- Suggesting premium plans

### Medium Risk

A customer is classified as Medium Risk when the churn probability is between 30% and 70%.

Recommended actions may include:

- Sending promotional offers
- Checking customer satisfaction
- Offering contract renewal discounts

### High Risk

A customer is classified as High Risk when the churn probability is above 70%.

Recommended actions may include:

- Contacting the customer immediately
- Offering personalized discounts
- Escalating the customer to the retention team

---

## Model Evaluation Visualizations

The application includes:

### Feature Importance

Shows which customer features have the greatest influence on the Random Forest model.

### Confusion Matrix

Shows:

- Customers correctly predicted to stay
- Customers correctly predicted to churn
- Customers incorrectly predicted to stay
- Customers incorrectly predicted to churn

### ROC Curve

Displays the relationship between:

- True Positive Rate
- False Positive Rate

The ROC-AUC score summarizes the model's ability to separate churners from non-churners.

---

## Challenges and Solutions

### Consistent Training and Prediction Features

**Challenge**

The model was trained using one-hot encoded columns, while the Streamlit form accepts the original customer categories.

**Solution**

The feature column names were saved during model training and reused during prediction. Customer inputs are converted into the same feature format expected by the trained model.

### Handling Decimal Charge Values

**Challenge**

Monthly charges and total charges contain decimal values, but the initial prediction DataFrame used integer data types.

**Solution**

The prediction input DataFrame was explicitly created using the `float64` data type so that decimal values could be processed correctly.

### Model File Paths During Deployment

**Challenge**

The deployed application initially could not locate the trained model files.

**Solution**

The model files were organized inside the `models` directory, and the application was updated to use the correct relative paths.

### Reusing Model Files Efficiently

**Challenge**

Loading the model every time the Streamlit application reran could reduce efficiency.

**Solution**

Streamlit resource caching was used to load and reuse the model, scaler, and feature columns.

### Prediction Input Compatibility

**Challenge**

The model required exactly the same feature order used during training.

**Solution**

The saved `feature_columns.pkl` file is used to create the prediction input DataFrame in the correct order.

---

## What I Learned

Through this project, I gained practical experience in:

- Data cleaning and preprocessing
- Working with categorical data
- One-hot encoding
- Feature scaling
- Classification model development
- Random Forest models
- Training and testing data splits
- Model evaluation
- Accuracy, precision, recall, and F1 score
- Confusion matrix interpretation
- ROC curve and ROC-AUC analysis
- Building Streamlit web applications
- Creating interactive Plotly visualizations
- Saving and loading machine learning artifacts
- Organizing a machine learning project
- Git and GitHub version control
- Debugging deployment issues
- Deploying machine learning applications online

---

## Future Improvements

- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- Cross-validation
- Improving recall for churn customers
- Comparing multiple machine learning algorithms
- Adding Logistic Regression and XGBoost models
- Adding SHAP-based model explanations
- Supporting batch prediction with uploaded CSV files
- Saving prediction history
- Adding a database
- Adding user authentication
- Automating model retraining
- Adding tests for prediction functions

---

## Interview Explanation

A concise way to explain this project in an interview:

> I developed and deployed an end-to-end customer churn prediction web application using Python, scikit-learn, Random Forest, Streamlit, and Plotly. I cleaned and encoded the telecom customer data, trained and evaluated the classification model, saved the model and preprocessing objects, and built an interactive interface where users can enter customer information and receive a churn prediction, probability, risk level, and retention recommendation. I also added feature importance, customer analytics, model metrics, a confusion matrix, an ROC curve, and a downloadable prediction report. The final application was deployed using Streamlit Community Cloud and managed using Git and GitHub.

---

## Repository Highlights

This repository demonstrates:

- End-to-end machine learning development
- Data preprocessing
- Classification
- Model evaluation
- Interactive dashboard development
- Business-focused recommendations
- Clean project organization
- Version control
- Cloud deployment

---

## Author

**Sanjana Rao**

BSc Data Science student interested in machine learning, data analytics, and building practical data-driven applications.

- GitHub: [raosanjana07](https://github.com/raosanjana07)
- Live App: [Customer Churn Prediction](https://customer-churn-prediction-z8vtrr2kbxca4yfmeqihmj.streamlit.app/)