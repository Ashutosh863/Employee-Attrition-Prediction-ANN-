Employee Attrition Prediction using Artificial Neural Networks (ANN)
📌 Project Overview
Employee attrition is a critical challenge for organizations, leading to increased hiring costs, loss of productivity, and disruption of team dynamics.
This project builds an Artificial Neural Network (ANN) using Keras to predict whether an employee is likely to leave the company, based on historical HR data.

The model helps HR teams proactively identify employees at risk of attrition and take data-driven retention actions.

🎯 Problem Statement
Predict employee attrition (Yes / No) using employee demographic, job, and performance-related features.

Type: Binary Classification

Target Variable: Attrition

1 → Employee will leave

0 → Employee will stay

📊 Dataset
Name: IBM HR Analytics Employee Attrition & Performance Dataset
Source: Kaggle


Dataset Details:

Rows: 1,470 employees
Features: 35 (numerical + categorical)
Missing Values: None



Project Workflow:

Raw HR Dataset
   ↓
Data Preprocessing
   • Categorical Encoding
   • Numerical Feature Scaling
   ↓
Train / Validation / Test Split
   ↓
Artificial Neural Network (ANN)
   ↓
Model Evaluation & Metrics

⚙️ Technologies & Tools

Python
TensorFlow / Keras
Pandas
Scikit-learn

🧩 Data Preprocessing
Converted target variable Attrition into binary format

Applied:
StandardScaler for numerical features
OneHotEncoder for categorical features
Used ColumnTransformer for clean and reproducible preprocessing

Performed stratified train–test split to handle class imbalance

🧠 Model Architecture (ANN)
Input Layer
↓
Dense Layer (64 neurons, ReLU)
↓
Dropout (0.3)
↓
Dense Layer (32 neurons, ReLU)
↓
Output Layer (1 neuron, Sigmoid)


Loss Function: Binary Cross-Entropy

Optimizer: Adam
Regularization: Dropout
Overfitting Control: Early Stopping

Model Performance:
Metric	Score
Training Accuracy	91%
Validation Accuracy	~90%
Test Accuracy	%86

✔ Training and test accuracies are close, indicating strong generalization
✔ Early stopping helped reduce overfitting and improve test performance

🧪 Evaluation Metrics:

Accuracy
Confusion Matrix
Precision, Recall, F1-score
Since employee attrition is an imbalanced problem, additional metrics such as recall and F1-score were analyzed instead of relying solely on accuracy.

Key Insights:

ANN effectively captures non-linear relationships between employee attributes
Early stopping significantly improves model generalization
Proper preprocessing of mixed data types is crucial for deep learning models
Accuracy alone is not sufficient for imbalanced classification problems


Install dependencies:
pip install -r requirements.txt


📌Future Enhancements:

Handle class imbalance using class weights or SMOTE
Compare ANN performance with Logistic Regression and tree-based models
Deploy the model using Streamlit or Flask
Add explainability using SHAP values
