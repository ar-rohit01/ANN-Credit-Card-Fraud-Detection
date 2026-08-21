# Credit Card Fraud Detection using Artificial Neural Network (ANN)

An end-to-end Deep Learning project for detecting fraudulent credit card transactions using an Artificial Neural Network (ANN), with comparison against traditional Machine Learning algorithms.

---

## Project Overview

Credit card fraud detection is a binary classification problem where the objective is to identify fraudulent credit card transactions while minimizing false positives.

This project demonstrates an end-to-end Machine Learning and Deep Learning workflow including:

- Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- Handling class imbalance using SMOTE
- Feature scaling
- Train, validation and test splitting
- Building a Vanilla Artificial Neural Network
- Hyperparameter tuning using Keras Tuner
- Building a Tuned ANN
- Comparison with traditional Machine Learning algorithms
- Model evaluation using multiple classification metrics
- Saving the trained model and preprocessing artifacts

---

## Project Objectives

1. Understand and explore the credit card transaction dataset.
2. Perform data cleaning and preprocessing.
3. Analyze the class imbalance in the dataset.
4. Apply SMOTE to handle class imbalance.
5. Build a baseline Vanilla ANN.
6. Fine-tune the ANN architecture using Keras Tuner.
7. Compare ANN models with traditional Machine Learning algorithms.
8. Evaluate the models using appropriate classification metrics.
9. Identify the best-performing model.
10. Save the final trained model and supporting preprocessing files.

---

## Dataset

The project uses the Credit Card Fraud Detection dataset.

The target variable is `Class`:

- `0` → Genuine transaction
- `1` → Fraudulent transaction

### Main Features

- `Time`
- `V1` to `V28`
- `Amount`
- `Class`

The `V1`–`V28` features are anonymized/PCA-transformed features.

> Note: The original dataset is not included in this GitHub repository. Please obtain the dataset from its original source and place the required CSV file inside the `dataset/` directory.

---

## Exploratory Data Analysis

The project performs both non-visual and visual analysis.

### Non-Visual Analysis

- Dataset shape
- Data types
- Missing values
- Duplicate records
- Descriptive statistics
- Target variable distribution
- Unique values
- Class imbalance

### Visual Analysis

- Class distribution
- Transaction amount distribution
- Transaction amount by class
- Transaction time distribution
- Feature correlation

### Key EDA Observations

- The dataset contains a severe class imbalance.
- Fraudulent transactions represent a very small proportion of all transactions.
- Transaction amounts are highly skewed.
- The `Time` feature provides information about transaction timing.
- The anonymized `V1`–`V28` features have different relationships with the target variable.

---

## Data Preprocessing

The preprocessing workflow includes:

1. Data cleaning
2. Handling missing target values
3. Removing duplicate records
4. Separating features and target
5. Stratified train-validation-test split
6. Feature scaling
7. Handling class imbalance using SMOTE
8. Preparing the processed data for model training

The dataset is divided into:

- Training set
- Validation set
- Test set

The test set is kept separate for final model evaluation.

---

## Handling Class Imbalance using SMOTE

The dataset contains a highly imbalanced target variable.

To address this issue, SMOTE (Synthetic Minority Over-sampling Technique) was applied to the training data.

SMOTE generates synthetic samples for the minority class instead of simply duplicating existing observations.

> SMOTE was applied only to the training data to avoid data leakage into the validation and test sets.

---

## Vanilla Artificial Neural Network

A baseline Artificial Neural Network was developed using TensorFlow and Keras.

### Architecture

```text
Input Layer
     ↓
Dense Layer - 64 Neurons
     ↓
ReLU Activation
     ↓
Dropout - 30%
     ↓
Dense Layer - 32 Neurons
     ↓
ReLU Activation
     ↓
Sigmoid Output Layer
```

### Training Configuration

- Optimizer: Adam
- Loss Function: Binary Cross-Entropy
- Output Activation: Sigmoid
- Early Stopping
- Validation Monitoring

### Vanilla ANN Results

| Metric | Score |
|---|---:|
| Precision | 0.182 |
| Recall | 0.926 |
| F1-score | 0.304 |
| ROC-AUC | ~0.997 |
| PR-AUC | ~0.731 |

The Vanilla ANN achieved high recall but relatively low precision, resulting in a larger number of false positives.

---

## ANN Fine-Tuning using Keras Tuner

Keras Tuner was used to search for a better-performing ANN architecture.

### Hyperparameters Tuned

- Number of hidden layers
- Number of neurons
- Dropout rate
- Learning rate

### Selected Tuned ANN Architecture

```text
Input Layer
     ↓
Dense Layer - 32 Neurons
     ↓
Dropout - 20%
     ↓
Dense Layer - 64 Neurons
     ↓
Sigmoid Output Layer
```

### Tuned ANN Results

| Metric | Score |
|---|---:|
| Precision | 0.806 |
| Recall | 0.735 |
| F1-score | 0.769 |
| ROC-AUC | 0.994 |
| PR-AUC | 0.738 |

The Tuned ANN provided a much better balance between precision and recall compared with the Vanilla ANN.

---

## ANN vs Traditional Machine Learning

The following models were evaluated:

1. Logistic Regression
2. Random Forest
3. XGBoost
4. Vanilla ANN
5. Tuned ANN

### Model Comparison

| Model | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Logistic Regression | 0.069 | 0.912 | 0.127 |
| Vanilla ANN | 0.182 | 0.926 | 0.304 |
| Tuned ANN | 0.806 | 0.735 | 0.769 |
| XGBoost | 0.792 | 0.897 | 0.841 |
| Random Forest | 0.905 | 0.838 | 0.870 |

### Best Performing Model

Random Forest achieved the highest F1-score of **0.870** among the compared models.

The Tuned ANN achieved an F1-score of **0.769**, showing a significant improvement over the Vanilla ANN.

---

## Model Evaluation

Because fraud detection is a highly imbalanced classification problem, accuracy alone is not sufficient for evaluating model performance.

The following metrics were used:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve

### Precision

Measures how many transactions predicted as fraud were actually fraudulent.

### Recall

Measures how many actual fraudulent transactions were successfully detected.

### F1-score

Provides a balance between Precision and Recall.

### ROC-AUC

Measures the model's ability to distinguish between the two classes.

### PR-AUC

Provides useful evaluation information for highly imbalanced classification problems.

---

## Project Visualizations

### Exploratory Data Analysis

- Class distribution
- Transaction amount distribution
- Transaction amount by class
- Transaction time distribution
- Correlation heatmap

### Model Development and Evaluation

- Vanilla ANN confusion matrix
- SMOTE before/after comparison
- Tuned ANN learning curve
- Model F1-score comparison
- ROC curve
- Precision-Recall curve
- Tuned ANN confusion matrix

All visualization files are available inside the `images/` directory.

---

## Saved Model and Supporting Files

The final trained model and supporting preprocessing artifacts are included in the `model/` directory.

```text
model/
├── credit_card_fraud_tuned_ann.keras
├── credit_card_fraud_scaler.pkl
└── credit_card_fraud_threshold.txt
```

### File Description

**credit_card_fraud_tuned_ann.keras**

The final trained Tuned ANN model.

**credit_card_fraud_scaler.pkl**

The feature scaler used during preprocessing.

**credit_card_fraud_threshold.txt**

The classification threshold used by the project.

---

## Project Structure

```text
ANN-Credit-Card-Fraud-Detection/
│
├── README.md
├── requirements.txt
├── ANN_Credit_Card_Fraud_Detection.ipynb
│
├── dataset/
│   └── creditcard.csv
│
├── images/
│   ├── class_distribution.png
│   ├── correlation_heatmap.png
│   ├── model_comparison_f1.png
│   ├── precision_recall_curve.png
│   ├── roc_curve.png
│   ├── smote_before_after.png
│   ├── transaction_amount_by_class.png
│   ├── transaction_amount_distribution.png
│   ├── transaction_time_distribution.png
│   ├── tuned_ann_confusion_matrix.png
│   ├── tuned_ann_loss_curve.png
│   └── vanilla_ann_confusion_matrix.png
│
├── model/
│   ├── credit_card_fraud_tuned_ann.keras
│   ├── credit_card_fraud_scaler.pkl
│   └── credit_card_fraud_threshold.txt
│
└── results/
    └── model_comparison_results.csv
```

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow
- Keras
- Keras Tuner
- Imbalanced-learn
- XGBoost
- Google Colab

---

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/ar-rohit01/ANN-Credit-Card-Fraud-Detection.git
```

### 2. Navigate to the Project Directory

```bash
cd ANN-Credit-Card-Fraud-Detection
```

### 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 4. Add the Dataset

Download the Credit Card Fraud Detection dataset from its original source.

Place the dataset inside:

```text
dataset/
```

The expected dataset file is:

```text
creditcard.csv
```

### 5. Open the Notebook

Open:

```text
ANN_Credit_Card_Fraud_Detection.ipynb
```

The notebook can be run using:

- Google Colab
- Jupyter Notebook
- JupyterLab

Run the cells sequentially.

---

## Key Findings

- The dataset suffers from severe class imbalance.
- SMOTE was used to improve minority-class representation during training.
- The Vanilla ANN achieved very high recall but comparatively low precision.
- Keras Tuner improved the ANN architecture and precision-recall balance.
- The Tuned ANN achieved an F1-score of **0.769**.
- Traditional Machine Learning models performed strongly on this dataset.
- Random Forest achieved the highest F1-score of **0.870** among the evaluated models.
- Fraud detection requires careful consideration of Precision, Recall and F1-score rather than relying only on accuracy.

---

## Future Scope

- Cost-sensitive learning
- Advanced anomaly detection techniques
- Further hyperparameter optimization
- Threshold optimization based on business requirements
- Handling concept drift
- Real-time fraud detection
- Model monitoring and periodic retraining
- Deployment as an API or web application

---

## Limitations

- The V1–V28 features are anonymized/PCA-transformed features, limiting direct business interpretation.
- Fraud detection performance can change when transaction patterns change over time.
- The project is developed as an academic/project implementation and is not a production fraud detection system.
- The original dataset is not included in the GitHub repository.

---

# Author

## Rohit Rajaram Yadav

**B.Tech in Automation & Robotics**

### Areas of Interest

- Artificial Intelligence
- Machine Learning
- Deep Learning
- Data Science
- Computer Vision
- Intelligent Automation

### Connect With Me

- LinkedIn: https://www.linkedin.com/in/rohit-yadav-6b8999299
- GitHub: https://github.com/ar-rohit01

---

## Project Summary

This project demonstrates an end-to-end approach to credit card fraud detection using Artificial Neural Networks and traditional Machine Learning algorithms.

The complete workflow includes:

**Data Exploration → Data Preprocessing → Class Imbalance Handling → Vanilla ANN → Keras Tuner → Tuned ANN → Machine Learning Comparison → Model Evaluation → Final Analysis**

The project demonstrates the practical implementation of Deep Learning techniques for an imbalanced binary classification problem.
