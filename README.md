# 🩺 Breast Cancer Prediction using Machine Learning
OPEN THE FILE NAME: "Breast_cancer_predictor.ipnyb" for the main code for the ML model
## 📌 Project Overview
This project is part of the AI/ML Disease Prediction Toolkit. It demonstrates how machine learning can be applied in healthcare to assist in predicting whether a tumor is benign or malignant based on medical features.
Using the Breast Cancer Wisconsin (Diagnostic) Dataset from Kaggle, I built and compared multiple ML models (Logistic Regression, Random Forest, SVM, KNN). The project follows a complete pipeline from data preprocessing to model training, evaluation, and real-world prediction using user-provided data.
## 🎯 Objectives

Preprocess healthcare datasets (handle missing values, drop unnecessary columns, scale features).

Train and evaluate multiple ML models using scikit-learn.

Compare models with metrics such as accuracy, precision, recall, and F1-score.

Save the best-performing model and scaler for real-world predictions.

Create a user-friendly CSV template so anyone can input patient data and get predictions.

## ⚙️ Tech Stack

Python 3.12

Google Colab / Jupyter Notebook

Libraries: pandas, numpy, scikit-learn, matplotlib, joblib

## 📊 Workflow

Dataset Loading – Breast Cancer dataset from Kaggle.

Data Preprocessing – Dropped irrelevant columns (id, Unnamed:32), imputed missing values, scaled numeric features.

Model Training – Logistic Regression, Random Forest, SVM, and KNN.

Evaluation – Compared models using accuracy and classification reports.

Best Model – Random Forest achieved the highest accuracy.

Model Saving – Exported trained model (breast_rf_model.pkl) and scaler (breast_scaler.pkl).

User Prediction – Created Breast_user_template.csv (headers only) and Breast_user_sample.csv (sample patients). Users can input their own values and get predictions.

## 🚀 Results

Achieved ~95–97% accuracy (depending on random split).

Random Forest performed best in balancing precision and recall.

Final pipeline outputs predictions in a clear, tabular format with both class labels and probability of malignancy.

## 📂 Repository Structure

📦 Disease-Prediction-Toolkit
 ┣ 📜 BreastCancer_Prediction.ipynb   # Main Colab notebook
 ┣ 📜 breast_rf_model.pkl             # Trained Random Forest model
 ┣ 📜 breast_scaler.pkl               # Scaler for feature normalization
 ┣ 📜 Breast_user_template.csv        # Empty template for user input
 ┣ 📜 Breast_user_sample.csv          # Sample file with 4 patient rows
 ┣ 📜 requirements.txt                # Project dependencies
 ┗ 📜 README.md                       # Project documentation
## 🖥️ How to Run

Open the notebook in Google Colab.

Upload the dataset or use the Kaggle API.

Run all cells to train and save the model.

Use Breast_user_sample.csv (or fill in Breast_user_template.csv) to test predictions.

The pipeline outputs a prediction table with both class labels and probabilities.

## 🎥 Demo & Presentation

Demo Video → Shows training results and prediction outputs.

Slides → Explains dataset, preprocessing, model comparison, results, and conclusions.

## 📌 Future Improvements

Add support for more healthcare datasets (e.g., heart disease, diabetes).

Deploy the model using Gradio or Streamlit for a web-based UI.

Experiment with deep learning models for improved accuracy.
