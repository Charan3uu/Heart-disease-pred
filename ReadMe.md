# Heart Disease Prediction using Machine Learning

This project is a Jupyter Notebook for predicting the presence of heart disease using various Machine Learning algorithms. The notebook demonstrates the end-to-end process of data analysis, preprocessing, model training, and evaluation.

## Project Overview

Heart disease is a leading cause of mortality worldwide. Early detection through predictive analytics can help in timely treatment and prevention. In this project, we use a public heart disease dataset and apply three different machine learning models to predict whether a patient has heart disease or not.

## Features

- Data analysis and visualization
- Data preprocessing (handling missing values, encoding, scaling)
- Model training and evaluation
- Comparison of three ML models:
  - Random Forest (RF)
  - Logistic Regression (LR)
  - K-Nearest Neighbors (KNN)
- Performance metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix

## Dataset

The dataset contains several features including age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, restecg, maximum heart rate, exercise-induced angina, oldpeak, the slope of the peak exercise ST segment, number of major vessels, and thalassemia.

## Models Used

1. **Random Forest (RF):**
   - Ensemble method
   - Handles non-linearity and feature interactions well

2. **Logistic Regression (LR):**
   - Simple and interpretable baseline
   - Suitable for binary classification

3. **K-Nearest Neighbors (KNN):**
   - Instance-based learning
   - Makes predictions based on the closest data points

## How to Run

1. Clone this repository.
2. Install required dependencies (see below).
3. Open the Jupyter Notebook (`.ipynb` file) and run the cells sequentially.

## Requirements

- Python 3.x
- Jupyter Notebook
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Install dependencies using pip:

```bash
pip install jupyter pandas numpy matplotlib seaborn scikit-learn
```

## Usage

1. Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
2. Open the notebook file and follow the steps for:
   - Data loading & exploration
   - Preprocessing
   - Model training
   - Evaluation & comparison

## Results

The notebook presents the evaluation metrics for each of the models, helping you understand which algorithm performs best for this particular dataset.

## License

This project is for educational and research purposes.


---
Feel free to use or adapt this project for your own learning or research!
