# 📊 Machine Learning Regression 

**Machine Learning Regression** repository! 🚀 This project demonstrates the implementation of various regression models using Python and scikit-learn.

## 📌 Project Overview  
This repository contains hands-on implementations of regression models to predict values based on given features. The implemented models include:

- **Simple Linear Regression (SLR)** → Predicting salary based on years of experience.

- **Multiple Linear Regression (MLR)** → Predicting profit based on multiple features such as R&D Spend, Administration, Marketing Spend, and State.

- **Decision Tree Regression** → A non-linear regression approach using decision trees.

- **Random Forest Regressio**n → An ensemble learning technique that improves regression accuracy.

- **Support Vector Machine Regression (SVR)** → A powerful regression model based on Support Vector Machines.

- **Automated Model Selection** → A pipeline for selecting the best regression model for a given dataset.


---

## 🚀 Key Feature: Automated Model Selection  
This project includes an **Automated Model Selection** pipeline that evaluates multiple regression models and selects the best one based on performance metrics.  
This helps in finding the most accurate regression model for a given dataset with minimal manual effort.

---

## 🛠️ Tech Stack Used  

| Technology | Purpose |
|------------|---------|
| **Python** 🐍 | Programming language |
| **Pandas** 📊 | Data manipulation & CSV reading |
| **Scikit-learn** 🤖 | Machine Learning library |
| **Pickle** 📦 | Saving & loading models |
| **Matplotlib** 📈	| Data visualization |

---

## 📊 Models Implemented

1. **Automated Model Selection**

 - Objective: Automatically select the best regression model for a dataset.
 - How It Works: The pipeline evaluates multiple regression models and picks the one with the highest R² Score.
 - Benefit: Saves time and ensures the most suitable model is used for predictions.
 - Evaluation Metric: Compares R² Score across different models.
 - Model Storage: Pickle format.

2. **Simple Linear Regression (SLR)**
   
 - Objective: Predict salary based on years of experience using Simple Linear Regression.
 - Dataset: A CSV file with YearsExperience and Salary.
 - Evaluation Metric: R² Score (Goodness of fit for the model).
 - Model Storage: Pickle is used to save and load the trained model.

3. **Multiple Linear Regression (MLR)**
   
  - Objective: Predict profit based on multiple features such as R&D Spend, Administration, Marketing Spend, and categorical features like State using Multiple Linear Regression.
  - Dataset: A CSV file with features like R&D Spend, Administration, Marketing Spend, State, and Profit as the target.
  - Evaluation Metric: R² Score (Goodness of fit for the model).
  - Model Storage: Pickle is used to save and load the trained model.

4. **Decision Tree Regression**

 - Objective: Predict target values using a tree-based non-linear model.
 - Dataset: Works with various feature sets and target variables.
 - Evaluation Metric: R² Score.
 - Model Storage: Pickle format.

5. **Random Forest Regression**
   
 - Objective: Improve prediction accuracy by combining multiple decision trees.
 - Dataset: Works with various feature sets and target variables.
 - Evaluation Metric: R² Score.
 - Model Storage: Pickle format.

6. **Support Vector Machine Regression (SVR)**

 - Objective: Use Support Vector Machines to model complex regression relationships.
 - Dataset: Works with various feature sets and target variables.
 - Evaluation Metric: R² Score.
 - Model Storage: Pickle format.



## 📊 Model Evaluation
- Metric Used: R² Score (Goodness of fit for the model)
- Model Storage: pickle for saving and loading trained models.


## 📬 Reach Me

📧 Email: divyarbe@gmail.com

🔗 GitHub: github.com/DivyaVithyakar

🚀 LinkedIn: linkedin.com/in/divya-ramanujam-sdet

## 📃 Notes

**SLR** is focused on predicting salary based on years of experience.
**MLR** can handle multiple predictors and is used for predicting profit using more complex datasets.
Additional regression models like **Decision Tree, Random Forest, and SVR** enhance model diversity.
This repository will continue to expand with new implementations and improvements.

🔄 Stay tuned for more updates!
