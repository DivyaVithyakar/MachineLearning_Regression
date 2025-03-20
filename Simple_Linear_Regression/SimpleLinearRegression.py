import pandas as pd  # For data manipulation and reading CSV files
from sklearn.model_selection import train_test_split  # For splitting data into train and test sets
from sklearn.linear_model import LinearRegression  # To create and train a linear regression model
from sklearn.metrics import r2_score  # To evaluate the model's performance
import pickle  # To save and load the trained model

"""
Pseudocode
1. Import required libraries (pandas, sklearn, pickle)
2. Read the dataset from a CSV file
3. Split the dataset into:
   - Independent variables 
   - Dependent variable
4. Divide the dataset into training and testing sets (70% training, 30% testing)
5. Create a Linear Regression model
6. Train the model using the training dataset
7. Extract model coefficients (weight, bias)
8. Predict salaries using the test dataset
9. Evaluate model performance using R² score
10. Save the trained model using Pickle
11. Load the saved model from the file
12. Get user input (Years of Experience)
13. Predict the salary using the loaded model
14. Print the predicted salary
"""

# Read dataset from CSV
dataset = pd.read_csv("../data/Salary_Data.csv")

# Split independent and dependent variables
independent = dataset[["YearsExperience"]]
dependent = dataset[["Salary"]]

# Divide dataset into training and testing sets (70% train, 30% test)
x_train, x_test, y_train, y_test = train_test_split(independent, dependent, test_size=0.30, random_state=0)

# Create and train the Linear Regression model
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Extract model coefficients
weight = regressor.coef_
bias = regressor.intercept_

# Make predictions on the test dataset
y_predict = regressor.predict(x_test)

# Evaluate model performance using R² score
r_score = r2_score(y_test, y_predict)
print(f"Model R² Score: {r_score:.4f}")

# Save the trained model using Pickle
filename = "final_salary_prediction.sav"
pickle.dump(regressor, open(filename, 'wb'))

# Load the saved model
load_model = pickle.load(open(filename, 'rb'))

# Get user input and predict salary
user_data = int(input("Enter years of experience to predict salary: "))
result = load_model.predict([[user_data]])
print(f"Predicted Salary: {result[0][0]:.2f}")
