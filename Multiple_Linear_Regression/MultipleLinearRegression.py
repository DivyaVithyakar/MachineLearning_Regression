# Import required libraries
import pandas as pd  # For data manipulation and reading CSV files
from sklearn.model_selection import train_test_split  # For splitting data into train and test sets
from sklearn.linear_model import LinearRegression  # To create and train a linear regression model
from sklearn.metrics import r2_score  # To evaluate the model's performance
import pickle  # To save and load the trained model

# Read the dataset from CSV file
dataset = pd.read_csv("../data/50_Startups.csv")

# Convert categorical variables into dummy variables (one-hot encoding) and drop the first column to avoid multicollinearity
dataset = pd.get_dummies(dataset, drop_first=True)

# Split the dataset into independent variables  and dependent variable
independent = dataset[['R&D Spend', 'Administration', 'Marketing Spend', 'State_Florida', 'State_New York']]
dependent = dataset[['Profit']]

# Split the dataset into training and testing sets (70% train, 30% test)
x_train, x_test, y_train, y_test = train_test_split(independent, dependent, test_size=0.30, random_state=0)

# Create a Linear Regression model
regressor = LinearRegression()

# Train the model using the training dataset
regressor.fit(x_train, y_train)

# Extract the coefficients (weights) and the intercept (bias) from the trained model
weight = regressor.coef_
bias = regressor.intercept_

# Make predictions on the test dataset
y_predict = regressor.predict(x_test)

# Evaluate the model's performance using R² score
r_score = r2_score(y_test, y_predict)

# Save the trained model to a file using pickle
filename = "finalized_multiple_profit.sav"
pickle.dump(regressor, open(filename, 'wb'))

# Load the saved model from the file
load_model = pickle.load(open("finalized_multiple_profit.sav", 'rb'))

# Make a prediction for a new set of input values (user-provided data)
result = load_model.predict([[1234, 123, 4565, 0, 0]])

# Print the predicted result
print(result)
