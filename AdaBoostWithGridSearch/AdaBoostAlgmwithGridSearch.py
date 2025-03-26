import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import pickle

# Load dataset
dataset = pd.read_csv("../data/insurance_pre.csv")
dataset = pd.get_dummies(dataset, drop_first=True)

# Define independent (X) and dependent (y) variables
independent = dataset[['age', 'bmi', 'children', 'sex_male', 'smoker_yes']]
dependent = dataset[['charges']]

# Split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(independent, dependent, test_size=0.30, random_state=42)

#Define hyperparameter grid for Decision Tree (base estimator)
param_grid_tree = {
    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    'splitter': ['best', 'random'],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Run GridSearchCV for DecisionTreeRegressor
grid_tree = GridSearchCV(DecisionTreeRegressor(), param_grid_tree, cv=5, refit=True, verbose=3, n_jobs=-1)
grid_tree.fit(x_train, y_train)

# Print the best parameters for the base model (DecisionTree)
print("Best Parameters for DecisionTree:", grid_tree.best_params_)

# Create the AdaBoostRegressor using the best DecisionTree from GridSearch
best_tree = grid_tree.best_estimator_

# Define AdaBoost model with the best DecisionTree as base estimator
ada_regressor = AdaBoostRegressor(estimator=best_tree, n_estimators=50, learning_rate=1.5, random_state=42)

#Fit AdaBoostRegressor to the training data
ada_regressor.fit(x_train, y_train)

#Evaluate on test data
accuracy = ada_regressor.score(x_test, y_test)
print(f"AdaBoost Model Accuracy: {accuracy:.4f}")

#Evaluate R2 Score for the AdaBoost model
y_pred = ada_regressor.predict(x_test)
r2 = r2_score(y_test, y_pred)
print(f"R2 Score for AdaBoost Model: {r2:.4f}")

#Save the final AdaBoost model
filename = "finalised_AdaBoost_with_GridSearch_model.sav"
pickle.dump(ada_regressor, open(filename, 'wb'))
