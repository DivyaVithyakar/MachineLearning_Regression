import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
import xgboost as xgb
import pickle

# Load dataset
dataset = pd.read_csv("../data/50_Startups.csv")

# One-hot encoding
dataset = pd.get_dummies(dataset, drop_first=True)

# Define independent (X) and dependent (y) variables
X = dataset[['R&D Spend', 'Administration', 'Marketing Spend', 'State_Florida', 'State_New York']]
y = dataset['Profit']  # Already a 1D array

# Split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Define the XGBoost regressor
xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror')

# Define the hyperparameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

# Setup GridSearchCV
grid = GridSearchCV(estimator=xgb_regressor, param_grid=param_grid, cv=5, refit=True, verbose=3, n_jobs=-1)

# Fit GridSearchCV
grid.fit(x_train, y_train)

# Print best hyperparameters found by GridSearchCV
print("Best Parameters:", grid.best_params_)

#  Evaluate the best model on the TEST data
best_model = grid.best_estimator_
y_pred = best_model.predict(x_test)

# Calculate R2 score for the test data
r2 = r2_score(y_test, y_pred)
print(f"Final R2 Score on Test Data: {r2:.4f}")

# Store the best model using pickle
filename = "finalised_xgboost_model.sav"
pickle.dump(best_model, open(filename, 'wb'))
