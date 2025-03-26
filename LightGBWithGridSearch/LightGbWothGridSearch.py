import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
import lightgbm as lgb
import pickle

# Load your dataset
dataset = pd.read_csv("../data/50_Startups.csv")

# One-hot encoding
dataset = pd.get_dummies(dataset, drop_first=True)

# Define independent (X) and dependent (y) variables
X = dataset[['R&D Spend', 'Administration', 'Marketing Spend', 'State_Florida', 'State_New York']]
y = dataset['Profit']  # Already a 1D array

# Split into train and test sets BEFORE GridSearchCV
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Define hyperparameter grid for LightGBM
param_grid = {
    'boosting_type': ['gbdt', 'dart', 'goss'],  # The type of boosting to use
    'num_leaves': [31, 50, 100],  # Number of leaves in one tree
    'learning_rate': [0.05, 0.1, 0.2],  # Step size at each iteration
    'n_estimators': [50, 100, 150],  # Number of boosting iterations
    'max_depth': [-1, 5, 10],  # Maximum depth of the tree
    'min_child_samples': [10, 20, 30],  # Minimum number of samples per leaf
    'subsample': [0.6, 0.8, 1.0],  # Fraction of samples to train each tree
    'colsample_bytree': [0.6, 0.8, 1.0]  # Fraction of features to use for each tree
}

# Initialize LightGBM Regressor
lgbm = lgb.LGBMRegressor()

# Perform GridSearchCV
grid_search = GridSearchCV(estimator=lgbm, param_grid=param_grid, cv=5, n_jobs=-1, verbose=3, refit=True)

# Fit the model to the training data
grid_search.fit(x_train, y_train)

# Print best hyperparameters found
print("Best Parameters found: ", grid_search.best_params_)

#Evaluate the best model on TEST DATA
best_model = grid_search.best_estimator_
y_pred = best_model.predict(x_test)

# Calculate R2 score for performance evaluation
r2 = r2_score(y_test, y_pred)
print(f"Final R2 Score on Test Data: {r2:.4f}")

# Save the best model
filename = "finalized_lightgbm_model_with_gridsearch.sav"
pickle.dump(best_model, open(filename, 'wb'))
