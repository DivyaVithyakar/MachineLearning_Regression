import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
import pickle

# Load dataset
dataset = pd.read_csv("../data/50_Startups.csv")

# One-hot encoding for categorical variables
dataset = pd.get_dummies(dataset, drop_first=True)

# Define independent (X) and dependent (y) variables
X = dataset[['R&D Spend', 'Administration', 'Marketing Spend', 'State_Florida', 'State_New York']]
y = dataset['Profit']  # 1D array

#  Split into train and test sets BEFORE tuning
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Define hyperparameter grid for Random Forest
param_grid = {
    'n_estimators': [10, 50, 100, 200],  # Number of trees
    'criterion': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
    'max_depth': [5, 10, 15, None],  # Depth of trees
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4]  # Minimum samples per leaf
}

#  Run GridSearchCV on TRAINING DATA ONLY
grid = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, refit=True, verbose=3, n_jobs=-1)
grid.fit(x_train, y_train)

# Print best hyperparameters
print("Best Parameters:", grid.best_params_)

# Step 5: Evaluate the best model on TEST DATA
best_model = grid.best_estimator_
y_pred = best_model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print(f"Final R2 Score on Test Data: {r2:.4f}")

filename = "finalised_randomForestGridSearch_model.sav"
pickle.dump(best_model, open(filename, 'wb'))
