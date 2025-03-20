import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pickle

dataset = pd.read_csv("../data/50_Startups.csv")
dataset = pd.get_dummies(dataset, drop_first=True)
independent = dataset[['R&D Spend', 'Administration', 'Marketing Spend','State_Florida', 'State_New York']]
dependent = dataset[['Profit']]
x_train,x_test,y_train,y_test = train_test_split(independent, dependent, test_size=0.30, random_state=0)
n_estimator = [10,50,100]
criterions = ['squared_error', 'absolute_error', 'friedman_mse', 'poisson']

best_r_score = float('-inf')  # Initialize with a very low value
best_model = None
best_params = {}

for criterion in criterions:
    for n_estimators in n_estimator:
        try:
            # Train the model
            regressor = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion)
            regressor.fit(x_train,y_train.values.ravel())
            y_predict = regressor.predict(x_test)
            r_score = r2_score(y_test,y_predict)
            print(f'Criterion: {criterion}, n_estimators: {n_estimators}, R_Score: {r_score:.4f}')

            # Update the best model if the current one is better
            if r_score > best_r_score:
                best_r_score = r_score
                best_model = regressor
                best_params = {'criterion': criterion, 'n_estimators': n_estimators}
        except Exception as e:
            print(f'Criterion: {criterion}, n_estimators: {n_estimators}, Error: {str(e)}')

filename = "finalised_randomForest_model.sav"
pickle.dump(best_model, open(filename, 'wb'))
print(f"\nBest Model Saved: {best_params}, R_Score: {best_r_score:.4f}")

load_model = pickle.load(open("finalised_randomForest_model.sav", 'rb'))
result = load_model.predict([[1234, 123, 4565, 0, 0]])
print(result)
