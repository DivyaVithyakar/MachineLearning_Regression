import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

dataset = pd.read_csv("../data/insurance_pre.csv")
dataset = pd.get_dummies(dataset, drop_first=True)
print(dataset.columns)
independent = dataset[['age', 'bmi', 'children', 'sex_male', 'smoker_yes']]
dependent = dataset[['charges']]
x_train,x_test,y_train,y_test = train_test_split(independent, dependent, test_size=0.30, random_state=42)
regressor = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=3), n_estimators=50, learning_rate=1.5,  random_state=42)
regressor.fit(x_train,y_train)
accuracy = regressor.score(x_test, y_test)
#y_predict = regressor.predict(x_test)
#r_score = r2_score(y_test, y_predict)
print(accuracy)
#0.8457037182041467
