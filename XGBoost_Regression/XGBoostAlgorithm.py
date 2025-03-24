import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

dataset = pd.read_csv("../data/insurance_pre.csv")
dataset = pd.get_dummies(dataset, drop_first=True)
print(dataset.columns)
independent = dataset[['age', 'bmi', 'children', 'sex_male', 'smoker_yes']]
dependent = dataset[['charges']]
x_train,x_test,y_train,y_test = train_test_split(independent, dependent, test_size=0.30, random_state=42)
regressor = XGBRegressor(objective= 'reg=squarederror',learning_rate= 0.1,max_depth= 3,n_estimators= 100,subsample= 0.8,colsample_bytree= 0.8,reg_alpha= 0.1,reg_lambda= 0.1)
regressor.fit(x_train,y_train)
y_predict = regressor.y_predict(x_test)
r_score = r2_score(y_test,y_predict)
print(r_score)
