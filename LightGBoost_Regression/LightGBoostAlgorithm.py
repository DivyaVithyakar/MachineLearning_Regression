import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score


dataset = pd.read_csv("../data/insurance_pre.csv")
dataset = pd.get_dummies(dataset, drop_first=True)
print(dataset.columns)
independent = dataset[['age', 'bmi', 'children', 'sex_male', 'smoker_yes']]
dependent = dataset[['charges']]
x_train,x_test,y_train,y_test = train_test_split(independent, dependent, test_size=0.30, random_state=42)
regressor = LGBMRegressor(boosting_type= 'gbdt',objective= 'regression',metric= 'rmse',num_leaves= 31,learning_rate= 0.05,feature_fraction= 0.9,num_boost_round=100)
regressor.fit(x_train,y_train)
y_predict = regressor.predict(x_test)
r_score = r2_score(y_test, y_predict)
print(r_score)
