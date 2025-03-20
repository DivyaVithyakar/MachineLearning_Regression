import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import r2_score


dataset = pd.read_csv("50_Startups.csv")
dataset = pd.get_dummies(dataset, drop_first=True)
independent = dataset[['R&D Spend', 'Administration', 'Marketing Spend','State_Florida', 'State_New York']]
dependent = dataset[['Profit']]
x_train,x_test,y_train,y_test = train_test_split(independent, dependent, test_size=0.30, random_state=0)
criterions = ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
splitters = ['best','random']

for criterion in criterions:
    for splitter in splitters:
        try:
            regressor = DecisionTreeRegressor(criterion=criterion, splitter=splitter)
            regressor.fit(x_train,y_train)
            tree.plot_tree(regressor)
            plt.show()
            y_predict = regressor.predict(x_test)
            r_score = r2_score(y_test,y_predict)
            print(f'Criterion: {criterion}, splitter: {splitter}, R_Score: {r_score:.4f}')
        except Exception as e:
            print(f'Criterion: {criterion}, splitter: {splitter}, Error: {str(e)}')


