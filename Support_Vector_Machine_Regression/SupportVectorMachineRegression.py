import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import  SVR
from sklearn.metrics import r2_score


dataset = pd.read_csv("50_Startups.csv")
dataset = pd.get_dummies(dataset,drop_first=True)
independent = dataset[['R&D Spend', 'Administration', 'Marketing Spend','State_Florida', 'State_New York']]
dependent = dataset[['Profit']]
x_train,x_test,y_train,y_test = train_test_split(independent, dependent, test_size=0.30, random_state=0)
kernels = ['linear', 'rbf','poly', 'sigmoid']
c_values = [1,10,100]
for kernel in kernels:
    for c in c_values:
        try:
            regressor = SVR(kernel = kernel,C=c)
            regressor.fit(x_train, y_train.values.ravel())
            y_predict = regressor.predict((x_test))
            r_score = r2_score(y_test,y_predict)
            print(f'Kernel: {kernel},C: {c}, R_Score: {r_score:.4f}')
        except Exception as e:
            print(f'Kernel: {kernel}, C: {c}, Error: {str(e)}')





