import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import r2_score
import pickle


dataset = pd.read_csv("../data/50_Startups.csv")
dataset = pd.get_dummies(dataset, drop_first=True)
independent = dataset[['R&D Spend', 'Administration', 'Marketing Spend','State_Florida', 'State_New York']]
dependent = dataset[['Profit']]
x_train,x_test,y_train,y_test = train_test_split(independent, dependent, test_size=0.30, random_state=0)
criterions = ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
splitters = ['best','random']

best_r_score = float('-inf')  # Initialize with a very low value
best_model = None
best_params = {}

for criterion in criterions:
    for splitter in splitters:
        try:
            regressor = DecisionTreeRegressor(criterion=criterion, splitter=splitter)
            regressor.fit(x_train,y_train)
            #tree.plot_tree(regressor)
            #plt.show()
            y_predict = regressor.predict(x_test)
            r_score = r2_score(y_test,y_predict)
            print(f'Criterion: {criterion}, splitter: {splitter}, R_Score: {r_score:.4f}')
            # Update the best model if the current one is better
            if r_score > best_r_score:
                best_r_score = r_score
                best_model = regressor
                best_params = {'criterion': criterion, 'splitter': splitter}
        except Exception as e:
            print(f'Criterion: {criterion}, splitter: {splitter}, Error: {str(e)}')

# Plotting the best decision tree
if best_model:
    tree.plot_tree(best_model)
    plt.title(f"Best Model: Criterion={best_params['criterion']}, Splitter={best_params['splitter']}")
    plt.show()

# Save the best model
filename = "best_decision_tree_model.sav"
pickle.dump(best_model, open(filename, 'wb'))
print(f"\nBest Model Saved: {best_params}, R_Score: {best_r_score:.4f}")

# Load and use the best model to make predictions
load_model = pickle.load(open("best_decision_tree_model.sav", 'rb'))
result = load_model.predict([[1234, 123, 4565, 0, 0]])
print(f"Prediction for input [1234, 123, 4565, 0, 0]: {result[0]:.2f}")


