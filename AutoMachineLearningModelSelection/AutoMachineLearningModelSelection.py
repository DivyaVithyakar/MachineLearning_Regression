import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pickle
import matplotlib.pyplot as plt
from sklearn import tree

class BestModelSelector:
    def __init__(self, dataset_path):
        self.dataset = pd.read_csv(dataset_path)
        self.dataset = pd.get_dummies(self.dataset, drop_first=True)
        self.independent = self.dataset[['age', 'bmi', 'children', 'sex_male', 'smoker_yes']]
        self.dependent = self.dataset[['charges']]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.independent, self.dependent, test_size=0.30, random_state=0)
        self.best_model = None
        self.best_r_score = -float('inf')

    def train_and_evaluate(self, regressor, model_name, params=None):
        # Train the model and evaluate its performance
        if params:
            regressor.set_params(**params)
        regressor.fit(self.x_train, self.y_train)
        y_predict = regressor.predict(self.x_test)
        r_score = r2_score(self.y_test, y_predict)
        print(f'{model_name} - R_Score: {r_score:.4f}')

        # If it's a DecisionTreeRegressor, plot the tree
        if isinstance(regressor, DecisionTreeRegressor):
          tree.plot_tree(regressor)
          plt.show()

        # Save the best model based on R2 score
        if r_score > self.best_r_score:
            self.best_r_score = r_score
            self.best_model = regressor

        return r_score

    def select_best_model(self):
        # Linear Regression Model (Baseline)
        print("Evaluating Linear Regression...")
        regressor = LinearRegression()
        self.train_and_evaluate(regressor, 'Linear Regression')

        # Support Vector Machine Model (SVM)
        print("\nEvaluating Support Vector Machine (SVM)...")
        kernels = ['linear', 'rbf', 'poly', 'sigmoid']
        c_values = [1, 10, 100]
        for kernel in kernels:
            for c in c_values:
                params = {'kernel': kernel, 'C': c}
                self.train_and_evaluate(SVR(), 'SVM', params)

        # Decision Tree Model
        print("\nEvaluating Decision Tree Regressor...")
        criterions = ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
        splitters = ['best', 'random']
        for criterion in criterions:
            for splitter in splitters:
                params = {'criterion': criterion, 'splitter': splitter}
                self.train_and_evaluate(DecisionTreeRegressor(), 'Decision Tree', params)

        # Random Forest Model
        print("\nEvaluating Random Forest Regressor...")
        n_estimators_list = [10, 50, 100]
        criterions = ['squared_error', 'absolute_error', 'friedman_mse', 'poisson']
        for criterion in criterions:
            for n_estimators in n_estimators_list:
                params = {'n_estimators': n_estimators, 'criterion': criterion}
                self.train_and_evaluate(RandomForestRegressor(), 'Random Forest', params)

        print(f"\nBest Model Selected: {self.best_model} with R_Score: {self.best_r_score:.4f}")

    def save_best_model(self, filename="best_model.sav"):
        # Save the best model to a file
        if self.best_model:
            pickle.dump(self.best_model, open(filename, 'wb'))
            print(f"Best model saved as {filename}")
        else:
            print("No model has been trained yet.")

    def load_best_model(self, filename="best_model.sav"):
        # Load the best model from a file
        self.best_model = pickle.load(open(filename, 'rb'))
        print(f"Best model loaded from {filename}")

    def predict_with_best_model(self, input_data):
        if self.best_model:
            result = self.best_model.predict([input_data])
            print(f"Prediction with Best Model: {result}")
        else:
            print("No best model found. Please train and select a model first.")

# Create an instance of BestModelSelector
best_model_selector = BestModelSelector("insurance_pre.csv")

# Select the best model based on R2 score
best_model_selector.select_best_model()

# Save the best model to a file
best_model_selector.save_best_model("best_model.sav")
