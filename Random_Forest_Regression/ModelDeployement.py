import pickle

load_model = pickle.load(open("finalised_randomForest_model.sav", 'rb'))
result = load_model.predict([[1234, 123, 4565, 0, 0]])
print(f"Prediction for input [1234, 123, 4565, 0, 0]: {result[0]:.2f}")