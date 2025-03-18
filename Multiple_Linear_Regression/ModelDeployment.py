import pickle

# Load the saved model from the file
load_model = pickle.load(open("finalized_multiple_profit.sav", 'rb'))

# Make a prediction for a new set of input values (user-provided data)
result = load_model.predict([[1234, 123, 4565, 0, 0]])

# Print the predicted result
print(result)