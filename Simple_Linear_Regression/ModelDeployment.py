import pickle

# Load the saved model
load_model = pickle.load(open("final_salary_prediction.sav", 'rb'))

# Get user input and predict salary
user_data = int(input("Enter years of experience to predict salary: "))
result = load_model.predict([[user_data]])
print(f"Predicted Salary: {result[0][0]:.2f}")
