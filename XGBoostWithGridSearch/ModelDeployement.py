import pickle
import numpy as np

# To load the model later, use:
loaded_model = pickle.load(open("finalised_xgboost_model.sav", 'rb'))
new_data = np.array([[1234, 123, 4565, 0, 1]])
predict = loaded_model.predict(new_data)
print(predict)