import pickle
import numpy as np

# Load the best model and make predictions (for example, using new data)
loaded_model = pickle.load(open("finalized_lightgbm_model_with_gridsearch.sav", 'rb'))
new_data = np.array([[1234, 123, 4565, 0, 1]])
prediction = loaded_model.predict(new_data)
print(prediction)