import pickle
import numpy as np

load_model = pickle.load(open('finalised_AdaBoost_with_GridSearch_model.sav', 'rb'))

new_data = np.array([[1234, 123, 4565, 0, 1]])
predict = load_model.predict(new_data)
print(predict)