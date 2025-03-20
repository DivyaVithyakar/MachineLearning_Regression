import pickle

import AutoMachineLearningModelSelection

best_model_selector = AutoMachineLearningModelSelection.BestModelSelector("insurance_pre.csv")
# Load the best model from a file (if needed)
best_model_selector.load_best_model("best_model.sav")

# Make a prediction with the best model
best_model_selector.predict_with_best_model([40, 30, 1, 1, 0])
