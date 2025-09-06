import joblib

clf = joblib.load("xgboost_model.pkl")
if hasattr(clf.best_estimator_, "feature_names_in_"):
    print(clf.best_estimator_.feature_names_in_)
    print(len(clf.best_estimator_.feature_names_in_))
else:
    print("feature_names_in_ is not available. The model may have been trained with a NumPy array or without feature names.")
