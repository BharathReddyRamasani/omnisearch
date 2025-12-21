from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, confusion_matrix
import numpy as np

def evaluate_model(model, X, y, task):
    preds = model.predict(X)

    if task == "classification":
        return {
            "accuracy": accuracy_score(y, preds),
            "confusion_matrix": confusion_matrix(y, preds).tolist()
        }

    return {
        "rmse": mean_squared_error(y, preds, squared=False),
        "r2": r2_score(y, preds)
    }
