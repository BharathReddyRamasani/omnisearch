from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def get_model(task, algo):
    if task == "classification":
        if algo == "logistic":
            return LogisticRegression(max_iter=1000)
        return RandomForestClassifier(n_estimators=200, random_state=42)

    if algo == "linear":
        return LinearRegression()
    return RandomForestRegressor(n_estimators=200, random_state=42)
