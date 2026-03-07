import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    data = pd.read_csv(path)
    return data

def preprocess_data(data):

    data = data.dropna()

    X = data.drop("median_house_value", axis=1)
    y = data["median_house_value"]

    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test