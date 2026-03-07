import joblib
import pandas as pd

model = joblib.load("models/model.pkl")

def predict_price(features):

    df = pd.DataFrame([features])

    prediction = model.predict(df)

    return prediction[0]