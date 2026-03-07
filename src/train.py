import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

data = pd.read_csv(r"E:\PYCHARM\PycharmProjects\House_Price_Predictor\data\housing.csv")

X = data.drop("median_house_value", axis=1)
y = data["median_house_value"]

num_cols = X.drop("ocean_proximity", axis=1).columns
cat_cols = ["ocean_proximity"]

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

cat_pipeline = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

model = Pipeline([
    ("preprocessing", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

joblib.dump(model, r"E:\PYCHARM\PycharmProjects\House_Price_Predictor\models\model.pkl")

print("Model saved successfully!")