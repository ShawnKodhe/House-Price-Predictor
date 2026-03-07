# House-Price-Predictor
This project predicts house prices using machine learning.

This project uses a Scikit-learn Pipeline with ColumnTransformer
to ensure preprocessing during inference matches training.

## Features
- Data preprocessing
- Exploratory data analysis
- Random Forest model
- Model evaluation
- Web app using Streamlit

## Technologies
Python
Scikit-learn
Pandas
Streamlit

## How to Run

1. Install requirements
pip install -r requirements.txt

2. Add a model.pkl file
Add the file under a folder named models

4. Train the model
python src/train.py

5. Run web app
streamlit run app.py

N/B: Ensure you reference the files in train.py, i.e "data" and joblib.dump(....) correctly to their file locations
