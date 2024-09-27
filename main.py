import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from catboost import CatBoostRegressor
import warnings
import joblib
import sklearn
import requests
import tempfile
import os

sklearn.set_config(transform_output="pandas")

warnings.filterwarnings('ignore')


current_directory = os.path.dirname(__file__)
model_path = os.path.join(current_directory, 'model.cbm')

model = CatBoostRegressor()
model.load_model(model_path)
preprocessor = joblib.load('/home/aleksey/DS_bootcamp/houses_prices/preprocessor.pkl')

st.title('Предсказание цены квартиры')

uploaded_file = st.file_uploader("Выберите CSV файл", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data = preprocessor.transform(data)
    prediction = np.exp(model.predict(data))
    st.write("Результаты предсказания:")
    st.write(prediction)