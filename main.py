import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import catboost
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from catboost import CatBoostRegressor
import warnings
import joblib
import requests
import tempfile
import os

sklearn.set_config(transform_output="pandas")

warnings.filterwarnings('ignore')


# model = CatBoostRegressor()
# model.load_model('/home/aleksey/DS_bootcamp/houses_prices/model.cbm')
ml_pipline = joblib.load('ml_pipline.pkl')

st.title('Предсказание цены квартиры')

uploaded_file = st.file_uploader("Выберите CSV файл", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    prediction = np.exp(ml_pipline.predict(data))
    # prediction = np.exp(model.predict(data))
    st.write("Результаты предсказания:")
    st.write(prediction)