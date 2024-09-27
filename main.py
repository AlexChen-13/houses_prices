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

sklearn.set_config(transform_output="pandas")

warnings.filterwarnings('ignore')


# model_url = 'https://github.com/AlexChen-13/houses_prices/blob/main/model.cbm'
# preprocessor_url = 'https://github.com/AlexChen-13/houses_prices/blob/main/preprocessor.pkl'


# # Скачивание модели
# model_response = requests.get(model_url)
# with tempfile.NamedTemporaryFile(delete=False) as temp_model_file:
#     temp_model_file.write(model_response.content)
#     temp_model_file_path = temp_model_file.name

# # Загрузка модели
# model = CatBoostRegressor()
# model.load_model(temp_model_file_path)

# # Скачивание процессора
# preprocessor_response = requests.get(preprocessor_url)
# with tempfile.NamedTemporaryFile(delete=False) as temp_preprocessor_file:
#     temp_preprocessor_file.write(preprocessor_response.content)
#     temp_preprocessor_file_path = temp_preprocessor_file.name

# # Загрузка процессора
# preprocessor = joblib.load(temp_preprocessor_file_path)

model = CatBoostRegressor()
model.load_model('/home/aleksey/DS_bootcamp/houses_prices/model.cbm')
preprocessor = joblib.load('/home/aleksey/DS_bootcamp/houses_prices/preprocessor.pkl')

st.title('Предсказание цены квартиры')

uploaded_file = st.file_uploader("Выберите CSV файл", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data = preprocessor.transform(data)
    prediction = np.exp(model.predict(data))
    st.write("Результаты предсказания:")
    st.write(prediction)