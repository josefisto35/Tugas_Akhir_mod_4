import numpy as np

import pandas as pd

from sklearn import linear_model

from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.model_selection import train_test_split

import streamlit as st

import glob

import seaborn as sns

import matplotlib.pyplot as plt

# mengambil seluruh file CSV yang ada pada folder

path = 'dataset_coin/'  # ubah dengan path ke folder CSV Anda

all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:

    df = pd.read_csv(filename, index_col=None, header=0)

    # menambahkan kolom 'Coin' untuk menunjukkan nama koin

    coin_name = filename.split('/')[-1].split('.')[0]

    df['Coin'] = coin_name

    li.append(df)

# menggabungkan seluruh DataFrame menjadi satu

df = pd.concat(li, axis=0, ignore_index=True)

# membuat tampilan aplikasi dengan Streamlit

st.set_option('deprecation.showPyplotGlobalUse', False) # tambahkan baris ini

st.write("<h1 style='text-align: center'>Prediksi Harga Koin Kripto</h1>", unsafe_allow_html=True)

# Menambahkan input form prediksi

coin_list = sorted(df['Coin'].unique())

coin_choice = st.sidebar.selectbox('Pilih Koin', coin_list)

feature_cols = ['High', 'Low', 'Open', 'Close', 'Volume', 'Marketcap']

target_col_start = st.sidebar.selectbox('Pilih Kolom yang Akan Dijadikan Target Awal', feature_cols)

target_col_end = st.sidebar.selectbox('Pilih Kolom yang Akan Dijadikan Target Akhir', feature_cols)

# mengecek apakah target awal lebih kecil daripada target akhir

if feature_cols.index(target_col_start) >= feature_cols.index(target_col_end):

    st.error('Kolom target awal harus lebih kecil dari kolom target akhir')

else:

    target_cols = feature_cols[feature_cols.index(target_col_start):feature_cols.index(target_col_end) + 1]

    # memilih data koin tertentu

    df_choice = df[df['Coin'] == coin_choice]

    # memilih variabel yang akan digunakan untuk prediksi

    X = df_choice[feature_cols]

    # membagi data menjadi data latih dan data uji

    train_size = st.sidebar.slider('Ukuran Data Latih', min_value=0.1, max_value=0.9, value=0.7, step=0.1)

    test_size = 1 - train_size

    random_state = 42

    X_train, X_test, y_train, y_test = train_test_split(X, df_choice[target_cols], test_size=test_size, random_state=random_state)

    # Melatih model regresi linier

    reg = linear_model.LinearRegression()

    reg.fit(X_train, y_train)

    # Memprediksi harga koin pada data uji

    y_pred = reg.predict(X_test)

    # Menghitung nilai rata-rata error

    mse = mean_squared_error(y_test, y_pred)

    mae = mean_absolute_error(y_test, y_pred)

    # Menampilkan hasil prediksi dan evaluasi model

    st.write('### Hasil Prediksi')

    st.write(y_pred)

    st.write('### Evaluasi Model')

    st.write('Mean Squared Error: ', mse)

    st.write('Mean Absolute Error: ', mae)

    # Menampilkan grafik hasil prediksi

    plt.figure(figsize=(16,8))

    plt.plot(y_test.values, label='Actual')

    plt.plot(y_pred, label='Predicted')

    plt.xlabel('Data')

    plt.ylabel('Harga Koin')

    plt.title('Grafik Prediksi Harga Koin')

    plt.legend()

    st.pyplot()



