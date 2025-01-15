# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import streamlit as st

# Header Aplikasi
st.title("Analisis Transaksi dengan Algoritma Apriori")
st.markdown("Upload dataset transaksi Anda untuk melakukan analisis menggunakan algoritma Apriori dan menemukan aturan asosiasi.")

# Upload Dataset
uploaded_file = st.file_uploader("Upload CSV File", type="csv")

if uploaded_file is not None:
    # Baca Dataset
    data = pd.read_csv(uploaded_file, delimiter=';', decimal=',', low_memory=False)
    st.subheader("Data Awal:")
    st.dataframe(data.head())

    # Bersihkan Data
    st.subheader("Cek Missing Values:")
    st.write(data.isnull().sum())

    # Drop baris dengan nilai kosong di kolom CustomerID dan Itemname
    data = data.dropna(subset=['CustomerID', 'Itemname'])

    # Normalisasi kolom Itemname
    data['Itemname'] = data['Itemname'].astype(str).str.strip()

    # Gabungkan Item Berdasarkan BillNo
    data = data.groupby(['BillNo']).agg({
        'Itemname': lambda x: list(x),
        'Quantity': 'sum',
        'Date': 'first',
        'Price': 'sum',
        'CustomerID': 'first',
        'Country': 'first'
    }).reset_index()

    st.subheader("Data Setelah Penggabungan:")
    st.dataframe(data.head())

    # Konversi Kolom Date ke Format Datetime
    data['Date'] = pd.to_datetime(data['Date'], format='%d.%m.%Y %H:%M', errors='coerce')
    data['Year'] = data['Date'].dt.year

    # Filter Data Tahun Terbaru
    latest_year = data['Year'].max()
    data = data[data['Year'] == latest_year]
    st.subheader(f"Data Tahun Terbaru ({latest_year}):")
    st.dataframe(data.head())

    # One-Hot Encoding untuk Item Transaksi
    te = TransactionEncoder()
    basket_encoded = te.fit(data['Itemname']).transform(data['Itemname'])
    basket_encoded = pd.DataFrame(basket_encoded, columns=te.columns_)

    # Jalankan Algoritma Apriori
    min_support = st.slider("Pilih Nilai Minimum Support:", 0.01, 0.1, 0.01)
    frequent_itemsets = apriori(basket_encoded, min_support=min_support, use_colnames=True)

    if frequent_itemsets.empty:
        st.error("Tidak ada frequent itemsets yang ditemukan. Coba turunkan nilai minimum support.")
    else:
        # Tampilkan Frequent Itemsets
        frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(lambda x: ', '.join(sorted(list(x))))
        st.subheader("Frequent Itemsets:")
        st.dataframe(frequent_itemsets)

        # Menentukan Aturan Asosiasi
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

        if rules.empty:
            st.warning("Tidak ada aturan asosiasi yang ditemukan.")
        else:
            st.subheader("Aturan Asosiasi:")
            st.dataframe(rules)

            # Visualisasi Distribusi
            st.subheader("Visualisasi Distribusi Support, Confidence, dan Lift")
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Support
            axes[0].hist(rules['support'], bins=10, edgecolor='black')
            axes[0].set_title('Distribusi Support')
            axes[0].set_xlabel('Support')
            axes[0].set_ylabel('Frekuensi')

            # Confidence
            axes[1].hist(rules['confidence'], bins=10, edgecolor='black')
            axes[1].set_title('Distribusi Confidence')
            axes[1].set_xlabel('Confidence')
            axes[1].set_ylabel('Frekuensi')

            # Lift
            axes[2].hist(rules['lift'], bins=10, edgecolor='black')
            axes[2].set_title('Distribusi Lift')
            axes[2].set_xlabel('Lift')
            axes[2].set_ylabel('Frekuensi')

            plt.tight_layout()
            st.pyplot(fig)
