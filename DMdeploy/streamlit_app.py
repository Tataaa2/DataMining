# 1. Import Library
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# 2. Judul Aplikasi
st.title("Aplikasi Analisis Aturan Asosiasi")

# 3. Upload Dataset
uploaded_file = st.file_uploader("Unggah file dataset (CSV)", type="csv")
if uploaded_file:
    # 4. Baca Dataset
    data = pd.read_csv(uploaded_file, delimiter=';', decimal=',', low_memory=False)
    st.subheader("Data Awal:")
    st.write(data.head())

    # 5. Bersihkan Data
    st.subheader("Pembersihan Data")
    data = data.dropna(subset=['CustomerID', 'Itemname'])
    st.write("Data setelah menghapus nilai kosong:", data.head())

    # 6. Gabungkan Item Berdasarkan BillNo
    data = data.groupby(['BillNo']).agg({
        'Itemname': lambda x: list(x),
        'Quantity': 'sum',
        'Date': 'first',
        'Price': 'sum',
        'CustomerID': 'first',
        'Country': 'first'
    }).reset_index()
    st.subheader("Data Setelah Penggabungan:")
    st.write(data.head())

    # 7. Konversi Kolom Date ke Format Datetime
    data['Date'] = pd.to_datetime(data['Date'], format='%d.%m.%Y %H:%M')
    data['Year'] = data['Date'].dt.year

    # Filter Data Tahun Terbaru
    latest_year = data['Year'].max()
    data = data[data['Year'] == latest_year]
    st.subheader(f"Data Tahun Terbaru ({latest_year}):")
    st.write(data.head())

    # 8. One-Hot Encoding untuk Item Transaksi
    te = TransactionEncoder()
    basket_encoded = te.fit(data['Itemname']).transform(data['Itemname'])
    basket_encoded = pd.DataFrame(basket_encoded, columns=te.columns_)
    st.subheader("Data Setelah One-Hot Encoding:")
    st.write(basket_encoded.head())

    # 9. Parameter Minimum Support
    min_support = st.slider("Pilih Support Minimum", 0.01, 0.1, 0.05)

    # 10. Menjalankan Algoritma Apriori
    frequent_itemsets = apriori(basket_encoded, min_support=min_support, use_colnames=True)
    st.subheader("Frequent Itemsets:")
    st.write(frequent_itemsets)

    # 11. Menentukan Aturan Asosiasi
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0, num_itemsets=len(frequent_itemsets))
    st.subheader("Aturan Asosiasi:")
    st.write(rules)

    # 12. Evaluasi Akurasi (Rata-rata Confidence)
    if not rules.empty:
        average_confidence = rules['confidence'].mean()
        st.write(f"Rata-rata Confidence (Akurasi): {average_confidence:.2%}")
    else:
        st.write("Tidak ada aturan asosiasi yang ditemukan.")

    # 13. Visualisasi Distribusi Support, Confidence, dan Lift
    st.subheader("Visualisasi Distribusi Support, Confidence, dan Lift")
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Support
    axs[0].hist(rules['support'], bins=10, edgecolor='black')
    axs[0].set_title('Distribusi Support')
    axs[0].set_xlabel('Support')
    axs[0].set_ylabel('Frekuensi')

    # Confidence
    axs[1].hist(rules['confidence'], bins=10, edgecolor='black')
    axs[1].set_title('Distribusi Confidence')
    axs[1].set_xlabel('Confidence')
    axs[1].set_ylabel('Frekuensi')

    # Lift
    axs[2].hist(rules['lift'], bins=10, edgecolor='black')
    axs[2].set_title('Distribusi Lift')
    axs[2].set_xlabel('Lift')
    axs[2].set_ylabel('Frekuensi')

    st.pyplot(fig)
