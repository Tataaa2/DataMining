import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Judul Aplikasi
st.title("Analisis Market Basket dengan Apriori")

# Upload File
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
if uploaded_file is not None:
    try:
        # Baca file yang diunggah dengan penanganan delimiter dan format yang tepat
        data = pd.read_csv(uploaded_file, delimiter=';', decimal=',', low_memory=False)
        st.write("Data yang diunggah:")
        st.dataframe(data.head())

        # Pilih kolom untuk analisis
        bill_col = st.selectbox("Pilih Kolom untuk Bill No (misalnya, 'BillNo')", options=data.columns)
        item_col = st.selectbox("Pilih Kolom untuk Item Name (misalnya, 'Itemname')", options=data.columns)

        # Pastikan kolom yang dipilih tidak kosong
        if bill_col and item_col:
            # Drop missing values
            data_cleaned = data.dropna(subset=[bill_col, item_col])

            # Mengelompokkan data berdasarkan BillNo dan membuat list item yang dibeli
            basket_data = data_cleaned.groupby(bill_col)[item_col].apply(list)

            # Transformasikan data ke dalam format yang dapat digunakan oleh Apriori
            te = TransactionEncoder()
            te_array = te.fit(basket_data).transform(basket_data)
            basket_encoded = pd.DataFrame(te_array, columns=te.columns_)

            # Periksa format data
            st.write("Data Biner untuk Analisis Apriori:")
            st.dataframe(basket_encoded.head())  # Debug: Periksa data yang akan diproses

            # Pastikan data hanya berisi 0/1
            if not basket_encoded.isin([0, 1]).all().all():
                st.error("Data tidak dalam format biner. Pastikan setiap nilai adalah 0 atau 1.")
            else:
                # Min Support Slider
                min_support = st.slider("Pilih Minimum Support", min_value=0.01, max_value=0.5, value=0.01, step=0.01)

                # Jalankan Apriori
                frequent_itemsets = apriori(basket_encoded, min_support=min_support, use_colnames=True)
                st.write("Frequent Itemsets:")
                st.dataframe(frequent_itemsets)

                # Pastikan bahwa frequent_itemsets tidak kosong
                if not frequent_itemsets.empty:
                    # Min Confidence Slider
                    min_confidence = st.slider("Pilih Minimum Confidence", min_value=0.1, max_value=1.0, value=0.5, step=0.1)

                    # Generate Association Rules
                    # Tambahkan parameter 'num_itemsets' jika diperlukan
                    num_itemsets = len(frequent_itemsets)  # Ini memberi tahu jumlah itemsets
                    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence, num_itemsets=num_itemsets)
                    st.write("Association Rules:")
                    st.dataframe(rules)
                else:
                    st.warning("Tidak ada frequent itemsets yang ditemukan dengan minimum support yang dipilih.")

        else:
            st.warning("Pilih kolom yang sesuai untuk analisis.")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")

else:
    st.info("Silakan unggah file CSV untuk melanjutkan analisis.") 