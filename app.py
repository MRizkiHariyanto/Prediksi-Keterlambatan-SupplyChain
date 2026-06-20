import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Konfigurasi Halaman Utama
st.set_page_config(
    page_title="Sistem Prediksi Risiko Logistik",
    page_icon="🚚",
    layout="wide"
)

# 1. LOAD ARTIFAK MODEL & ENCODER
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('model_logistik.pkl')
        encoder = joblib.load('encoder_rute.pkl')
        return model, encoder
    except Exception as e:
        st.error(f"Gagal memuat file model/encoder. Pastikan file .pkl ada di repositori. Error: {e}")
        return None, None

model, encoder = load_artifacts()

# 2. HEADER APLIKASI
st.title("🚚 AI Risk Monitor: Deteksi Keterlambatan Logistik")
st.markdown("""
Aplikasi ini mendeteksi risiko keterlambatan pengiriman paket (*Late Delivery Risk*) berdasarkan parameter operasional, rute, dan beban logistik gudang menggunakan arsitektur **XGBoost Teroptimasi** (Intervensi Threshold: **0.45**).
""")

st.hr()

# 3. SIDEBAR INFO & SIMULASI FINANSIAL
st.sidebar.header("⚙️ Parameter Bisnis (Audit SLA)")
cost_SLA_penalty = st.sidebar.number_input("Denda SLA per Paket Telat ($)", value=50.0, step=5.0)
cost_false_alarm = st.sidebar.number_input("Biaya Lembur Alarm Palsu ($)", value=5.0, step=1.0)
cost_intervention = st.sidebar.number_input("Biaya Intervensi Efektif ($)", value=5.0, step=1.0)

# 4. AREA UPLOAD DATASET BARU
st.subheader("📋 Upload Data Manifest Operasional Terbaru")
uploaded_file = st.file_uploader("Unggah file berkstensi .csv yang berisi data transaksi pengiriman", type=["csv"])

if uploaded_file is not None and model is not None:
    try:
        # Membaca data mentah dari user
        df_input = pd.read_csv(uploaded_file, encoding='latin1')
        st.success(f"Berhasil memuat {df_input.shape[0]} baris data manifest!")
        
        # Tampilkan cuplikan data asli
        with st.expander("Lihat Data Mentah yang Diunggah"):
            st.dataframe(df_input.head(10))
            
        # 5. PIPELINE FEATURE ENGINEERING (Harus persis seperti di Colab)
        with st.spinner("Menjalankan Alur Rekayasa Fitur & Sintesis Rute..."):
            
            # Eliminasi Kolom Noise & Leakage
            kolom_buang = [
                'Days for shipping (real)', 'Delivery Status', 'shipping date (DateOrders)', 'Order Status',
                'Customer Email', 'Customer Password', 'Customer Fname', 'Customer Lname', 'Customer Street', 'Customer Zipcode',
                'Order Id', 'Customer Id', 'Order Customer Id', 'Product Card Id', 'Order Item Id', 'Order Item Cardprod Id',
                'Department Id', 'Category Id', 'Product Category Id', 'Product Image', 'Product Description', 'Product Status',
                'Order Zipcode', 'Late_delivery_risk' # Buang target jika kebetulan ada di file upload
            ]
            df_clean = df_input.drop(columns=kolom_buang, errors='ignore')
            
            # Feature Engineering: Waktu & Beban Logistik
            if 'order date (DateOrders)' in df_input.columns:
                df_input['order date (DateOrders)'] = pd.to_datetime(df_input['order date (DateOrders)'])
                df_clean['Order_DayOfWeek'] = df_input['order date (DateOrders)'].dt.dayofweek
            else:
                df_clean['Order_DayOfWeek'] = 0
                
            if 'Order Item Quantity' in df_input.columns and 'Days for shipment (scheduled)' in df_input.columns:
                df_clean['Logistics_Burden'] = df_input['Order Item Quantity'] / (df_input['Days for shipment (scheduled)'] + 0.001)
            else:
                df_clean['Logistics_Burden'] = 0.0
                
            if 'Order City' in df_input.columns and 'Customer City' in df_input.columns:
                df_clean['Route'] = df_input['Order City'] + " to " + df_input['Customer City']
            else:
                df_clean['Route'] = "Unknown to Unknown"
                
            # Memastikan urutan kolom input sama persis dengan yang diharapkan encoder
            # TargetEncoder akan menyelaraskan kolom saat transform dijalankan
            X_encoded = encoder.transform(df_clean)
            
        # 6. EKSEKUSI PREDIKSI KECERDASAN BUATAN
        with st.spinner("Mesin AI sedang memetakan probabilitas risiko..."):
            y_pred_proba = model.predict_proba(X_encoded)[:, 1]
            # Intervensi Manajerial sesuai Threshold Audit Bisnis (0.45)
            y_pred_custom = np.where(y_pred_proba > 0.45, 1, 0)
            
        # 7. INJEKSI HASIL PREDIKSI KE DATAFRAME UTAMA
        df_result = df_input.copy()
        df_result['Risk_Probability'] = y_pred_proba
        df_result['AI_Prediction_Status'] = np.where(y_pred_custom == 1, '⚠️ RISIKO TERLAMBAT', '✅ AMAN')
        
        # 8. PANEL RINGKASAN EKSEKUTIF (METRICS)
        total_paket = len(y_pred_custom)
        total_telat = int(np.sum(y_pred_custom))
        rasio_telat = (total_telat / total_paket) * 100
        
        st.subheader("📊 Laporan Realitas Risiko & Dampak Finansial")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Pengiriman Diproses", f"{total_paket} Paket")
        with col2:
            st.metric("Paket Terindikasi Bahaya (AI)", f"{total_telat} Paket", f"{rasio_telat:.1f}% dari total")
        with col3:
            # Simulasi Finansial Sederhana untuk Tampilan Dashboard
            # Skenario Tanpa AI: Mengasumsikan semua paket berisiko akan lolos dan kena denda penuh
            cost_baseline = total_telat * cost_SLA_penalty
            # Skenario Dengan AI: Mengasumsikan intervensi berhasil menekan denda, diganti biaya operasional intervensi
            cost_ai = total_telat * cost_intervention
            penghematan = cost_baseline - cost_ai
            st.metric("Estimasi Anggaran Diselamatkan", f"${penghematan:,.2f}", delta="Efisiensi AI", delta_color="inverse")
            
        st.hr()
        
        # 9. INTERACTIVE DATA VIEWER
        st.subheader("🔍 Filter & Telusuri Hasil Keputusan Sistem AI")
        status_filter = st.selectbox("Filter Status Pengiriman:", ["Semua Data", "⚠️ RISIKO TERLAMBAT", "✅ AMAN"])
        
        if status_filter != "Semua Data":
            df_display = df_result[df_result['AI_Prediction_Status'] == status_filter]
        else:
            df_display = df_result
            
        st.dataframe(df_display[['Order Id', 'Customer City', 'Order City', 'Product Name', 'Risk_Probability', 'AI_Prediction_Status']].sort_values(by='Risk_Probability', ascending=False))
        
        # 10. FITUR UNDUH LAPORAN HASIL PREDIKSI
        st.subheader("📥 Unduh Dokumen Hasil Audit")
        csv_output = df_result.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Unduh File CSV Hasil Prediksi AI",
            data=csv_output,
            file_name="Hasil_Prediksi_Risiko_Logistik.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file CSV: {e}")
        st.info("Pastikan file CSV yang Anda unggah memiliki struktur kolom yang sama dengan dataset DataCo asli saat fase training.")

else:
    st.info("Silakan unggah file manifest pengiriman (.csv) untuk memulai proses analisis risiko logistik.")