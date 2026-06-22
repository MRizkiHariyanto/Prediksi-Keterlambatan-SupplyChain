import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Direktori tempat app.py berada (aman di Streamlit Cloud maupun lokal)
BASE_DIR = Path(__file__).parent

# ==============================================================================
# KONFIGURASI HALAMAN
# ==============================================================================
st.set_page_config(
    page_title="DSS Logistik | Prediksi Risiko Keterlambatan",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# CSS KUSTOM — DESAIN PREMIUM DARK MODE
# ==============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Header Utama */
    .dss-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 0.5rem 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        border-left: 6px solid #7c3aed;
        box-shadow: 0 8px 32px rgba(124, 58, 237, 0.2);
    }
    .dss-header h1 { color: #ffffff; margin: 0; font-size: 1.1rem; font-weight: 800; letter-spacing: -0.5px; }
    .dss-header p  { color: #a78bfa; margin: 0.1rem 0 0 0; font-size: 0.75rem; }

    /* Section Title */
    .section-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #7c3aed;
        margin-bottom: 0.6rem;
        padding-left: 2px;
    }

    /* Risk Result Cards */
    .result-danger {
        background: linear-gradient(135deg, rgba(239,68,68,0.15) 0%, rgba(185,28,28,0.1) 100%);
        border: 2px solid rgba(239,68,68,0.5);
        border-radius: 10px;
        padding: 0.5rem 1rem;
        text-align: center;
        box-shadow: 0 0 40px rgba(239, 68, 68, 0.15);
        animation: pulse-red 2s infinite;
    }
    .result-safe {
        background: linear-gradient(135deg, rgba(16,185,129,0.15) 0%, rgba(5,150,105,0.1) 100%);
        border: 2px solid rgba(16,185,129,0.5);
        border-radius: 10px;
        padding: 0.5rem 1rem;
        text-align: center;
        box-shadow: 0 0 40px rgba(16, 185, 129, 0.15);
    }
    @keyframes pulse-red {
        0%, 100% { box-shadow: 0 0 30px rgba(239,68,68,0.15); }
        50%       { box-shadow: 0 0 50px rgba(239,68,68,0.35); }
    }

    /* Gauge Container */
    .gauge-label {
        font-size: 0.78rem; color: #94a3b8; text-align: center; margin-top: 0.3rem;
    }

    /* Form styling */
    [data-testid="stForm"] {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 0.5rem 1rem;
    }

    /* Submit button */
    [data-testid="stFormSubmitButton"] button {
        background: linear-gradient(90deg, #7c3aed, #4f46e5) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 0.65rem 2rem !important;
        width: 100%;
        transition: all 0.2s ease;
        box-shadow: 0 4px 15px rgba(124,58,237,0.3) !important;
    }
    [data-testid="stFormSubmitButton"] button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(124,58,237,0.45) !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #0d1117; }
    [data-testid="stSidebar"] * { color: #94a3b8 !important; }

    /* Info pills */
    .pill {
        display: inline-block;
        background: rgba(124,58,237,0.12);
        border: 1px solid rgba(124,58,237,0.3);
        color: #a78bfa;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 1. MUAT ARTEFAK MODEL & ENCODER
# ==============================================================================
@st.cache_resource(show_spinner="⏳ Memuat model AI ke memori...")
def load_artifacts():
    """Memuat model XGBoost dan Target Encoder dari file .pkl.
    
    Menggunakan path absolut (BASE_DIR) agar berjalan konsisten
    di lingkungan lokal maupun Streamlit Cloud.
    """
    model_path   = BASE_DIR / 'model_logistik.pkl'
    encoder_path = BASE_DIR / 'encoder_rute.pkl'
    
    try:
        model   = joblib.load(model_path)
        encoder = joblib.load(encoder_path)
        return model, encoder
    except FileNotFoundError as e:
        st.error(
            f"❌ File artefak tidak ditemukan: `{e}`\n\n"
            f"Pastikan file `model_logistik.pkl` dan `encoder_rute.pkl` "
            f"ada di direktori yang sama dengan `app.py` di repositori GitHub."
        )
        return None, None
    except Exception as e:
        st.error(f"❌ Gagal memuat model/encoder: {e}")
        return None, None

model, encoder = load_artifacts()


# ==============================================================================
# 2. HEADER APLIKASI
# ==============================================================================
st.markdown("""
<div class="dss-header">
    <h1>🚚 DSS Logistik — Prediksi Risiko Keterlambatan Pengiriman</h1>
    <p>
        <em>Decision Support System</em> berbasis <strong>XGBoost Teroptimasi</strong>.
        Isi formulir operasional di bawah untuk mendapatkan asesmen risiko keterlambatan secara real-time.
    </p>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# 3. SIDEBAR — INFO MODEL & PANDUAN
# ==============================================================================
with st.sidebar:
    st.markdown("## 📊 Informasi Model")
    st.markdown("---")
    st.markdown("""
    **Arsitektur:** XGBoost Classifier  
    **Threshold Intervensi:** `0.45`  
    **Target Variabel:** `Late_delivery_risk`
    """)
    st.markdown("---")
    st.markdown("### 📖 Fitur Otomatis (Backend)")
    st.markdown("""
    Dua fitur berikut dihitung **otomatis** oleh sistem dan **tidak perlu** diinput manual:

    **`Logistics_Burden`**  
    `= Qty / (Days_Scheduled + 0.001)`

    **`Route`**  
    `= Order City + " to " + Customer City`
    """)
    st.markdown("---")
    st.markdown("### 🔴 Interpretasi Risiko")
    st.markdown("""
    | Probabilitas | Status |
    |---|---|
    | > 45% | ⚠️ Intervensi |
    | ≤ 45% | ✅ Aman |
    """)
    st.markdown("---")
    st.caption("© 2025 DSS Logistik · Riset Keterlambatan Supply Chain")


# ==============================================================================
# LOOKUP: NEGARA → (LATITUDE, LONGITUDE) REPRESENTATIF
# Berdasarkan distribusi geografis dataset DataCo Supply Chain
# ==============================================================================
COUNTRY_GEO: dict[str, tuple[float, float]] = {
    "EE. UU. (United States)": (39.50, -98.35),
    "Puerto Rico":             (18.22, -66.59),
    "México":                  (23.63, -102.55),
    "Francia":                 (46.23,   2.21),
    "Alemania":                (51.17,  10.45),
    "Brasil":                  (-14.24, -51.93),
    "Australia":               (-25.27, 133.78),
    "Argentina":               (-38.42, -63.62),
    "Colombia":                (4.57,  -74.30),
    "China":                   (35.86, 104.20),
    "Indonesia":               (-0.79, 113.92),
    "India":                   (20.59,  78.96),
    "Nigeria":                 (9.08,    8.68),
    "Egipto":                  (26.82,  30.80),
    "Sudáfrica":               (-28.47,  24.68),
    "Otros / Lainnya":         (0.00,    0.00),
}
COUNTRY_LABELS = list(COUNTRY_GEO.keys())

# Mapping label → nama asli di dataset DataCo
COUNTRY_DATACO: dict[str, str] = {
    "EE. UU. (United States)": "EE. UU.",
    "Puerto Rico":             "Puerto Rico",
    "México":                  "México",
    "Francia":                 "Francia",
    "Alemania":                "Alemania",
    "Brasil":                  "Brasil",
    "Australia":               "Australia",
    "Argentina":               "Argentina",
    "Colombia":                "Colombia",
    "China":                   "China",
    "Indonesia":               "Indonesia",
    "India":                   "India",
    "Nigeria":                 "Nigeria",
    "Egipto":                  "Egipto",
    "Sudáfrica":               "Sudáfrica",
    "Otros / Lainnya":         "Others",
}


try:
    from city_data import ORDER_CITIES, CUSTOMER_CITIES, ORDER_CITY_MAP, CUSTOMER_CITY_MAP
except ImportError:
    ORDER_CITIES = ["Chicago"]
    CUSTOMER_CITIES = ["Los Angeles"]
    ORDER_CITY_MAP = {"Chicago": ("Illinois", "Estados Unidos")}
    CUSTOMER_CITY_MAP = {"Los Angeles": ("CA", "EE. UU.")}


def build_input_dataframe(
    order_city:      str,
    customer_city:   str,
    shipping_mode:   str,
    days_scheduled:  int,
    order_type:      str,
    day_of_week:     int,
) -> pd.DataFrame:
    """
    Menyusun 1 baris DataFrame dengan 31 fitur model.

    INPUT USER (6 fitur):
      Order City, Customer City, Shipping Mode, Days for shipment, Type, Order_DayOfWeek

    OTOMATIS DIHITUNG:
      Route = Order City + " to " + Customer City
      State & Country (Order & Customer) = diturunkan otomatis dari peta Kota di dataset
      Latitude/Longitude = diturunkan dari Customer Country
      Logistics_Burden = default 0.75
    """
    route = f"{order_city.strip()} to {customer_city.strip()}"
    
    # Auto-derive state & country from city
    order_state, order_country = ORDER_CITY_MAP.get(order_city, ("California", "Estados Unidos"))
    customer_state, customer_country = CUSTOMER_CITY_MAP.get(customer_city, ("CA", "EE. UU."))
    
    # Auto-derive lat/lon based on Customer Country (only 2 in dataset: EE. UU. or Puerto Rico)
    lat, lon = (18.22, -66.59) if customer_country == "Puerto Rico" else (39.50, -98.35)

    row = {
        "Type":                          order_type,
        "Days for shipment (scheduled)": days_scheduled,
        "Benefit per order":             10.0,
        "Sales per customer":            200.0,
        "Category Name":                 "Fishing",
        "Customer City":                 customer_city.strip(),
        "Customer Country":              customer_country,
        "Customer Segment":              "Consumer",
        "Customer State":                customer_state.strip(),
        "Department Name":               "Fan Shop",
        "Latitude":                      lat,
        "Longitude":                     lon,
        "Market":                        "USCA",
        "Order City":                    order_city.strip(),
        "Order Country":                 order_country,
        "Order Item Discount":           5.0,
        "Order Item Discount Rate":      0.05,
        "Order Item Product Price":      99.99,
        "Order Item Profit Ratio":       0.18,
        "Order Item Quantity":           3,
        "Sales":                         284.97,
        "Order Item Total":              299.97,
        "Order Profit Per Order":        53.99,
        "Order Region":                  "West of USA",
        "Order State":                   order_state.strip(),
        "Product Name":                  "Field & Stream Sportsman 16 Gun Fire Safe",
        "Product Price":                 99.99,
        "Shipping Mode":                 shipping_mode,
        "Order_DayOfWeek":               day_of_week,
        "Logistics_Burden":              0.75,
        "Route":                         route,
    }

    return pd.DataFrame([row])


def run_prediction(df_raw: pd.DataFrame):
    """
    Menjalankan transformasi encoder dan inferensi model.

    Returns
    -------
    float
        Probabilitas keterlambatan (kelas 1) dalam rentang [0, 1].
    """
    X_encoded    = encoder.transform(df_raw)
    proba_array  = model.predict_proba(X_encoded)
    return float(proba_array[0, 1])


# ==============================================================================
# 5. FORMULIR INPUT DSS — HANYA FITUR PALING BERPENGARUH
# ==============================================================================
st.markdown('<p class="section-label">📋 Formulir Asesmen Operasional — Berbasis Top-10 Fitur SHAP</p>',
            unsafe_allow_html=True)

with st.form(key="dss_prediction_form", border=True):

    st.markdown("**🗺️ Rute Pengiriman** <small style='color:#7c3aed'>(#1 SHAP)</small>", unsafe_allow_html=True)
    col_ocity, col_ccity = st.columns(2)
    with col_ocity:
        order_city = st.selectbox(
            "Kota Asal Gudang (Order City) ★",
            options=ORDER_CITIES if ORDER_CITIES else ["Chicago", "Los Angeles", "New York"],
            index=0,
            help="[SHAP Rank #8] Kota tempat gudang pengirim berada."
        )
    with col_ccity:
        customer_city = st.selectbox(
            "Kota Tujuan Pelanggan (Customer City) ★",
            options=CUSTOMER_CITIES if CUSTOMER_CITIES else ["Los Angeles", "Chicago", "Houston"],
            index=0,
            help="[SHAP Rank #7] Kota tujuan pengiriman pelanggan."
        )

    st.markdown("**🚛 Mode & Jadwal Pengiriman** <small style='color:#7c3aed'>(#2 & #6 SHAP)</small>", unsafe_allow_html=True)
    col_mode, col_days, col_type, col_dow = st.columns(4)
    with col_mode:
        shipping_mode = st.selectbox(
            "Moda Pengiriman ★",
            options=["Standard Class", "Second Class", "First Class", "Same Day"],
            index=0,
            help="[SHAP Rank #2 | Gini #3] Kelas layanan pengiriman yang paling kuat mempengaruhi keterlambatan."
        )
    with col_days:
        days_scheduled = st.number_input(
            "Target Hari Pengiriman ★",
            min_value=0, max_value=30, value=4, step=1,
            help="[SHAP Rank #6 | Gini #4] Estimasi durasi pengiriman terjadwal."
        )
    with col_type:
        order_type = st.selectbox(
            "Tipe Transaksi ★",
            options=["DEBIT", "TRANSFER", "CASH", "PAYMENT"],
            index=0,
            help="[SHAP Rank #5] Metode pembayaran transaksi."
        )
    with col_dow:
        day_of_week = st.selectbox(
            "Hari Pemesanan",
            options=list(range(7)),
            format_func=lambda x: ["Senin","Selasa","Rabu","Kamis","Jumat","Sabtu","Minggu"][x],
            index=0,
            help="Hari dalam minggu saat pesanan dibuat."
        )

    st.caption("💡 **Smart Location:** Lokasi & koordinat akan dideteksi otomatis berdasarkan pilihan Kota.")

    # ── Tombol Submit ─────────────────────────────────────────────────────────
    submitted = st.form_submit_button(
        "🔍 Prediksi Risiko Keterlambatan",
        use_container_width=True
    )


# ==============================================================================
# 6. LOGIKA BISNIS — DIEKSEKUSI SAAT FORM DIKIRIM
# ==============================================================================
if submitted:
    # ── Validasi Input Wajib ──────────────────────────────────────────────────
    errors = []
    if not order_city.strip():
        errors.append("🔴 **Kota Asal Gudang** tidak boleh kosong.")
    if not customer_city.strip():
        errors.append("🔴 **Kota Tujuan Pelanggan** tidak boleh kosong.")
    if order_city.strip().lower() == customer_city.strip().lower() and order_city.strip():
        errors.append("⚠️ Kota asal dan tujuan identik — harap periksa kembali data rute.")

    if errors:
        for err in errors:
            st.error(err)
        st.stop()

    if model is None or encoder is None:
        st.error("❌ Model atau encoder gagal dimuat. Periksa keberadaan file `.pkl`.")
        st.stop()

    # ── Proses Prediksi ───────────────────────────────────────────────────────
    with st.spinner("🤖 Sistem AI sedang mengevaluasi risiko pengiriman..."):
        # Panggil fungsi build dataframe (HANYA MENGGUNAKAN 6 INPUT)
        df_input = build_input_dataframe(
            order_city=order_city,
            customer_city=customer_city,
            shipping_mode=shipping_mode,
            days_scheduled=days_scheduled,
            order_type=order_type,
            day_of_week=day_of_week
        )
        try:
            probability = run_prediction(df_input)
            is_risky    = probability > 0.45

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat menjalankan model: {e}")
            st.info("Periksa bahwa nama kolom pada DataFrame sudah sesuai dengan yang diharapkan encoder.")
            st.stop()

    # ── Tampilkan Hasil ───────────────────────────────────────────────────────
    st.markdown('<p class="section-label">📡 Hasil Asesmen Risiko Real-Time</p>', unsafe_allow_html=True)

    col_result, col_detail = st.columns([1.4, 1])

    with col_result:
        pct = probability * 100

        if is_risky:
            st.markdown(f"""
            <div class="result-danger">
                <div style="font-size:1.5rem; margin-bottom:0.1rem;">⚠️</div>
                <div style="font-size:1rem; font-weight:800; color:#f87171; margin-bottom:0.1rem;">
                    RISIKO KETERLAMBATAN TERDETEKSI
                </div>
                <div style="font-size:2rem; font-weight:900; color:#ef4444; letter-spacing:-1px;">
                    {pct:.1f}%
                </div>
                <div style="color:#fca5a5; font-size:0.75rem; margin-top:0.1rem;">
                    Probabilitas Keterlambatan · Threshold: 45.0%
                </div>
                <div style="margin-top:0.3rem; font-size:0.75rem; color:#fca5a5; background:rgba(239,68,68,0.1);
                     border-radius:8px; padding:0.3rem 0.5rem;">
                    ⚡ <strong>Rekomendasi:</strong> Lakukan intervensi segera.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-safe">
                <div style="font-size:1.5rem; margin-bottom:0.1rem;">✅</div>
                <div style="font-size:1rem; font-weight:800; color:#34d399; margin-bottom:0.1rem;">
                    JADWAL PENGIRIMAN AMAN
                </div>
                <div style="font-size:2rem; font-weight:900; color:#10b981; letter-spacing:-1px;">
                    {pct:.1f}%
                </div>
                <div style="color:#6ee7b7; font-size:0.75rem; margin-top:0.1rem;">
                    Probabilitas Keterlambatan · Threshold: 45.0%
                </div>
                <div style="margin-top:0.3rem; font-size:0.75rem; color:#6ee7b7; background:rgba(16,185,129,0.1);
                     border-radius:8px; padding:0.3rem 0.5rem;">
                    ✔️ <strong>Rekomendasi:</strong> Pengiriman sesuai jadwal standar.
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_detail:
        st.markdown("**📋 Ringkasan Parameter Prediksi**")

        # Fitur yang dihitung otomatis
        customer_state, customer_country = CUSTOMER_CITY_MAP.get(customer_city, ("CA", "EE. UU."))
        order_state, order_country       = ORDER_CITY_MAP.get(order_city, ("California", "Estados Unidos"))
        
        lat_val, lon_val = (18.22, -66.59) if customer_country == "Puerto Rico" else (39.50, -98.35)
        route_val        = f"{order_city.strip()} → {customer_city.strip()}"
        day_names        = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]

        st.markdown(f"""
        | Parameter | Nilai |
        |---|---|
        | 📍 Rute | `{route_val}` |
        | 🌍 Negara Pelanggan | `{customer_country}` |
        | 📌 Negara Gudang | `{order_country}` |
        | 📌 State Pelanggan | `{customer_state}` |
        | 🚛 Moda Kirim | `{shipping_mode}` |
        | ⏱️ Target Hari | `{days_scheduled} hari` |
        | 💳 Tipe Transaksi | `{order_type}` |
        | 📅 Hari Pesan | `{day_names[day_of_week]}` |
        | 🗺️ Koordinat | `{lat_val:.2f}°, {lon_val:.2f}°` |
        """)



        # Gauge visual probabilitas menggunakan progress bar
        st.markdown("**📊 Indikator Probabilitas Risiko**")
        bar_color = "normal" if not is_risky else "off"

        st.markdown(f"""
        <div style="margin: 0.5rem 0;">
            <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:#94a3b8; margin-bottom:4px;">
                <span>0%</span><span>Threshold: 45%</span><span>100%</span>
            </div>
            <div style="background:#1e293b; border-radius:8px; height:20px; position:relative; overflow:hidden;">
                <div style="
                    position:absolute; left:0; top:0; height:100%; width:{pct:.1f}%;
                    background:{'linear-gradient(90deg,#ef4444,#dc2626)' if is_risky else 'linear-gradient(90deg,#10b981,#059669)'};
                    border-radius:8px; transition: width 0.5s ease;
                "></div>
                <div style="
                    position:absolute; left:45%; top:0; height:100%; width:2px;
                    background:#f59e0b; opacity:0.8;
                "></div>
            </div>
            <div class="gauge-label">{pct:.2f}% · {'Di atas threshold → intervensi diperlukan' if is_risky else 'Di bawah threshold → aman'}</div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence badge
        if probability > 0.75:
            level, badge_color = "Sangat Tinggi", "#ef4444"
        elif probability > 0.55:
            level, badge_color = "Tinggi", "#f97316"
        elif probability > 0.45:
            level, badge_color = "Moderat", "#eab308"
        elif probability > 0.25:
            level, badge_color = "Rendah", "#22c55e"
        else:
            level, badge_color = "Sangat Rendah", "#10b981"

        st.markdown(f"""
        <div style="margin-top:0.8rem; text-align:center;">
            <span style="
                background:{badge_color}22; border:1px solid {badge_color}66;
                color:{badge_color}; border-radius:20px; padding:4px 14px;
                font-size:0.8rem; font-weight:700;
            ">Tingkat Risiko: {level}</span>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# 7. LAYAR AWAL — INSTRUKSI SAAT FORM BELUM DIISI
# ==============================================================================
else:
    st.markdown("""
    <div style="
        background: rgba(124,58,237,0.05);
        border: 1px dashed rgba(124,58,237,0.3);
        border-radius: 12px;
        padding: 1.5rem 2rem;
        text-align: center;
        margin-top: 1rem;
        color: #94a3b8;
    ">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔍</div>
        <strong style="color:#a78bfa;">Lengkapi formulir di atas</strong> dan tekan tombol
        <strong style="color:#a78bfa;">"Prediksi Risiko Keterlambatan"</strong>
        untuk mendapatkan asesmen risiko real-time dari sistem AI.
    </div>
    """, unsafe_allow_html=True)