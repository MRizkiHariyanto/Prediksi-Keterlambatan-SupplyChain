import streamlit as st
import pandas as pd
import numpy as np
import joblib

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
        padding: 2rem 2.5rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
        border-left: 6px solid #7c3aed;
        box-shadow: 0 8px 32px rgba(124, 58, 237, 0.2);
    }
    .dss-header h1 { color: #ffffff; margin: 0; font-size: 1.9rem; font-weight: 800; letter-spacing: -0.5px; }
    .dss-header p  { color: #a78bfa; margin: 0.5rem 0 0 0; font-size: 0.92rem; }

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
        border-radius: 14px;
        padding: 2rem 2.5rem;
        text-align: center;
        box-shadow: 0 0 40px rgba(239, 68, 68, 0.15);
        animation: pulse-red 2s infinite;
    }
    .result-safe {
        background: linear-gradient(135deg, rgba(16,185,129,0.15) 0%, rgba(5,150,105,0.1) 100%);
        border: 2px solid rgba(16,185,129,0.5);
        border-radius: 14px;
        padding: 2rem 2.5rem;
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
        border-radius: 14px;
        padding: 1.5rem;
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
    """Memuat model XGBoost dan Target Encoder dari file .pkl."""
    try:
        model   = joblib.load('model_logistik.pkl')
        encoder = joblib.load('encoder_rute.pkl')
        return model, encoder
    except FileNotFoundError as e:
        st.error(f"❌ File artefak tidak ditemukan: {e}")
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
# 4. FUNGSI SINTESIS FITUR & PREDIKSI
# ==============================================================================
def build_input_dataframe(
    qty: int,
    days_scheduled: int,
    order_city: str,
    customer_city: str,
    shipping_mode: str,
    day_of_week: int,
    order_type: str,
    customer_segment: str,
    market: str,
    order_region: str,
    product_price: float,
    discount_rate: float,
    profit_ratio: float,
) -> pd.DataFrame:
    """
    Menyusun satu baris DataFrame dengan TEPAT 31 fitur yang diharapkan model.

    Fitur model (urutan asli dari training):
      Type, Days for shipment (scheduled), Benefit per order, Sales per customer,
      Category Name, Customer City, Customer Country, Customer Segment,
      Customer State, Department Name, Latitude, Longitude, Market,
      Order City, Order Country, Order Item Discount, Order Item Discount Rate,
      Order Item Product Price, Order Item Profit Ratio, Order Item Quantity,
      Sales, Order Item Total, Order Profit Per Order, Order Region, Order State,
      Product Name, Product Price, Shipping Mode, Order_DayOfWeek,
      Logistics_Burden, Route

    Fitur `Logistics_Burden` dan `Route` disintesis otomatis dari input user.
    Kolom sekunder diisi dengan nilai default representatif dataset DataCo.
    """
    logistics_burden = qty / (days_scheduled + 0.001)
    route            = f"{order_city.strip()} to {customer_city.strip()}"
    discount_amount  = product_price * discount_rate
    total_item       = product_price * qty
    sales            = total_item * (1 - discount_rate)
    profit_per_order = total_item * profit_ratio

    # Latitude/Longitude: default pusat geografis USA (nilai representatif DataCo)
    # TargetEncoder tidak menggunakan kolom ini untuk encoding (bukan di encoder.cols),
    # sehingga nilai numerik default sudah cukup aman.
    row = {
        "Type":                         order_type,
        "Days for shipment (scheduled)": days_scheduled,
        "Benefit per order":            10.0,
        "Sales per customer":           200.0,
        "Category Name":               "Fishing",
        "Customer City":               customer_city.strip(),
        "Customer Country":            "EE. UU.",
        "Customer Segment":            customer_segment,
        "Customer State":              "CA",
        "Department Name":             "Fan Shop",
        "Latitude":                    -30.0,          # nilai rata-rata DataCo
        "Longitude":                   -70.0,          # nilai rata-rata DataCo
        "Market":                      market,
        "Order City":                  order_city.strip(),
        "Order Country":               "EE. UU.",
        "Order Item Discount":         discount_amount,
        "Order Item Discount Rate":    discount_rate,
        "Order Item Product Price":    product_price,
        "Order Item Profit Ratio":     profit_ratio,
        "Order Item Quantity":         qty,
        "Sales":                       sales,
        "Order Item Total":            total_item,
        "Order Profit Per Order":      profit_per_order,
        "Order Region":                order_region,
        "Order State":                 "California",
        "Product Name":                "Field & Stream Sportsman 16 Gun Fire Safe",
        "Product Price":               product_price,
        "Shipping Mode":               shipping_mode,
        "Order_DayOfWeek":             day_of_week,
        "Logistics_Burden":            logistics_burden,
        "Route":                       route,
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
# 5. FORMULIR INPUT DSS
# ==============================================================================
st.markdown('<p class="section-label">📋 Formulir Asesmen Operasional</p>', unsafe_allow_html=True)

with st.form(key="dss_prediction_form", border=True):

    # ── Baris 1: Informasi Paket & Waktu ─────────────────────────────────────
    st.markdown("**📦 Data Paket & Jadwal Pengiriman**")
    col_qty, col_days, col_mode, col_type = st.columns(4)

    with col_qty:
        qty = st.number_input(
            "Jumlah Item Pesanan",
            min_value=1, max_value=500, value=3, step=1,
            help="Jumlah unit barang dalam satu transaksi pengiriman."
        )
    with col_days:
        days_scheduled = st.number_input(
            "Target Hari Pengiriman",
            min_value=0, max_value=30, value=4, step=1,
            help="Estimasi durasi pengiriman terjadwal (dalam hari kerja)."
        )
    with col_mode:
        shipping_mode = st.selectbox(
            "Moda Pengiriman",
            options=["Standard Class", "Second Class", "First Class", "Same Day"],
            index=0,
            help="Kelas layanan pengiriman yang dipilih pelanggan."
        )
    with col_type:
        day_of_week = st.selectbox(
            "Hari Pemesanan (0=Senin)",
            options=list(range(7)),
            format_func=lambda x: ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"][x],
            index=0,
            help="Hari dalam minggu saat pesanan dibuat."
        )

    st.divider()

    # ── Baris 2: Rute Pengiriman ──────────────────────────────────────────────
    st.markdown("**🗺️ Rute Pengiriman (Asal → Tujuan)**")
    col_ocity, col_ccity = st.columns(2)

    with col_ocity:
        order_city = st.text_input(
            "Kota Asal Gudang (Order City)",
            value="",
            placeholder="Contoh: Chicago",
            help="Kota asal gudang pengirim barang."
        )
    with col_ccity:
        customer_city = st.text_input(
            "Kota Tujuan Pelanggan (Customer City)",
            value="",
            placeholder="Contoh: Los Angeles",
            help="Kota tujuan pengiriman ke pelanggan akhir."
        )

    st.divider()

    # ── Baris 3: Konteks Pelanggan & Pasar ───────────────────────────────────
    st.markdown("**🌐 Segmen Pasar & Pelanggan**")
    col_seg, col_market, col_region, col_order_type = st.columns(4)

    with col_seg:
        customer_segment = st.selectbox(
            "Segmen Pelanggan",
            options=["Consumer", "Corporate", "Home Office"],
            index=0
        )
    with col_market:
        market = st.selectbox(
            "Pasar",
            options=["USCA", "Europe", "LATAM", "Pacific Asia", "Africa"],
            index=0
        )
    with col_region:
        order_region = st.selectbox(
            "Wilayah Pesanan",
            options=[
                "Western Europe", "Central America", "Oceania",
                "Eastern Asia", "West of USA", "US Center",
                "East of USA", "Canada", "Southern Asia",
                "South America", "Southeast Asia", "Eastern Europe",
                "West Africa", "Central Asia", "North Africa",
                "Southern Africa", "Middle East", "Caribbean"
            ],
            index=4
        )
    with col_order_type:
        order_type = st.selectbox(
            "Tipe Transaksi",
            options=["DEBIT", "TRANSFER", "CASH", "PAYMENT"],
            index=0
        )

    st.divider()

    # ── Baris 4: Parameter Finansial Produk ───────────────────────────────────
    st.markdown("**💰 Parameter Finansial Produk**")
    col_price, col_disc, col_profit = st.columns(3)

    with col_price:
        product_price = st.number_input(
            "Harga Satuan Produk ($)",
            min_value=0.01, max_value=5000.0, value=99.99, step=0.01,
            format="%.2f"
        )
    with col_disc:
        discount_rate = st.number_input(
            "Tingkat Diskon (0–1)",
            min_value=0.0, max_value=1.0, value=0.05, step=0.01,
            format="%.2f",
            help="Rasio diskon: 0.05 = diskon 5%."
        )
    with col_profit:
        profit_ratio = st.number_input(
            "Rasio Profit (0–1)",
            min_value=-1.0, max_value=1.0, value=0.18, step=0.01,
            format="%.2f",
            help="Margin keuntungan per item."
        )

    st.divider()

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
        try:
            df_input    = build_input_dataframe(
                qty=qty,
                days_scheduled=days_scheduled,
                order_city=order_city,
                customer_city=customer_city,
                shipping_mode=shipping_mode,
                day_of_week=day_of_week,
                order_type=order_type,
                customer_segment=customer_segment,
                market=market,
                order_region=order_region,
                product_price=product_price,
                discount_rate=discount_rate,
                profit_ratio=profit_ratio,
            )
            probability = run_prediction(df_input)
            is_risky    = probability > 0.45

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat menjalankan model: {e}")
            st.info("Periksa bahwa nama kolom pada DataFrame sudah sesuai dengan yang diharapkan encoder.")
            st.stop()

    # ── Tampilkan Hasil ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-label">📡 Hasil Asesmen Risiko Real-Time</p>', unsafe_allow_html=True)

    col_result, col_detail = st.columns([1.4, 1])

    with col_result:
        pct = probability * 100

        if is_risky:
            st.markdown(f"""
            <div class="result-danger">
                <div style="font-size:3.5rem; margin-bottom:0.5rem;">⚠️</div>
                <div style="font-size:1.6rem; font-weight:800; color:#f87171; margin-bottom:0.4rem;">
                    RISIKO KETERLAMBATAN TERDETEKSI
                </div>
                <div style="font-size:3.8rem; font-weight:900; color:#ef4444; letter-spacing:-2px;">
                    {pct:.1f}%
                </div>
                <div style="color:#fca5a5; font-size:0.88rem; margin-top:0.3rem;">
                    Probabilitas Keterlambatan · Threshold: 45.0%
                </div>
                <div style="margin-top:1rem; font-size:0.85rem; color:#fca5a5; background:rgba(239,68,68,0.1);
                     border-radius:8px; padding:0.6rem 1rem;">
                    ⚡ <strong>Rekomendasi:</strong> Lakukan intervensi segera — re-routing, eskalasi kurir,
                    atau notifikasi proaktif kepada pelanggan.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-safe">
                <div style="font-size:3.5rem; margin-bottom:0.5rem;">✅</div>
                <div style="font-size:1.6rem; font-weight:800; color:#34d399; margin-bottom:0.4rem;">
                    JADWAL PENGIRIMAN AMAN
                </div>
                <div style="font-size:3.8rem; font-weight:900; color:#10b981; letter-spacing:-2px;">
                    {pct:.1f}%
                </div>
                <div style="color:#6ee7b7; font-size:0.88rem; margin-top:0.3rem;">
                    Probabilitas Keterlambatan · Threshold: 45.0%
                </div>
                <div style="margin-top:1rem; font-size:0.85rem; color:#6ee7b7; background:rgba(16,185,129,0.1);
                     border-radius:8px; padding:0.6rem 1rem;">
                    ✔️ <strong>Rekomendasi:</strong> Pengiriman dapat dilanjutkan sesuai jadwal standar.
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_detail:
        st.markdown("**📋 Ringkasan Parameter Prediksi**")

        # Fitur yang dihitung otomatis
        logistics_burden_val = qty / (days_scheduled + 0.001)
        route_val            = f"{order_city.strip()} → {customer_city.strip()}"
        day_names            = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]

        st.markdown(f"""
        | Parameter | Nilai |
        |---|---|
        | 📍 Rute | `{route_val}` |
        | 📦 Jumlah Item | `{qty} unit` |
        | ⏱️ Target Hari | `{days_scheduled} hari` |
        | 🚛 Moda Kirim | `{shipping_mode}` |
        | 📅 Hari Pesan | `{day_names[day_of_week]}` |
        | 🔢 Beban Logistik | `{logistics_burden_val:.3f}` |
        | 👤 Segmen | `{customer_segment}` |
        | 🌐 Pasar | `{market}` |
        """)

        st.divider()

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