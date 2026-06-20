import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# ==============================================================================
# KONFIGURASI HALAMAN
# ==============================================================================
st.set_page_config(
    page_title="AI Risk Monitor | Sistem Prediksi Keterlambatan Logistik",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Injeksi CSS Kustom untuk tampilan profesional
st.markdown("""
<style>
    /* Font & Background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Header Utama */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border-left: 5px solid #e94560;
    }
    .main-header h1 { color: #ffffff; margin: 0; font-size: 2rem; font-weight: 700; }
    .main-header p  { color: #a8b2d8; margin: 0.5rem 0 0 0; font-size: 0.95rem; }

    /* Badge Mode */
    .badge-audit {
        display: inline-block;
        background: linear-gradient(90deg, #11998e, #38ef7d);
        color: white; padding: 4px 14px;
        border-radius: 20px; font-size: 0.78rem; font-weight: 600;
        margin-bottom: 1rem;
    }
    .badge-predict {
        display: inline-block;
        background: linear-gradient(90deg, #4776E6, #8E54E9);
        color: white; padding: 4px 14px;
        border-radius: 20px; font-size: 0.78rem; font-weight: 600;
        margin-bottom: 1rem;
    }

    /* Kartu Metrik Kustom */
    .metric-card {
        background: #1e2130;
        border: 1px solid #2d3250;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-card .label { color: #8892b0; font-size: 0.82rem; font-weight: 500; }
    .metric-card .value { color: #ccd6f6; font-size: 1.8rem; font-weight: 700; margin-top: 4px; }
    .metric-card .delta { font-size: 0.8rem; margin-top: 4px; }
    .delta-pos { color: #38ef7d; }
    .delta-neg { color: #e94560; }

    /* Info Box */
    .info-box {
        background: rgba(67, 97, 238, 0.1);
        border: 1px solid rgba(67, 97, 238, 0.4);
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        color: #a8b2d8;
        font-size: 0.88rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #0d1117; }
    [data-testid="stSidebar"] .stNumberInput label { color: #8892b0 !important; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. MUAT ARTEFAK MODEL & ENCODER (CACHED)
# ==============================================================================
@st.cache_resource
def load_artifacts():
    """Memuat model XGBoost dan encoder rute dari file .pkl."""
    try:
        model   = joblib.load('model_logistik.pkl')
        encoder = joblib.load('encoder_rute.pkl')
        return model, encoder
    except FileNotFoundError as e:
        st.error(f"❌ Artefak tidak ditemukan: {e}. Pastikan file `.pkl` ada di direktori yang sama dengan `app.py`.")
        return None, None
    except Exception as e:
        st.error(f"❌ Gagal memuat artefak model/encoder. Detail: {e}")
        return None, None

model, encoder = load_artifacts()

# ==============================================================================
# 2. HEADER APLIKASI
# ==============================================================================
st.markdown("""
<div class="main-header">
    <h1>🚚 AI Risk Monitor: Sistem Deteksi Keterlambatan Logistik</h1>
    <p>
        Platform intelijen risiko berbasis <strong>XGBoost Teroptimasi</strong> (Threshold Intervensi: <strong>0.45</strong>)
        untuk mendeteksi potensi keterlambatan pengiriman paket secara akurat dan mengukur dampak finansialnya.
    </p>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. SIDEBAR — PARAMETER BISNIS (INPUT CBA)
# ==============================================================================
with st.sidebar:
    st.markdown("## ⚙️ Parameter Audit SLA")
    st.markdown("---")
    st.markdown("**💰 Komponen Biaya Operasional**")

    cost_SLA = st.number_input(
        "Denda SLA per Paket Terlambat ($)",
        value=50.0, step=5.0, min_value=0.0,
        help="Biaya penalti kontrak SLA yang harus dibayar per 1 paket yang benar-benar terlambat dan tidak tertangani."
    )
    cost_FP = st.number_input(
        "Biaya Alarm Palsu / False Positive ($)",
        value=5.0, step=1.0, min_value=0.0,
        help="Biaya lembur atau intervensi yang terbuang sia-sia karena AI salah memprediksi paket aman sebagai berbahaya."
    )
    cost_TP = st.number_input(
        "Biaya Intervensi Berhasil / True Positive ($)",
        value=5.0, step=1.0, min_value=0.0,
        help="Biaya operasional (lembur kurir, re-routing) yang dikeluarkan saat AI berhasil mendeteksi dan menangani paket berbahaya."
    )
    st.markdown("---")
    st.markdown(
        '<div class="info-box">ℹ️ Parameter ini digunakan untuk menghitung '
        '<strong>Cost-Benefit Analysis (CBA)</strong> pada <em>Mode Audit Evaluasi</em>.</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.caption("© 2025 AI Risk Monitor · Dibuat untuk keperluan riset logistik akademis.")

# ==============================================================================
# 4. AREA UNGGAH DATASET
# ==============================================================================
st.markdown("### 📂 Unggah Manifest Pengiriman")
uploaded_file = st.file_uploader(
    "Unggah file berekstensi `.csv` yang berisi data transaksi pengiriman",
    type=["csv"],
    help="Sistem akan otomatis mendeteksi apakah data Anda memiliki kolom target (Late_delivery_risk) "
         "dan beralih ke mode yang sesuai."
)

# ==============================================================================
# 5. FUNGSI PIPELINE FEATURE ENGINEERING
# ==============================================================================
def run_feature_engineering(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Menjalankan seluruh pipeline rekayasa fitur:
    eliminasi noise, pembuatan fitur waktu, beban logistik, dan sintesis rute.

    Parameters
    ----------
    df_input : pd.DataFrame
        DataFrame mentah dari file CSV yang diunggah pengguna.

    Returns
    -------
    pd.DataFrame
        DataFrame siap-prediksi setelah melewati seluruh transformasi.
    """
    # --- Eliminasi Kolom Noise & Target Leakage ---
    kolom_buang = [
        'Days for shipping (real)', 'Delivery Status',
        'shipping date (DateOrders)', 'Order Status',
        'Customer Email', 'Customer Password',
        'Customer Fname', 'Customer Lname',
        'Customer Street', 'Customer Zipcode',
        'Order Id', 'Customer Id', 'Order Customer Id',
        'Product Card Id', 'Order Item Id', 'Order Item Cardprod Id',
        'Department Id', 'Category Id', 'Product Category Id',
        'Product Image', 'Product Description', 'Product Status',
        'Order Zipcode', 'Late_delivery_risk'
    ]
    df_clean = df_input.drop(columns=kolom_buang, errors='ignore')

    # --- Fitur Temporal: Hari Pemesanan dalam Seminggu ---
    if 'order date (DateOrders)' in df_input.columns:
        try:
            order_dates = pd.to_datetime(df_input['order date (DateOrders)'], infer_datetime_format=True, errors='coerce')
            df_clean['Order_DayOfWeek'] = order_dates.dt.dayofweek
        except Exception:
            df_clean['Order_DayOfWeek'] = 0
    else:
        df_clean['Order_DayOfWeek'] = 0

    # --- Fitur Derivatif: Indeks Beban Logistik Gudang ---
    if ('Order Item Quantity' in df_input.columns and
            'Days for shipment (scheduled)' in df_input.columns):
        df_clean['Logistics_Burden'] = (
            df_input['Order Item Quantity'] /
            (df_input['Days for shipment (scheduled)'] + 0.001)
        )
    else:
        df_clean['Logistics_Burden'] = 0.0

    # --- Fitur Sintesis: Rute Pengiriman (Asal → Tujuan) ---
    if ('Order City' in df_input.columns and
            'Customer City' in df_input.columns):
        df_clean['Route'] = (
            df_input['Order City'].astype(str) + " to " +
            df_input['Customer City'].astype(str)
        )
    else:
        df_clean['Route'] = "Unknown to Unknown"

    return df_clean


# ==============================================================================
# 6. FUNGSI KALKULASI CBA (COST-BENEFIT ANALYSIS)
# ==============================================================================
def calculate_cba(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_sla: float,
    cost_fp: float,
    cost_tp: float
) -> dict:
    """
    Menghitung analisis biaya-manfaat (CBA) berbasis matriks konfusi.

    Rumus:
        Cost_Tanpa_AI  = (TP + FN) × cost_SLA
        Cost_Dengan_AI = (FN × cost_SLA) + (FP × cost_FP) + (TP × cost_TP)
        Uang_Diselamatkan = Cost_Tanpa_AI - Cost_Dengan_AI
        ROI_AI (%) = (Uang_Diselamatkan / Cost_Tanpa_AI) × 100
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    total_telat_aktual = int(tp + fn)
    cost_tanpa_ai      = total_telat_aktual * cost_sla
    cost_dengan_ai     = (fn * cost_sla) + (fp * cost_fp) + (tp * cost_tp)
    uang_diselamatkan  = cost_tanpa_ai - cost_dengan_ai
    roi_persen         = (uang_diselamatkan / cost_tanpa_ai * 100) if cost_tanpa_ai > 0 else 0.0

    return {
        "TP": int(tp), "FP": int(fp),
        "TN": int(tn), "FN": int(fn),
        "total_telat_aktual": total_telat_aktual,
        "cost_tanpa_ai":      cost_tanpa_ai,
        "cost_dengan_ai":     cost_dengan_ai,
        "uang_diselamatkan":  uang_diselamatkan,
        "roi_persen":         roi_persen
    }


# ==============================================================================
# 7. FUNGSI STYLING TABEL (HIGHLIGHT BARIS RISIKO)
# ==============================================================================
def highlight_risk(row):
    """Memberi warna merah pada baris yang diprediksi berisiko terlambat."""
    if row.get('AI_Prediction_Status', '') == '⚠️ RISIKO TERLAMBAT':
        return ['background-color: rgba(233, 69, 96, 0.15); color: #ff6b8a'] * len(row)
    return [''] * len(row)


# ==============================================================================
# 8. MAIN LOGIC — DIEKSEKUSI SAAT FILE DIUNGGAH
# ==============================================================================
if uploaded_file is not None and model is not None:
    try:
        df_input = pd.read_csv(uploaded_file, encoding='latin1')
        st.success(f"✅ Berhasil memuat **{df_input.shape[0]:,} baris** × **{df_input.shape[1]} kolom** data manifest.")

        # Deteksi otomatis: apakah kolom target tersedia?
        HAS_TARGET = 'Late_delivery_risk' in df_input.columns
        MODE       = "A" if HAS_TARGET else "B"

        # Tampilkan badge mode aktif
        if MODE == "A":
            st.markdown('<span class="badge-audit">📊 MODE A: Audit Evaluasi Model (Data Historis Berlabel)</span>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-predict">🔮 MODE B: Prediksi Buta (Manifest Operasional Baru)</span>',
                        unsafe_allow_html=True)

        with st.expander("🔎 Pratinjau Data Mentah (10 Baris Pertama)"):
            st.dataframe(df_input.head(10), use_container_width=True)

        # ----- PIPELINE FEATURE ENGINEERING -----
        with st.spinner("⚙️ Menjalankan pipeline rekayasa fitur..."):
            df_clean   = run_feature_engineering(df_input)
            X_encoded  = encoder.transform(df_clean)

        # ----- INFERENSI MODEL -----
        with st.spinner("🤖 Mesin AI sedang memetakan probabilitas risiko..."):
            y_pred_proba  = model.predict_proba(X_encoded)[:, 1]
            y_pred_custom = np.where(y_pred_proba > 0.45, 1, 0)

        # Injeksi hasil ke DataFrame utama
        df_result = df_input.copy()
        df_result['Risk_Probability']    = y_pred_proba
        df_result['AI_Prediction_Status'] = np.where(
            y_pred_custom == 1, '⚠️ RISIKO TERLAMBAT', '✅ AMAN'
        )

        # ==================== TAB LAYOUT ====================
        if MODE == "A":
            tab1, tab2, tab3 = st.tabs([
                "📋 Ringkasan Eksekutif",
                "💰 Analisis Finansial (CBA)",
                "🗃️ Tabel Data Rinci"
            ])
        else:
            tab1, tab3 = st.tabs([
                "📋 Ringkasan Eksekutif",
                "🗃️ Tabel Data Rinci"
            ])

        # ============================================================
        # TAB 1 — RINGKASAN EKSEKUTIF
        # ============================================================
        with tab1:
            total_paket = len(y_pred_custom)
            total_telat = int(np.sum(y_pred_custom))
            rasio_telat = (total_telat / total_paket) * 100

            st.markdown("#### 📊 Metrik Risiko Operasional")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Pengiriman Dianalisis",  f"{total_paket:,} Paket")
            c2.metric("Paket Terindikasi Berbahaya",  f"{total_telat:,} Paket",
                      f"{rasio_telat:.1f}% dari total")
            c3.metric("Paket Dinyatakan Aman",
                      f"{total_paket - total_telat:,} Paket",
                      f"{100 - rasio_telat:.1f}% dari total", delta_color="off")

            if MODE == "A":
                st.divider()
                y_true = df_input['Late_delivery_risk'].values
                acc    = accuracy_score(y_true, y_pred_custom) * 100
                rec    = recall_score(y_true, y_pred_custom)   * 100
                f1     = f1_score(y_true, y_pred_custom)       * 100

                st.markdown("#### 🎯 Metrik Kinerja Model (Evaluasi Akurasi)")
                m1, m2, m3 = st.columns(3)
                m1.metric("Akurasi Model",   f"{acc:.2f}%")
                m2.metric("Recall (Sensitivitas)", f"{rec:.2f}%",
                          help="Kemampuan model mendeteksi SEMUA paket yang benar-benar terlambat.")
                m3.metric("F1-Score",        f"{f1:.2f}%",
                          help="Keseimbangan antara Precision dan Recall.")

                st.divider()

                # --- Confusion Matrix (Heatmap Plotly) ---
                st.markdown("#### 🗺️ Matriks Konfusi")
                cm   = confusion_matrix(y_true, y_pred_custom)
                fig_cm = px.imshow(
                    cm,
                    labels=dict(x="Prediksi AI", y="Realitas Aktual", color="Jumlah"),
                    x=["Prediksi: AMAN (0)", "Prediksi: TERLAMBAT (1)"],
                    y=["Aktual: AMAN (0)",   "Aktual: TERLAMBAT (1)"],
                    color_continuous_scale="Blues",
                    text_auto=True,
                    aspect="auto"
                )
                fig_cm.update_layout(
                    title="Confusion Matrix — XGBoost (Threshold: 0.45)",
                    plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                    font=dict(color="#ccd6f6"),
                    height=380
                )
                st.plotly_chart(fig_cm, use_container_width=True)

            st.divider()

            # --- Distribusi Probabilitas Risiko ---
            st.markdown("#### 📈 Distribusi Probabilitas Risiko Pengiriman")
            fig_dist = px.histogram(
                df_result, x="Risk_Probability",
                nbins=50,
                color="AI_Prediction_Status",
                color_discrete_map={
                    "⚠️ RISIKO TERLAMBAT": "#e94560",
                    "✅ AMAN":             "#38ef7d"
                },
                labels={"Risk_Probability": "Probabilitas Keterlambatan", "count": "Jumlah Paket"},
                title="Histogram Distribusi Skor Risiko AI (Threshold = 0.45)",
                opacity=0.85
            )
            fig_dist.add_vline(
                x=0.45, line_dash="dash", line_color="#f5a623",
                annotation_text="Threshold: 0.45",
                annotation_position="top right",
                annotation_font_color="#f5a623"
            )
            fig_dist.update_layout(
                plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                font=dict(color="#ccd6f6"),
                legend_title_text="Status Pengiriman",
                height=420
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        # ============================================================
        # TAB 2 — ANALISIS FINANSIAL CBA (MODE A ONLY)
        # ============================================================
        if MODE == "A":
            with tab2:
                y_true = df_input['Late_delivery_risk'].values
                cba    = calculate_cba(y_true, y_pred_custom, cost_SLA, cost_FP, cost_TP)

                st.markdown("#### 🔢 Komponen Matriks Konfusi")
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("True Positive  (TP)", f"{cba['TP']:,}",
                          help="Paket terlambat yang BERHASIL dideteksi dan ditangani AI.")
                k2.metric("False Positive (FP)", f"{cba['FP']:,}",
                          help="Paket aman yang salah diprediksi berbahaya (alarm palsu).")
                k3.metric("True Negative  (TN)", f"{cba['TN']:,}",
                          help="Paket aman yang dengan benar diklasifikasikan sebagai aman.")
                k4.metric("False Negative (FN)", f"{cba['FN']:,}",
                          help="Paket terlambat yang LOLOS dari deteksi AI (miss).")

                st.divider()

                st.markdown("#### 💵 Perbandingan Total Biaya Operasional")

                # Tampilkan formula yang digunakan
                with st.expander("📐 Lihat Rumus Kalkulasi CBA yang Digunakan"):
                    st.latex(r"""
                        \text{Cost}_{\text{Tanpa AI}} = (\text{TP} + \text{FN}) \times \text{cost\_SLA}
                    """)
                    st.latex(r"""
                        \text{Cost}_{\text{Dengan AI}} = (\text{FN} \times \text{cost\_SLA}) +
                        (\text{FP} \times \text{cost\_FP}) + (\text{TP} \times \text{cost\_TP})
                    """)
                    st.latex(r"""
                        \text{Uang Diselamatkan} = \text{Cost}_{\text{Tanpa AI}} - \text{Cost}_{\text{Dengan AI}}
                    """)

                f1, f2, f3 = st.columns(3)
                f1.metric(
                    "💸 Biaya Tanpa AI",
                    f"${cba['cost_tanpa_ai']:,.2f}",
                    help=f"Asumsi semua {cba['total_telat_aktual']:,} paket terlambat kena denda penuh @ ${cost_SLA}"
                )
                f2.metric(
                    "🤖 Biaya Dengan AI",
                    f"${cba['cost_dengan_ai']:,.2f}",
                    help="Total biaya aktual dengan intervensi AI (meliputi FN lolos + FP alarm palsu + TP ditangani)."
                )
                delta_sign = "+" if cba['uang_diselamatkan'] >= 0 else ""
                f3.metric(
                    "✅ Anggaran Diselamatkan",
                    f"${cba['uang_diselamatkan']:,.2f}",
                    f"ROI AI: {delta_sign}{cba['roi_persen']:.1f}%",
                    delta_color="normal"
                )

                st.divider()

                # --- Bar Chart Perbandingan Biaya ---
                st.markdown("#### 📊 Visualisasi Perbandingan Biaya")
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    name="Tanpa Intervensi AI",
                    x=["Skenario Operasional"],
                    y=[cba['cost_tanpa_ai']],
                    marker_color="#e94560",
                    text=[f"${cba['cost_tanpa_ai']:,.0f}"],
                    textposition="outside"
                ))
                fig_bar.add_trace(go.Bar(
                    name="Dengan Intervensi AI",
                    x=["Skenario Operasional"],
                    y=[cba['cost_dengan_ai']],
                    marker_color="#38ef7d",
                    text=[f"${cba['cost_dengan_ai']:,.0f}"],
                    textposition="outside"
                ))
                fig_bar.update_layout(
                    title="Perbandingan Total Biaya Operasional: Tanpa AI vs. Dengan AI",
                    yaxis_title="Total Biaya (USD)",
                    barmode="group",
                    plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                    font=dict(color="#ccd6f6"),
                    legend=dict(bgcolor="#1e2130", bordercolor="#2d3250", borderwidth=1),
                    height=420
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                # --- Waterfall Chart Dekomposisi Biaya AI ---
                st.markdown("#### 🌊 Dekomposisi Komponen Biaya Sistem AI")
                fig_wf = go.Figure(go.Waterfall(
                    name="Dekomposisi CBA",
                    orientation="v",
                    measure=["absolute", "relative", "relative", "relative", "total"],
                    x=[
                        f"Baseline<br>(Tanpa AI)",
                        f"Penghematan<br>TP ({cba['TP']:,} paket)",
                        f"Biaya FN<br>({cba['FN']:,} paket lolos)",
                        f"Biaya FP<br>({cba['FP']:,} alarm palsu)",
                        "Total Biaya<br>Dengan AI"
                    ],
                    y=[
                        cba['cost_tanpa_ai'],
                        -(cba['TP'] * (cost_SLA - cost_TP)),
                        0.0,
                        cba['FP'] * cost_FP,
                        None
                    ],
                    text=[
                        f"${cba['cost_tanpa_ai']:,.0f}",
                        f"-${cba['TP'] * (cost_SLA - cost_TP):,.0f}",
                        f"$0",
                        f"+${cba['FP'] * cost_FP:,.0f}",
                        f"${cba['cost_dengan_ai']:,.0f}"
                    ],
                    textposition="outside",
                    connector={"line": {"color": "#2d3250"}},
                    increasing=dict(marker=dict(color="#e94560")),
                    decreasing=dict(marker=dict(color="#38ef7d")),
                    totals=dict(marker=dict(color="#4776E6"))
                ))
                fig_wf.update_layout(
                    title="Waterfall Chart — Alur Penghematan Biaya dengan Sistem AI",
                    yaxis_title="Akumulasi Biaya (USD)",
                    plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                    font=dict(color="#ccd6f6"),
                    height=460
                )
                st.plotly_chart(fig_wf, use_container_width=True)

        # ============================================================
        # TAB 3 — TABEL DATA RINCI
        # ============================================================
        with tab3:
            st.markdown("#### 🔍 Filter & Telusuri Hasil Keputusan Sistem AI")

            col_filter, col_sort = st.columns([2, 1])
            with col_filter:
                status_filter = st.selectbox(
                    "Filter Status Pengiriman:",
                    ["Semua Data", "⚠️ RISIKO TERLAMBAT", "✅ AMAN"]
                )
            with col_sort:
                sort_order = st.radio("Urutan Probabilitas:", ["Tertinggi → Terendah", "Terendah → Tertinggi"],
                                      horizontal=True)

            # Terapkan filter
            if status_filter != "Semua Data":
                df_display = df_result[df_result['AI_Prediction_Status'] == status_filter].copy()
            else:
                df_display = df_result.copy()

            # Terapkan pengurutan
            ascending = (sort_order == "Terendah → Tertinggi")
            df_display = df_display.sort_values(by='Risk_Probability', ascending=ascending)

            # Pilih kolom yang relevan untuk ditampilkan
            kolom_tampil = [col for col in [
                'Order Id', 'Customer City', 'Order City', 'Product Name',
                'Days for shipment (scheduled)', 'Order Item Quantity',
                'Risk_Probability', 'AI_Prediction_Status'
            ] if col in df_display.columns]

            df_styled = df_display[kolom_tampil].style.apply(highlight_risk, axis=1).format(
                {"Risk_Probability": "{:.4f}"}
            )

            st.markdown(f"**Menampilkan {len(df_display):,} dari {len(df_result):,} data.**")
            st.dataframe(df_styled, use_container_width=True, height=480)

            st.divider()

            # --- Unduh Laporan CSV ---
            st.markdown("#### 📥 Unduh Dokumen Hasil Audit Risiko")
            csv_output = df_result.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Unduh File CSV Hasil Prediksi AI",
                data=csv_output,
                file_name="Laporan_Prediksi_Risiko_Logistik.csv",
                mime="text/csv",
                help="Mengunduh seluruh data beserta kolom Risk_Probability dan AI_Prediction_Status."
            )

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat memproses file CSV: {e}")
        st.info(
            "ℹ️ Pastikan file CSV yang Anda unggah memiliki struktur kolom yang kompatibel "
            "dengan dataset DataCo Supply Chain yang digunakan pada fase pelatihan model."
        )

# ==============================================================================
# 9. LAYAR SAMBUTAN (SAAT BELUM ADA FILE DIUNGGAH)
# ==============================================================================
else:
    if model is None:
        st.warning("⚠️ Model tidak berhasil dimuat. Periksa keberadaan file `model_logistik.pkl` dan `encoder_rute.pkl`.")
    else:
        st.markdown("""
        <div class="info-box">
        <strong>📌 Panduan Penggunaan Sistem:</strong><br><br>
        Unggah file <code>.csv</code> manifest pengiriman di panel di atas untuk memulai analisis.<br><br>
        Sistem akan secara otomatis mendeteksi tipe data dan mengaktifkan mode yang sesuai:
        <ul>
            <li>📊 <strong>Mode A – Audit Evaluasi</strong>: Aktif jika CSV berisi kolom <code>Late_delivery_risk</code>.
                Menghasilkan metrik akurasi, confusion matrix, dan analisis CBA penuh.</li>
            <li>🔮 <strong>Mode B – Prediksi Buta</strong>: Aktif untuk manifest operasional baru tanpa label.
                Menghasilkan tabel probabilitas risiko dan status setiap paket.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            **📊 Mode A — Audit Evaluasi Model**
            - ✅ Akurasi, Recall, F1-Score
            - ✅ Confusion Matrix Interaktif
            - ✅ Cost-Benefit Analysis (CBA) Berbasis TP/FP/TN/FN
            - ✅ Waterfall Chart Dekomposisi Biaya
            - ✅ Distribusi Probabilitas Risiko
            """)
        with col_b:
            st.markdown("""
            **🔮 Mode B — Prediksi Manifest Baru**
            - ✅ Prediksi Real-time Setiap Paket
            - ✅ Skor Probabilitas Keterlambatan
            - ✅ Filter & Pencarian Data Interaktif
            - ✅ Highlight Baris Paket Berisiko
            - ✅ Unduh Laporan CSV Lengkap
            """)