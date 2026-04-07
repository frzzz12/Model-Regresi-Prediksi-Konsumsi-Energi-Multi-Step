import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

# ----- 1. Konfigurasi Halaman Dasar Streamlit -----
st.set_page_config(page_title="PowerForecaster Dashboard", layout="wide")

st.markdown("<h1 style='text-align: center; color: #1f77b4;'>Dashboard Operasional Smart Grid PJM East</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2rem;'>Sistem Prediksi Beban Listrik (MW) Multi-Level Berbasis Algoritma Pembelajaran Lanjut</p>", unsafe_allow_html=True)
st.write("---")

# ----- 2. Fungsi Load Model & Scaler -----
@st.cache_resource
def load_components():
    try:
        model = joblib.load('best_model_lightgbm_multistep.pkl')
        scaler = joblib.load('scaler_features.pkl')
        return model, scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_components()

if model is None or scaler is None:
    st.error("Gagal menemukan file model .pkl ! Pastikan Anda sudah menjalankan model training.")
    st.stop()  

# ----- 3. Penataan UI Form Input -----
col_panel_1, col_panel_2 = st.columns(2, gap="large")

with col_panel_1:
    st.markdown("### Pengaturan Waktu")
    st.info("Pilih tanggal dan jam saat ini. Sistem secara otomatis mengkonversi input Anda ke kalender mesin AI.")
    
    col_w1, col_w2 = st.columns(2)
    with col_w1:
        tanggal_input = st.date_input("Kalender (Tanggal Prediksi)", datetime.date.today(), help="Klik kalender untuk memilih tanggal berjalan.")
    with col_w2:
        jam_input = st.time_input("Waktu / Jam Saat Ini", datetime.time(12, 0), help="Pilih perkiraan jam saat ini atau ketik langsung.")
        
    # --- KONVERSI CERDAS (Di belakang layar) ---
    jam_ini = jam_input.hour
    bulan = tanggal_input.month
    hari_pilihan = tanggal_input.weekday() 
    hari_ke_n = tanggal_input.timetuple().tm_yday 
    is_weekend = 1 if hari_pilihan >= 5 else 0
    
    nama_hari = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"][hari_pilihan]
    warna_status = "#e74c3c" if is_weekend else "#2ecc71"
    status_teks = "Akhir Pekan/Libur" if is_weekend else "Hari Kerja Aktif"
    
    st.markdown(f"""
    <div style='background-color:#f1f3f6; padding: 10px; border-radius: 5px; color: black;'>
    <b>Terjemahan Mesin (Verifikasi Sistem):</b><br>
    - Hari terdeteksi: <b>{nama_hari}</b> <span style='color:{warna_status};'>({status_teks})</span> <br>
    - Hari ini adalah hari ke-<b>{hari_ke_n}</b> dalam tahun ini.
    </div>
    """, unsafe_allow_html=True)


with col_panel_2:
    st.markdown("### Catatan Konsumsi Listrik Sebelumnya")
    st.info("Masukkan jejak daya dari gardu (dalam Megawatts / MW). Wilayah regional biasanya memakan sekitar 25,000 - 45,000 MW.")
    
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        lag_1 = st.number_input("Pemakaian 1 Jam Lalu", value=34000.0, step=100.0, help="Daya keseluruhan yang habis terpakai tepat 60 menit yang lalu.")
        lag_2 = st.number_input("Pemakaian 2 Jam Lalu", value=33000.0, step=100.0)
    with col_l2:
        lag_3 = st.number_input("Pemakaian 3 Jam Lalu", value=32500.0, step=100.0)
        lag_24 = st.number_input("Pemakaian Kemarin", value=34500.0, step=100.0, help="SANGAT PENTING: Daya yang terpakai 'Tepat pada jam yang sama di Hari Kemarin'.")

st.write("---")

# ----- 4. Logika Prediksi Utama Saat Tombol Ditekan -----
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

with col_btn2:
    btn_prediksi = st.button("PREDIKSI 3 JAM KE DEPAN SEKARANG", type="primary", use_container_width=True)

if btn_prediksi:
    # Blok Validasi Anti Error
    is_valid = True
    
    if any(lag < 15000 for lag in [lag_1, lag_2, lag_3, lag_24]):
        st.error("System Reject: Beban telemeter terlalu rendah (< 15,000 MW). Jika ini benar, wilayah sedang terjadi pemadaman massal (Blackout) yang tidak bisa diprediksi normal.")
        is_valid = False
        
    if any(lag > 95000 for lag in [lag_1, lag_2, lag_3, lag_24]):
        st.error("System Reject: Konsumsi melewati 95,000 MW! Mustahil menurut kapasitas teknis gardu pusat.")
        is_valid = False

    if is_valid:
        with st.spinner("Mengirim data ke Modul LightGBM MultiOutput..."):
            user_features = np.array([[jam_ini, bulan, hari_pilihan, hari_ke_n, is_weekend, lag_1, lag_2, lag_3, lag_24]])
            
            try:
                user_features_scaled = scaler.transform(user_features)
                hasil_target = model.predict(user_features_scaled)[0]
            except Exception as e:
                st.error(f"Sistem gagal mengeksekusi model: {e}")
                st.stop()
                
        # Menampilkan Dasbor Hasil
        st.write("---")
        st.markdown("<h2 style='text-align: center;'>Hasil Proyeksi Kebutuhan Kelistrikan</h2>", unsafe_allow_html=True)
        st.write("")
        
        res_col1, res_col2, res_col3 = st.columns(3)
        
        jam_t1 = (jam_ini + 1) % 24
        jam_t2 = (jam_ini + 2) % 24
        jam_t3 = (jam_ini + 3) % 24
        
        with res_col1:
            st.info(f"**1 JAM KE DEPAN** Pukul {jam_t1:02d}:00")
            st.metric(label="Estimasi Kebutuhan", value=f"{hasil_target[0]:,.1f} MW", delta=f"{hasil_target[0]-lag_1:+,.1f} MW (naik turun thd sejam lalu)")
            
        with res_col2:
            st.warning(f"**2 JAM KE DEPAN** Pukul {jam_t2:02d}:00")
            st.metric(label="Estimasi Kebutuhan", value=f"{hasil_target[1]:,.1f} MW", delta=f"{hasil_target[1]-hasil_target[0]:+,.1f} MW (naik turun thd jam lalu)")
            
        with res_col3:
            st.error(f"**3 JAM KE DEPAN** Pukul {jam_t3:02d}:00")
            st.metric(label="Estimasi Kebutuhan", value=f"{hasil_target[2]:,.1f} MW", delta=f"{hasil_target[2]-hasil_target[1]:+,.1f} MW (naik turun thd jam lalu)")

st.markdown("<br><br><p style='text-align: center; color: gray; font-size: 0.9em;'>© 2026 Muhammad Rizky Yamin</p>", unsafe_allow_html=True)
