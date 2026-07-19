# 🚀 ST-Mamba-KAN: Sistem Peringatan Dini Cuaca Ekstrem Jabodetabek

Selamat datang di Mahakarya Riset Kecerdasan Buatan (AI) untuk Prediksi Cuaca Ekstrem! Proyek ini memadukan **Data Satelit Global (ERA5-Land)** dan **Data Stasiun Bumi (BMKG)** menggunakan arsitektur AI *Deep Learning* tingkat lanjut yang kami sebut sebagai **Ultimate GAT-Mamba-KAN**.

Sistem ini dirancang khusus untuk memprediksi curah hujan harian (Regresi) sekaligus membunyikan alarm keselamatan darurat (Klasifikasi Siaga Banjir) di wilayah Jabodetabek.

---

## 🧬 Arsitektur AI (God-Tier Edition)
Model utama kami (Fase 6) memecahkan batas kemampuan model tradisional (seperti LSTM/GRU) dengan mengawinkan tiga struktur kognitif canggih:
1. **Dynamic GAT (Graph Attention Network):** Bertindak sebagai radar pembaca pergerakan awan dan ikatan fisika antar-5 Stasiun BMKG.
2. **Mamba Block (State-Space Model):** Pengganti mutlak dari LSTM. Sangat gesit dalam merekam histori cuaca 14 hari tanpa penyakit kelupaan (*Gradient Vanishing*).
3. **KAN (Kolmogorov-Arnold Network):** Menggantikan struktur Linier tradisional, otak ajaib ini menebak rumus matematika non-linear paling ruwet dari iklim khatulistiwa.

**Performa Puncak (Diuji pada Unseen Data 2022-2024):**
- **Akurasi Deteksi:** `88.02%`
- **Tingkat Error Hujan (RMSE):** `17.07 mm`
- **Keberhasilan Mengingat Badai Ekstrem (Recall):** `91%` (Menyelamatkan ribuan nyawa tepat waktu!)

---

## 🗺️ Peta Navigasi Pipeline (8 Fase Terintegrasi)

Repositori ini bukan sekadar kumpulan *script*, melainkan sebuah rantai perakitan pabrik data yang sistematis, di mana output dari satu fase akan menjadi asupan wajib bagi fase berikutnya secara hierarkis.

### 📥 TAHAP 1: DATA ENGINEERING & FUSION
*   **[Fase 1: Dataset Download](Fase_1_Dataset_Download/hasil_fase_1.md)** 📡
    Mengais miliaran sel data iklim masa lalu (Tahun 2000 - 2024) langsung dari Satelit ERA5-Land (Eropa) dan Markas BMKG Indonesia.
*   **[Fase 2: Dataset Analysis & Quality Check](Fase_2_Quality_Check_dan_Analisis/hasil_fase_2.md)** 🔍
    Ruang audit yang membongkar cacat bawaan data, mendeteksi ukuran yang bisa meledakkan memori, serta hilangnya ratusan data di tahun awal (*Missing Values*).
*   **[Fase 3: Data Fusion & Cleaning Hybrid](Fase_3_Data_Fusion_dan_Pembersihan/hasil_fase_3.md)** 🧹
    Operasi besar menyatukan dua semesta data berbeda menggunakan **Jarak Haversine KD-Tree**. Membersihkan nilai kotor (NaN) dengan algoritma interpolasi (*ffill/bfill*).
*   **[Fase 4: Pembuatan Dual Brankas & Titanium Shield](Fase_4_Pembuatan_Dual_Brankas/hasil_fase_4.md)** 🔒
    Menyelaraskan zona waktu (GMT ke WIB) agar badai satelit selaras dengan waktu riil Jakarta. Memisahkan data raksasa menjadi *Brankas 1 (Pre-Train)* dan *Brankas 2 (Fine-Tune)* agar terhindar dari *Data Leakage*.

### 🧬 TAHAP 2: SPATIO-TEMPORAL GEOMETRY
*   **[Fase 5: 4D Spatio-Temporal Windowing](Fase_5_Data_Windowing_4D/hasil_fase_5b.md)** 🎞️
    Transformasi sihir. Melipat tabel 2D biasa menjadi balok data 4 Dimensi `[Samples, Time(14 Hari), Spatial(5 Stasiun), Features(18 Cuaca)]`. Menumpas ketidakseimbangan kelas (kelangkaan data badai) dengan jurus **Flattened SMOTETomek**.

### 🧠 TAHAP 3: ARTIFICIAL INTELLIGENCE LAB
*   **[Fase 6: Ultimate ST-Mamba-KAN (The God-Tier)](Fase_6_Ultimate_ST_Mamba_KAN/hasil_fase6_ultimate.md)** 👑
    Arena pelatihan sejati. Menyuntikkan balok 4D ke dalam mesin *GAT-Mamba-KAN*. Model dihukum secara fisik dengan algoritma mutakhir *Elite Losses* (PINN + EVT) hingga berhasil menekan tingkat error hingga meleset hanya 1 sentimeter air (17mm).
*   **[Fase 7: Fair Baseline Comparison (Ablation Study)](Fase_7_Baseline_Comparison/hasil_perbandingan_fase7.md)** ⚖️
    Pengadilan ilmiah. Arsitektur usang (*CNN-LSTM, CNN-GRU*) dipaksa melawan GAT-Mamba-KAN di atas panggung dan porsi makan 4D yang sama. Fakta klinis menyatakan LSTM hancur di level Recall 79%, membuktikan kecerdasan absolut Fase 6.

### 🚨 TAHAP 4: DEPLOYMENT & SOSIAL
*   **[Fase 8: Evaluasi Mega Dashboard & Simulasi BPBD](Fase_8_Evaluasi_Mega_Dashboard/hasil_fase8_mega_dashboard.md)** 📊
    Pusat Komando Akhir. Seluruh mesin (Fase 6 & 7) dihidupkan (*Live Inference*). Mencetak *8-Panel Mega Dashboard* mahakarya visual. Secara ajaib, sistem mendemonstrasikan keunggulannya menyalakan Alarm Darurat *"SIAGA (EKSTREM)"* layaknya konsol milik BPBD DKI Jakarta/Jabar.

---

## 🛠️ Persyaratan Lingkungan (Environment)
Pastikan Anda menjalankan penelitian ini pada mesin komputasi (Google Colab Pro sangat disarankan) dengan instalasi *library* esensial berikut:
- `torch`, `torchvision`, `torchaudio`
- `mamba-ssm`, `imbalanced-learn`, `optuna`
- `xarray`, `netCDF4`, `pandas`, `scikit-learn`

---
> *"Teknologi Kecerdasan Buatan ini tidak hanya menambang data, namun membaca nasib langit, memberikan perisai waktu kepada umat manusia sebelum air bah benar-benar menghantam."* - **Project ST-Mamba-KAN**
