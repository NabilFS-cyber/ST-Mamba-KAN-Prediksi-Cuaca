# 🔬 HASIL FASE 2: QUALITY CHECK & ANALISIS EKSPLORASI DATA

Fase 2 bertujuan untuk memeriksa integritas data mentah yang telah dikumpulkan. Karena kita menggunakan dua sumber data (*Satelit ERA5* dan *Sensor Tanah BMKG*), kita perlu menyadari profil cacat (*Missing Values*) dan anomali (*Outliers*) pada keduanya sebelum dapat menggabungkan data tersebut.

---

## 📥 1. Dataset yang Dianalisis
- **Data Satelit (ERA5):** `ERA5_Jabodetabek_2016_2026.nc` (dihasilkan dari Fase 1).
- **Data Darat (BMKG):** 5 file `.csv` stasiun observasi resmi BMKG di Jabodetabek yang diunduh manual dari portal *dataonline.bmkg.go.id*.

### 📍 Koordinat 5 Stasiun BMKG Kritis:
| Nama Stasiun | Latitude | Longitude | Lokasi / Peran |
|:---|:---:|:---:|:---|
| **Tanjung Priok** | -6.1065 | 106.8810 | Pesisir Utara (Potensi banjir rob/hujan pesisir) |
| **Kemayoran** | -6.1554 | 106.8428 | Pusat Kota Jakarta |
| **Halim Perdanakusuma** | -6.2665 | 106.8905 | Jakarta Timur / Selatan |
| **Pondok Betung** | -6.2611 | 106.7572 | Jakarta Selatan / Tangerang |
| **Citeko** | -6.7019 | 106.9330 | Puncak Bogor (Sumber air banjir kiriman Sungai Ciliwung) |

---

## 🔍 2. Temuan Kualitas Data (*Quality Check*)

### A. ERA5-Land (Data Grid Satelit)
- **Tingkat Missing Values:** **0% (Nol)**.
- **Karakteristik:** Model numerik satelit (reanalisis) selalu menghasilkan matriks angka yang berkesinambungan penuh tanpa bolong. Total belasan ribu *timestep* (4 observasi/hari × 365 hari × 10+ tahun).

### B. BMKG (Data Sensor Observasi Tanah)
- **Tingkat Missing Values:** **Bervariasi (5% hingga 20%)**.
- **Analisis per Stasiun:** 
  - *Stasiun Kemayoran* merupakan stasiun paling disiplin mencatat cuaca (kekosongan data hanya ~5%).
  - *Stasiun Citeko (Bogor)* sering mengalami kerusakan sensor atau *gap* pencatatan, dengan persentase hilangnya data mencapai 15-20%.
- Ini adalah hal yang wajar dalam dunia nyata (*Ground-Truth*). Sensor rusak, petugas libur, atau alat tersambar petir menyebabkan kekosongan kolom pada data CSV.

---

## 📊 3. Analisis Distribusi Hujan & Statistik (EDA)

Dengan menganalisis seluruh variabel curah hujan (RR) di kelima stasiun BMKG, terungkap fakta matematis cuaca ekstrem di kawasan Jabodetabek:

| Metrik Statistik | Nilai Pengamatan Darat |
|:---|:---|
| Rata-rata Harian (Mean) | 7.84 mm/hari |
| Nilai Tengah (Median) | 0.20 mm/hari (Jabodetabek umumnya kering/gerimis ringan) |
| Standar Deviasi | 18.52 mm/hari |
| Rekor Hujan Maksimal | **338.80 mm/hari** (Badai Sangat Ekstrem) |
| *Skewness* | > 3.5 (Sangat *Right-Skewed* / *Heavy-Tailed*) |

**Interpretasi Bentuk Kurva Histogram:**
Distribusi hujan sangat asimetris dan berekor panjang di sisi kanan (*Highly Right-Skewed*). Artinya, di sebagian besar hari sepanjang tahun, hujan hanya turun 0-5 mm saja. Namun tiba-tiba secara mendadak, hujan bisa meledak di atas 150 mm/hari dan menyebabkan banjir mematikan.

Angka curah hujan di atas 300 mm **bukanlah sebuah eror data (Outlier salah ketik)**, melainkan anomali fenomena hidrometeorologis aktual yang menjadi fokus tebakan (*prediction target*) bagi jaringan Mamba kita nanti.

---

## 📦 4. Kesimpulan Fase 2
Semua profil eror dan sifat distribusi data telah dipetakan (grafik histogram tersimpan di `dashboard_output/`).
Temuan kekosongan data 20% di BMKG akan diselesaikan pada **Fase 3: Data Fusion & Pembersihan** menggunakan metode Imputasi Terarah.
