# 🌍 HASIL FASE 1: DATASET DOWNLOAD & EXTRACTION (ERA5-Land & BMKG)

Fase 1 adalah langkah awal dan fondasi dari seluruh *Pipeline* Prediksi Cuaca berbasis AI ini. Tujuan utamanya adalah mengumpulkan bahan baku mentah (data historis iklim) dalam jumlah masif dari dua sumber berbeda: Satelit Global dan Stasiun Bumi.

---

## 📡 1. Pengumpulan Data Satelit (ERA5-Land)
**Sistem Input:** API Copernicus Climate Data Store (CDS).
**Logika Kode:**
Skrip Python secara otomatis mengunduh data satelit ERA5-Land (resolusi tinggi 9 km) untuk wilayah batas geografis *Jabodetabek*. Data yang diunduh mencakup periode waktu dari **2016 hingga Mei 2026**.

**11 Variabel Cuaca Satelit yang Diunduh:**
1. *10m U-component of wind* (Angin Timur-Barat)
2. *10m V-component of wind* (Angin Utara-Selatan)
3. *2m Dewpoint Temperature* (Suhu Titik Embun)
4. *2m Temperature* (Suhu Udara Permukaan)
5. *Surface Pressure* (Tekanan Udara Permukaan)
6. *Total Precipitation* (Curah Hujan Harian)
7. *Surface Net Solar Radiation* (Radiasi Matahari Neto Permukaan)
8. *Skin Temperature* (Suhu Kulit Bumi)
9. *Volumetric Soil Water (Layer 1)* (Kelembapan Tanah Permukaan)
10. *Volumetric Soil Water (Layer 2)* (Kelembapan Tanah Lapis Kedua)
11. *Evaporation from Bare Soil* (Penguapan Tanah)

---

## 🏛️ 2. Pengumpulan Data Stasiun Bumi (BMKG)
Data kebenaran aktual (*Ground-Truth*) diperoleh secara resmi dari stasiun pengamatan BMKG di wilayah Jabodetabek. Data ini mencakup periode pengamatan dari **5 Juni 2024 hingga 31 Mei 2026**.

**6 Variabel Cuaca Stasiun Bumi (BMKG):**
1. **TX** (Suhu Maksimum)
2. **RH_AVG** (Kelembapan Rata-rata)
3. **RR** (Curah Hujan)
4. **SS** (Lama Penyinaran Matahari)
5. **FF_X** (Kecepatan Angin Maksimum)
6. **DDD_X** (Arah Angin Terbanyak)

**Stasiun yang Digunakan:**
1. Stasiun Meteorologi Kemayoran (Pusat Jakarta)
2. Stasiun Meteorologi Maritim Tanjung Priok (Pesisir Utara)
3. Stasiun Meteorologi Soekarno Hatta (Barat)
4. Stasiun Meteorologi Citeko (Dataran Tinggi Selatan)
5. Stasiun Klimatologi Jawa Barat (Area Bogor/Depok)

---

## ➡️ OUTPUT UNTUK FASE SELANJUTNYA
Bahan baku yang terkumpul di Fase 1 ini menghasilkan dua format data yang benar-benar mentah dan belum dapat disatukan:
1. **File NetCDF (`.nc`):** Data satelit ERA5-Land berbentuk grid multidimensi dengan perekaman *per jam*.
2. **File Excel/CSV BMKG (`.xlsx` / `.csv`):** Laporan data stasiun bumi berbentuk tabel dengan format waktu lokal dan kemungkinan terdapat nilai kosong (NaN).

Data mentah masif ini **diserahkan ke Fase 2 (Dataset Analysis & Quality Check)** untuk diaudit, dicari kecacatannya, dan diidentifikasi masalah resolusi spatio-temporalnya sebelum diizinkan masuk ke proses pembersihan.
