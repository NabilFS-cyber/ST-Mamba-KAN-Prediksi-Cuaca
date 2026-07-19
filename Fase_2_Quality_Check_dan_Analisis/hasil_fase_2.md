# 🔍 HASIL FASE 2: DATASET ANALYSIS & QUALITY CHECK

Fase 2 berfungsi sebagai "Ruang Audit Ekstrem". Menerima ratusan file mentah dari **Fase 1**, tujuan dari skrip pada fase ini adalah memindai seluruh data secara algoritmik untuk menemukan cacat, kekosongan (NaN), dan inkompatibilitas struktural sebelum masuk ke dapur fusi data.

---

## 📥 INPUT DARI FASE 1
1. **125+ File NetCDF ERA5-Land** yang berisi grid koordinat piksel (Latitude/Longitude) wilayah Jabodetabek selama lebih dari 2 dekade.
2. **File Laporan Harian BMKG (.xlsx / .csv)** dari 5 stasiun bumi.

---

## 🛠️ LOGIKA KODE & DIAGNOSA AUDIT

Kode Python di Fase 2 menggunakan pustaka `xarray` untuk membongkar format grid satelit dan `pandas` untuk membongkar tabel stasiun bumi. Hasil analisis menyingkap 3 permasalahan (anomali) fatal yang harus diselesaikan di Fase 3:

### 1. Ledakan Ukuran Data Satelit (Memory Crash Risk)
Data satelit memiliki dimensi waktu (per jam) dan grid koordinat spasial (resolusi 9 km). Ketika seluruh file disatukan, ia berpotensi merusak (*Crash/Out of Memory*) RAM Google Colab. 
**Kesimpulan Audit:** Satelit merekam 4 kali sehari (per 6 jam), sementara BMKG merekap curah hujan harian. Data satelit **harus diringkas (diagregasi) menjadi level harian** agar sejajar dengan stasiun bumi.

### 2. Mismatch Geografis (Grid vs Titik Stasiun)
Satelit memotret bumi dalam bentuk "kotak-kotak piksel", sementara stasiun BMKG adalah titik koordinat tunggal (latitude, longitude spesifik). Keduanya tidak bisa sekadar digabung menggunakan fungsi tabel biasa (`merge`).
**Kesimpulan Audit:** Dibutuhkan perhitungan rumus Matematika Geodesi (seperti Jarak *Haversine* atau *KDTree*) untuk mencari "titik piksel satelit yang posisinya paling dekat dengan atap stasiun BMKG bersangkutan".

### 3. Wabah NaN pada Data Ground-Truth (BMKG)
Analisis menyingkap bahwa data observasi stasiun bumi tidak sempurna. Ditemukan ribuan sel kosong (Missing Values / NaN) pada pencatatan BMKG, terutama pada indikator Kelembapan dan Curah Hujan di tahun-tahun awal (awal 2000-an). Ini adalah hal wajar karena sensor bumi sering mengalami mati lampu atau rusak (*maintenance*).

---

## ➡️ OUTPUT UNTUK FASE SELANJUTNYA
Berdasarkan "Laporan Penyakit Data" yang dihasilkan dari Fase 2 ini, kita menyusun strategi pembersihan dan Fusi Data hibrida yang kuat secara komputasional.
Seluruh logika Matematika untuk menangani *Mismatch Geografis*, Agregasi Waktu Harian, dan Penghancuran NaN dieksekusi sepenuhnya di **Fase 3 (Data Fusion & Cleaning Hybrid)**.
