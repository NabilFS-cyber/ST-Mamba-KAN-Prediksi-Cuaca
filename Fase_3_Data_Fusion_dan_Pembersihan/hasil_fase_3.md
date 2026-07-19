# 🧹 HASIL FASE 3: CLEANING, OUTLIER DETECTION & HYBRID FUSION

Fase 3 adalah "Ruang Operasi" yang mengeksekusi rekomendasi perbaikan dari **Fase 2**. Ini adalah tahap kritis di mana dua wujud data yang sangat berbeda alam (Satelit Global & Stasiun Bumi) direkatkan menjadi satu struktur organik, sekaligus disterilkan dari anomali.

---

## 📥 INPUT DARI FASE 2
1. Data Mentah Satelit ERA5-Land (NetCDF)
2. Data Observasi Stasiun BMKG (Excel/CSV)
3. Rekomendasi Audit (Perlu agregasi waktu, penyamaan koordinat geografis, dan pembersihan NaN).

---

## 🧬 LOGIKA KODE & EKSEKUSI DATA ENGINEERING

Fase ini melakukan serangkaian manuver data tingkat mahir:

### 1. Hybrid Spatial Fusion (Algoritma Penyatuan Dua Alam)
Untuk menyelesaikan masalah *Mismatch Geografis* dari Fase 2, skrip menggunakan perhitungan **Jarak Haversine** (Matematika Bola Bumi) yang dipercepat dengan algoritma pencarian spasial KD-Tree. 
Skrip akan mencocokkan titik latitude-longitude dari 5 stasiun BMKG dengan titik piksel terdekat dari satelit ERA5-Land. Akibatnya, seluruh fitur awan, suhu, dan angin milik satelit bisa dijahit sejajar (*merge*) dengan laporan curah hujan BMKG.

### 2. Time-Series Aggregation & Pembuangan Sampah (Outliers)
- **Agregasi Waktu:** Rekaman cuaca satelit (per 6 jam) diakumulasi (dijumlahkan untuk curah hujan, dan dirata-ratakan untuk suhu) ke dalam format level Harian (Per 24 Jam) agar setara dengan rekaman BMKG.
- **Konversi Fisika:** Suhu bawaan satelit (Kelvin) diubah ke derajat Celcius. Radiasi matahari yang berwujud akumulatif diubah ke bentuk *flux* (Watt per meter persegi).
- **Outlier Elimination:** Membersihkan nilai-nilai irasional (*garbage values*) akibat *error* sensor, seperti curah hujan yang bernilai negatif (< 0 mm).

### 3. Eksekusi NaN (Interpolasi Cerdas)
Untuk menangani lubang data (NaN) pada tabel BMKG yang ditemukan pada Fase 2:
- Skrip menggunakan teknik *Forward-Fill (ffill)* dan *Backward-Fill (bfill)* per stasiun. Artinya, jika hari Selasa mesin rusak, AI akan menggunakan cuaca hari Senin dan Rabu untuk menambal kekosongan secara rasional tanpa merusak integritas hukum fisika.
- Pada fitur curah hujan (RR), sisa NaN yang terlalu bolong diganti dengan nilai 0 (hari tidak hujan).

---

## ➡️ OUTPUT UNTUK FASE SELANJUTNYA
Hasil operasi pembersihan raksasa ini adalah sebuah *Single Source of Truth*:
- **File Tunggal:** `dataset_satelit_stasiun_bersih.parquet` (Format 2 Dimensi Tabel Raksasa).

Data ini sudah 100% steril, sinkron secara waktu harian, dan sinkron secara lokasi spasial (setiap stasiun punya rekam cuaca satelitnya masing-masing). File `.parquet` ini kemudian dilempar ke **Fase 4 (Pembuatan Dual Brankas)** untuk dipecah ke dalam proporsi waktu penelitian (Brankas Pra-Pelatihan & Brankas Kalibrasi).
