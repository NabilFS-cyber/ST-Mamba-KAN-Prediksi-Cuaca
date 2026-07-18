# Laporan Eksekusi: Fase 3 (Data Fusion & Pembersihan Lanjutan)

Fase 3 merupakan jantung dari arsitektur prapemrosesan model AI ini. Pada fase ini, data iklim satelit beresolusi spasial-temporal tinggi (ERA5-Land) dikawinkan (*Data Fusion*) secara presisi dengan data observasi stasiun darat (BMKG) menggunakan teknik *Hybrid Spatial Extraction*.

---

## 🛠️ Proses Teknis Utama

### 1. Perakitan Dimensi Waktu Satelit (Temporal Concatenation)
Seluruh 125 file NetCDF (data bulanan dari 2016 hingga Mei 2026) dibaca sekaligus menggunakan pustaka `xarray` (fungsi `open_mfdataset`). File-file ini dijahit secara berurutan berdasarkan waktu (*by_coords*), membentuk satu master linimasa cuaca Jabodetabek selama 10 tahun tanpa terputus.

### 2. Penanganan Kekosongan Data Satelit (Imputasi)
Jika terdapat grid spasial satelit yang terputus atau hilang (akibat anomali satelit pada jam tertentu), algoritma `ffill` (*Forward-Fill*) diaplikasikan. Data kosong diisi menggunakan data cuaca dari jam terdekat sebelumnya agar kontinuitas fisik dipertahankan. Master satelit yang bersih ini kemudian diekspor menjadi file tunggal raksasa: `era5_clean.nc`.

### 3. Ekstraksi Spasial (Spatial Slicing)
Berdasarkan lintang (latitude) dan bujur (longitude) resmi dari situs BMKG, fitur satelit diekstrak secara spesifik hanya pada titik-titik koordinat 5 stasiun berikut:
- **Stasiun Meteorologi Soekarno Hatta** (-6.12000, 106.65000)
- **Stasiun Meteorologi Maritim Tanjung Priok** (-6.10781, 106.88053)
- **Stasiun Meteorologi Kemayoran** (-6.15559, 106.84000)
- **Stasiun Meteorologi Citeko** (-6.70000, 106.85000)
- **Stasiun Klimatologi Jawa Barat** (-6.50000, 106.75000)

*Metode Ekstraksi:* `nearest` (mengambil grid satelit terdekat dengan lokasi stasiun bumi).

### 4. Agregasi Waktu & Imputasi Harian BMKG
- Data satelit yang asalnya per jam diagregasi menjadi harian (*Daily*). Fitur akumulatif seperti hujan (`tp`) dan evaporasi dijumlahkan (`sum`), sedangkan suhu dan tekanan dirata-ratakan (`mean`).
- Pada data BMKG, interpolasi linear digunakan untuk menambal kekosongan fitur suhu, kelembapan, dan angin. Khusus untuk Curah Hujan (`RR`), data yang kosong (NaN) diisi dengan nilai `0` (asumsi tidak ada hujan tercatat).

### 5. Fusi Data (Perkawinan Dataset)
Kedua *dataframe* harian ini (Satelit dan BMKG) digabungkan (*inner join*) berdasarkan kunci `TANGGAL_FUSI`. Data yang tidak beririsan waktu dibuang.

---

## 📊 Hasil Akhir (Dataset Hybrid Master)
Setiap stasiun berhasil menyumbangkan sekitar 722 hingga 726 baris data harian yang valid (Tahun 2024-2026 sesuai ketersediaan irisan BMKG-Satelit). Keseluruhan data ini kemudian disatukan secara vertikal (*concat*).

- **File Output Final:** `dataset_hybrid_clean_master.csv`
- **Dimensi Matriks Final:** **3625 Baris** x **18 Kolom**

Matriks berdimensi 3625x18 ini dijuluki sebagai **"Matriks Emas"**. Dataset inilah yang akan dimasukkan ke Fase 4 (Sliding Window) dan akhirnya dilatih ke dalam model Deep Learning Mamba & GNN di fase berikutnya.
