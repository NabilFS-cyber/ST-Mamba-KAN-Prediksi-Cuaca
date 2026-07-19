# 🔗 HASIL FASE 3: CLEANING, OUTLIER DETECTION & HYBRID FUSION

Fase 3 bertugas mengeksekusi operasi peleburan hibrida (Fusi) yang menyatukan data satelit ruang angkasa (ERA5-Land) dengan data lapangan nyata (BMKG) menjadi satu matriks berkesinambungan. Fase ini juga bertanggung jawab atas pembersihan data lanjutan, imputasi nilai kosong, dan pendeteksian data pencilan (*outliers*).

---

## 📥 1. Material Input & Struktur Folder
- **Dataset Iklim Satelit:** 125 berkas NetCDF bulanan (`dataset_era5_land_*.nc`).
- **Dataset Takaran Lapangan:** 5 File Excel (.xlsx) stasiun BMKG.
- Semua luaran bersih akan disimpan ke direktori `clean/` di dalam Google Drive.

**Daftar Koordinat Stasiun BMKG Asli:**
- Stasiun Meteorologi Soekarno Hatta
- Stasiun Meteorologi Maritim Tanjung Priok
- Stasiun Meteorologi Kemayoran
- Stasiun Meteorologi Citeko
- Stasiun Klimatologi Jawa Barat

---

## ⚙️ 2. Proses Rekayasa Data Hibrida (Data Engineering)

### A. Penjahitan Temporal & Agregasi Satelit
125 file bulanan NetCDF dirakit secara berkesinambungan menggunakan dimensi waktu (*temporal concatenation*). File master satelit ini kemudian dijaga keberlanjutannya menggunakan metode imputasi *Forward-Fill* dan diekspor sementara sebagai `era5_clean.nc`.

### B. Ekstraksi Spasial dengan *Nearest-Neighbor*
Karena satelit ERA5 memetakan dunia dalam bentuk *grid*, kita "menembak" jatuh tepat di atas letak geografis 5 stasiun BMKG menggunakan `.sel(method="nearest")` dari pustaka `xarray`.

### C. Agregasi Temporal (Per Jam ke Harian)
Satelit merekam cuaca bumi secara per jam. Agar bisa disatukan dengan BMKG yang merekap per hari, data satelit diringkas:
1. **Variabel Akumulatif (Dijumlahkan `sum`):** Presipitasi total (`tp`), evaporasi (`evabs`), fluks radiasi (`ssrd`).
2. **Variabel Instan (Dirata-rata `mean`):** Suhu udara, suhu titik embun, dll.

### D. Penjinakan & Interpolasi BMKG
- Sinkronisasi format penanggalan (*Datetime*) dari kolom DATE/TANGGAL/TIME.
- Penanganan eror fatal 8888 dan 9999 diubah menjadi `NaN`.
- Metode imputasi interpolasi linier (*Linear Interpolation*) digunakan untuk menambal nilai-nilai kosong dengan sangat aman secara kronologis.

### E. Pendeteksian Outlier Curah Hujan (RR)
Data curah hujan ekstrem merupakan tantangan utama. Menggunakan rentang Interkuartil (IQR - *Interquartile Range*), pencilan dideteksi dan dicatat dalam laporan terpisah `bmkg_outliers.csv` untuk memastikan fenomena badai benar-benar nyata, bukan sekadar eror sensor lapangan.

### F. Perkawinan Data (Fusi Hibrida)
Data satelit dan data BMKG akhirnya dijahit sejajar (*Inner Join*) menggunakan kolom `TANGGAL_FUSI`.

---

## 📦 3. Output Master Matriks Emas

Penggabungan seluruh 5 stasiun menghasilkan matriks raksasa yang kaya akan informasi.

- **Nama File Akhir:** `dataset_hybrid_clean_master.csv`
- **Format:** Comma-Separated Values (`.csv`)
- **Dimensi Matriks:** **3625 baris** x **18 kolom**
- **Isi Kolom:** 11 Fitur Satelit + 6 Variabel BMKG + 1 Kolom Nama Lokasi Stasiun.
- **Lokasi Penyimpanan:** `/content/drive/MyDrive/Riset_ERA5_Land/clean/dataset_hybrid_clean_master.csv`

File matriks emas berformat `.csv` ini akan dipotong menjadi *Training/Testing Set* pada **Fase 4 (Pembuatan Dual Brankas)**.
