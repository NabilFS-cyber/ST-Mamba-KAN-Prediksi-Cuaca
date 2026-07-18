# Laporan Eksekusi: Fase 2 (Dataset Analysis & Quality Check)

Fase 2 berfokus pada inspeksi kualitas data secara menyeluruh dan pembersihan (*data wrangling*) untuk kedua sumber dataset (ERA5-Land dan BMKG) sebelum dilakukan penggabungan (Fusi) di fase berikutnya.

---

## 🛰️ 1. Analisis 125 File Satelit (ERA5-Land)
Proses pemindaian (*scanning*) dilakukan terhadap **125 file NetCDF bulanan** (120 bulan dari 2016-2025, ditambah 5 bulan dari awal 2026).

### 📌 Temuan:
- **Status Pembacaan**: 125 file sukses dibaca tanpa *error* (*corrupt*) menggunakan *engine* `h5netcdf`.
- **Ekstraksi Statistik**: Model berhasil menghitung nilai *Mean, Median, Min, Max, Standar Deviasi,* dan *Missing Value Percentage* untuk ke-11 variabel cuaca.
- **Hasil Kunci**: Seluruh statistik ini telah dikunci (disimpan) ke dalam Google Drive sebagai file `era5_land_125files_stats.csv`.

---

## 🏢 2. Penjinakan Anomali Data Lapangan (5 Stasiun BMKG)
Data operasional harian yang berasal dari BMKG seringkali memiliki anomali bawaan alat atau kesalahan manusia (*human error*) saat perekaman. Oleh karena itu, diterapkan dua "Langkah Penyelamatan" utama pada kelima file stasiun:

### 🛠️ Langkah Pembersihan yang Diterapkan:
1. **Koreksi Desimal (Koma ke Titik)**: Data Excel BMKG sering menggunakan format Indonesia (menggunakan koma `,` sebagai desimal). Hal ini diubah secara paksa menjadi titik `.` agar terbaca sebagai tipe data numerik (*float*) murni oleh mesin AI.
2. **Netralisasi Flag Error 8888**: Alat BMKG terkadang mengeluarkan angka `8888` jika terjadi *error* pada sensor pengukur cuaca. Angka ekstrem ini dinetralisir dan diubah menjadi `NaN` (Kosong) agar tidak merusak perhitungan statistik.

### 📊 Laporan Missing Values (Cacat Data) per Stasiun
Pemindaian pada kolom numerik utama (`TX, RH_AVG, RR, SS, FF_X`) menunjukkan variasi data kosong (*missing values*), khususnya pada kolom **RR (Curah Hujan)** yang menjadi target utama:

- **Klimatologi Jawa Barat**: RR missing **4.96%**
- **Meteorologi Citeko**: RR missing **8.13%**
- **Meteorologi Kemayoran**: RR missing **6.47%**
- **Meteorologi Maritim Tj. Priok**: RR missing **12.28%** (Paling tinggi)
- **Meteorologi Soekarno Hatta**: RR missing **10.11%**

Statistik bersih ini telah diekspor dan diamankan dalam file `bmkg_clean_stats.csv`.

---

## ✅ Kesimpulan Fase 2
Pembersihan telah sukses 100%. Data satelit terbukti utuh tanpa kerusakan, dan data BMKG telah dibebaskan dari angka *error* (`8888`) serta format desimal yang salah. Tantangan utama yang ditemukan adalah adanya data yang bolong (*missing values*) pada kolom Curah Hujan BMKG (berkisar antara 4% hingga 12%). Kekosongan ini akan ditangani pada tahap selanjutnya.
