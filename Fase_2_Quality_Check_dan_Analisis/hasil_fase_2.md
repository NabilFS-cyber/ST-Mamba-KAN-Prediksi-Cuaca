# 🔬 HASIL FASE 2: QUALITY CHECK & ANALISIS EKSPLORASI DATA

Fase 2 bertujuan untuk mengeksekusi operasi pemindaian data (*Scanning*) dan pembersihan cacat bawaan dari dua sumber data yang kita miliki (*Satelit ERA5* dan *Sensor Tanah BMKG*). Pada fase ini, kita memetakan profil *Missing Values* dan menjinakkan anomali agar data siap disatukan.

---

## 📥 1. Dataset yang Dianalisis
- **Data Satelit (ERA5):** Kumpulan 125 file bulanan `dataset_era5_land_jabodetabek_YYYY_MM.nc` (dari 2016 hingga Mei 2026).
- **Data Darat (BMKG):** 5 file `.xlsx` stasiun observasi resmi BMKG di Jabodetabek (Juni 2024 - Mei 2026).

### 📍 Koordinat 5 Stasiun BMKG:
1. **Stasiun Klimatologi Jawa Barat**
2. **Stasiun Meteorologi Citeko**
3. **Stasiun Meteorologi Kemayoran**
4. **Stasiun Meteorologi Maritim Tanjung Priok**
5. **Stasiun Meteorologi Soekarno Hatta**

---

## 🔍 2. Temuan Kualitas Data (*Quality Check*) & Statistik

### A. ERA5-Land (Data Grid Satelit)
Proses ekstraksi 11 Variabel Hidrometeorologi Daratan Utama (`u10, v10, d2m, t2m, sp, tp, ssrd, skt, swvl1, swvl2, evabs`) terhadap 125 file satelit bulanan berhasil dieksekusi dengan *h5netcdf bypass*.
- **Tingkat Missing Values:** **0% (Nol)**.
- **Karakteristik:** Model numerik satelit selalu menghasilkan matriks angka yang berkesinambungan penuh tanpa bolong. Data berhasil diekstrak dan disimpan secara kumulatif dalam `era5_land_125files_stats.csv`.

### B. BMKG (Data Sensor Observasi Tanah)
Pemindaian dilakukan pada variabel utama: `TX, RH_AVG, RR, SS, FF_X`. Terdeteksi adanya dua masalah fatal khas data observasi darat Indonesia yang langsung ditangani:
1. **Koreksi Pemisah Desimal:** Simbol koma (`,`) khas format Indonesia diubah paksa menjadi titik (`.`) agar diakui sebagai nilai numerik *float*.
2. **Netralisir Flag Error 8888:** Kode alam/sensor rusak `8888` milik BMKG berhasil diubah menjadi nilai `NaN` (*Not a Number*) agar tidak merusak kalkulasi matematis *Deep Learning*.

**Persentase Cacat Data (Missing Values) Fitur Curah Hujan Harian (RR):**
- Stasiun Klimatologi Jawa Barat: **4.96%**
- Stasiun Meteorologi Citeko: **8.13%**
- Stasiun Meteorologi Kemayoran: **6.47%**
- Stasiun Meteorologi Maritim Tanjung Priok: **12.28%**
- Stasiun Meteorologi Soekarno Hatta: **10.11%**

Statistik bersih ini telah diamankan dalam file `bmkg_clean_stats.csv`.

---

## 📦 3. Kesimpulan Fase 2
Semua profil eror dan sifat distribusi data telah dipetakan, dan cacat string koma serta angka ajaib `8888` telah ternetralisir. Kekosongan data berkisar antara 4% hingga 12% pada fitur utama BMKG (*Curah Hujan / RR*). Kekosongan ini akan diselesaikan secara mutlak menggunakan teknik **Imputasi Cerdas (FFill/BFill/Median)** pada tahapan selanjutnya: **Fase 3: Data Fusion & Pembersihan**.
