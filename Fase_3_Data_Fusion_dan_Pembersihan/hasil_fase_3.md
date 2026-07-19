# 🔗 HASIL FASE 3: DATA FUSION & PEMBERSIHAN (PRE-PROCESSING)

Fase 3 bertugas mengeksekusi operasi peleburan (Fusi) yang menyatukan data satelit ruang angkasa (ERA5-Land) dengan data lapangan nyata (BMKG) menjadi satu matriks berkesinambungan. Fase ini juga bertanggung jawab membersihkan segala bentuk cacat (imputasi *missing values*) yang ditemukan pada Fase 2.

---

## 📥 1. Material Input
- **Dataset Iklim Satelit:** `ERA5_Jabodetabek_2016_2026.nc` (18 Variabel)
- **Dataset Takaran Lapangan:** 5 File CSV curah hujan stasiun BMKG.

---

## ⚙️ 2. Proses Rekayasa Data (Data Engineering)

### A. Ekstraksi Spasial dengan *Nearest-Neighbor*
Karena satelit ERA5 memetakan dunia dalam bentuk *grid* kotak-kotak piksel 0.1 derajat, kita harus "menembak" jatuh tepat di atas letak geografis ke-5 stasiun BMKG. 
Kode ini menggunakan fungsi `.sel(method='nearest')` dari pustaka `xarray` untuk mencuplik titik koordinat satelit yang paling dekat dengan titik koordinat *latitude/longitude* stasiun BMKG (seperti stasiun Citeko di Puncak).

### B. Agregasi Temporal 24 Jam
Satelit memotret bumi 4 kali sehari (per 6 jam). Namun BMKG merekap hujan harian satu kali per 24 jam. Agar bisa disatukan, data satelit harus diringkas (diagregasi) menjadi level Harian:
1. **Variabel Akumulatif (Dijumlahkan `sum`):** Fitur seperti presipitasi total, curah hujan konvektif, laju evaporasi, dan fluks radiasi akan ditambahkan sepanjang hari untuk mendapatkan total harian.
2. **Variabel Instan (Dirata-rata `mean`):** Fitur seperti suhu, tekanan udara, kecepatan angin, kelembapan tanah, dan tutupan awan akan dirata-ratakan selama 24 jam.

### C. Fusi (Merge Data)
Setelah ERA5 dan BMKG sama-sama memiliki format per hari (Tanggal), tabel mereka digabung (*Inner Join*) berdasarkan kolom `date`. Hasilnya, setiap baris tanggal kini memiliki data suhu, angin, dsb dari satelit **DAN** data jumlah takaran hujan asli dari kaleng BMKG.

### D. Imputasi Celah Kosong (*Missing Value Healing*)
Menindaklanjuti temuan Fase 2 bahwa BMKG bolong hingga 20%, metode imputasi bertingkat diaplikasikan secara berurutan:
1. **Forward Fill (`ffill`):** Jika hari Selasa kosong, salin data hari Senin. (Karena cuaca hari ini sangat berkaitan dengan cuaca kemarin).
2. **Backward Fill (`bfill`):** Jika `ffill` gagal, mundur salin dari hari besoknya.
3. **Median Fill:** Jika hujan hilang berhari-hari berturut-turut, diisi dengan nilai tengah historis bulan tersebut.
**Hasil:** Angka 0% *Missing Value* mutlak berhasil dicapai!

### E. Ekstraksi Fitur Tambahan (Feature Engineering)
Data ERA5 hanya memberikan suhu (*temperature*) dan titik embun (*dewpoint*). Padahal badai sangat sensitif terhadap kelembapan nisbi (RH - *Relative Humidity*). Kode ini menciptakan fitur kelembapan udara menggunakan rumus aproksimasi **Hukum Magnus**. Selain itu, fitur kecepatan angin gabungan (*Wind Speed*) diturunkan menggunakan hukum Pythagoras dari vektor komponen angin u (Timur/Barat) dan v (Utara/Selatan).

---

## 📦 3. Output Dataset Emas

Tabel cuaca ini kini bersih tanpa cela, kaya akan informasi, dan telah tersambung langsung dengan kebenaran lapangan (*Ground-Truth*).

- **Nama File Akhir:** `cleaned_merged_all_stations.pkl`
- **Format:** Pickle (`.pkl`) - Format biner serialisasi Python super cepat.
- **Isi Data:** Belasan ribu baris (Kombinasi 5 stasiun x 10+ tahun data harian)
- **Dimensi Fitur:** 20 Fitur Cuaca + 1 Label Target Curah Hujan BMKG.
- **Lokasi Penyimpanan:** `/content/drive/MyDrive/Riset_ERA5_Land/clean/`

File raksasa ini selanjutnya akan dirobek dan dipisah menjadi kompartemen *Training/Testing* pada **Fase 4 (Pembuatan Dual Brankas)**.
