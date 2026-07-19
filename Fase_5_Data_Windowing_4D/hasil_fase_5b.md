# 🎞️ HASIL FASE 5B: 4D SPATIO-TEMPORAL WINDOWING & FLATTENED SMOTETOMEK

Fase 5B adalah tahapan esensial dalam mempersiapkan data masukan (*input*) sebelum dicerna oleh arsitektur *Deep Learning* graf. Pada tahap ini, *Dataframe* 2D diubah menjadi Tensor 4 Dimensi murni agar kecerdasan buatan (GNN + Mamba) mampu memahami hukum fisika spasial dan temporal secara bersamaan.

---

## 📐 1. Arsitektur Tensor 4D (Spatio-Temporal Graph)
Data dibentuk ke dalam struktur matriks berlapis dengan format ukuran `[N_Samples, Window, Stations, Features]`.
- **N_Samples (Batch):** Jumlah kepingan observasi cuaca.
- **Window (Waktu):** 14 hari berturut-turut (Temporal).
- **Stations (Spasial):** 5 titik stasiun pengamatan (Tanjung Priok, Kemayoran, Halim/Soekarno-Hatta, Citeko, Jabar).
- **Features (Karakteristik):** 17 Fitur Cuaca + 1 Indikator *One-Hot* target stasiun (Total 18 Fitur).

**Dimensi Sampel yang Dihasilkan:**
- **Brankas 1 (Pre-Train):** 15.300 matriks 4D.
- **Brankas 2 (Fine-Tune):** 3.560 matriks 4D.

---

## ⚖️ 2. Multi-Class Target & SMOTETomek Flatten Trick
Selain meramal curah hujan persis dalam milimeter (Regresi), kita membagi intensitas bahaya menjadi 3 level kelas (Klasifikasi) yang akan diprediksi:
- **Kelas 0:** Ringan / Aman (< 20 mm)
- **Kelas 1:** Sedang / Waspada (20 mm - 50 mm)
- **Kelas 2:** Lebat & Ekstrem / Siaga (> 50 mm)

### 🐛 Bug Fix Krusial (SMOTETomek)
SMOTE (Synthetic Minority Over-sampling Technique) umumnya hanya dirancang untuk data 2D. Untuk bisa menangani graf cuaca 4D:
1. **Flatten Trick:** Tensor 4D `[14, 5, 18]` digepengkan sementara menjadi 1 Baris dengan `1260` kolom.
2. **Target Interpolation:** Nilai regresi curah hujan riil (`yr`) disuntikkan ke dalam matriks yang digepengkan sebelum di-SMOTE. Ini sangat krusial! Jika target curah hujan (*regresi*) tidak ikut diinterpolasi bersilangan dengan kelas, model Regresi akan buta total dan *error* prediksinya (RMSE) akan melonjak tajam >100mm.
3. Setelah SMOTE memoles data badai (Kelas 2) menjadi setara dengan kelas Aman, matriks dirakit ulang secara matematis kembali ke wujud 4 Dimensi.

**Hasil SMOTE (Keseimbangan Baru):**
- **Brankas 1 Train:** ~5.630 sampel seimbang per kelas.
- **Brankas 2 Train:** ~2.087 sampel seimbang per kelas.

---

## 🔒 3. Splitting & Standarisasi Emas
Data dipecah ke dalam proporsi Emas: **70% (Train) - 15% (Validation) - 15% (Test)**.
Penskalaan menggunakan *StandardScaler* hanya diterapkan ketat pada 17 fitur cuaca, dan dengan sengaja mem- *bypass* / tidak menyentuh kolom logika indikator One-Hot stasiun.

Seluruh output Tensor ini lalu disimpan dalam format PyTorch `.pt` langsung ke Google Drive, siap disuapkan ke dalam *"Mulut"* raksasa **ST-Mamba-GNN (Fase 6)**.
