# 🔒 HASIL FASE 4: PEMBUATAN DUAL BRANKAS (TRAIN/VAL/TEST SPLIT)

Dokumen ini merangkum hasil eksekusi **Fase 4**, di mana dataset bersih yang menggabungkan fitur iklim satelit ERA5-Land dan pengamatan darat BMKG (dari Fase 3) dibagi secara strategis menjadi "Dual Brankas" untuk mempersiapkan pelatihan model.

---

## 📥 1. Input Data
- **File Input:** `cleaned_merged_all_stations.pkl` (Berasal dari Fase 3)
- **Isi:** Data gabungan 5 stasiun (Klimatologi Jabar, Citeko, Kemayoran, Tanjung Priok, Soekarno Hatta) berisi 18 kolom data.

---

## 🔧 2. Strategi Pembagian Data (Temporal Split)

Untuk mencegah kebocoran data masa depan (*temporal data leaking*), pembagian data latih (Train), validasi (Val), dan uji (Test) **TIDAK** dilakukan secara acak (*random shuffle*). Pembagian dilakukan secara berurutan berdasarkan waktu:

- **Training Set (70%):** Data awal periode waktu (Digunakan untuk melatih model)
- **Validation Set (15%):** Data pertengahan periode waktu (Digunakan untuk kalibrasi *hyperparameter* dan *Early Stopping*)
- **Test Set (15%):** Data paling akhir (Hanya dibuka sekali di akhir untuk ujian nyata)

---

## 🛡️ 3. Arsitektur "Dual Brankas" (Double Vault)

Untuk memaksimalkan kapabilitas AI, data dibagi ke dalam dua skenario pelatihan terpisah:

### A. Brankas 1 (Satelit Murni / ERA5-Only)
- **Fitur:** 17 Fitur cuaca gabungan (11 Satelit + 6 Darat).
- **Label Regresi:** Kolom `tp` (Total Precipitation) bawaan dari satelit ERA5.
- **Tujuan:** Digunakan sebagai data *Pre-training* (Fusi Awal). Satelit mungkin tidak seakurat alat BMKG, tetapi data satelit tidak memiliki jeda (*missing value*) dan mencakup area yang sangat luas. Ini membantu model belajar fisika cuaca secara umum.

### B. Brankas 2 (Ground-Truth BMKG)
- **Fitur:** 17 Fitur cuaca gabungan (Sama dengan Brankas 1).
- **Label Regresi:** Kolom `rainfall_bmkg` (Curah Hujan observasi nyata dari takaran hujan stasiun).
- **Tujuan:** Kunci jawaban absolut. Ini adalah data final yang digunakan untuk menguji validitas model (Evaluasi Regresi HANYA dilakukan pada `Test Set` Brankas 2).

---

## 🚦 4. Labeling Klasifikasi (Ambang Batas Bencana)

Selain regresi milimeter, data curah hujan juga dikonversi menjadi kategori klasifikasi bahaya sesuai SOP BMKG:
- 🟢 **Aman (Kelas 0):** Curah hujan < 20 mm/hari
- 🟡 **Waspada (Kelas 1):** Curah hujan 20 - 50 mm/hari
- 🔴 **Siaga (Kelas 2):** Curah hujan ≥ 50 mm/hari (Potensi banjir ekstrem)

---

## 📦 5. Output File

Semua data telah distandarisasi menggunakan `StandardScaler` (hanya di-*fit* pada Training Set untuk mencegah *leaking*), lalu diekspor menjadi *Pickle Files*:

Lokasi Penyimpanan: `/content/drive/MyDrive/Riset_ERA5_Land/clean/`
1. **Brankas 1:** `b1_train.pkl`, `b1_val.pkl`, `b1_test.pkl`
2. **Brankas 2:** `b2_train.pkl`, `b2_val.pkl`, `b2_test.pkl`
3. **Scaler:** `scaler_features.pkl`

Semua file ini 100% steril, siap dibentuk menjadi tensor 4 dimensi pada Fase 5.
