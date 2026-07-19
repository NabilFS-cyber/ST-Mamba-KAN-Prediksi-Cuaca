# 🎞️ HASIL FASE 5: 4D SPATIO-TEMPORAL WINDOWING & FLATTENED SMOTETOMEK

Fase 5 bertindak sebagai Pabrik Transformasi Geometri. Di sini, data 2D berwujud tabel datar yang datang dari **Fase 4** dibentuk, dilipat, dan diisi ulang (Oversampling) agar berubah wujud menjadi matriks 4 Dimensi murni. Format 4D inilah satu-satunya bentuk data yang bisa dimengerti oleh Otak AI canggih di fase selanjutnya.

---

## 📥 INPUT DARI FASE 4
Dua buah brankas aman yang memiliki jarak waktu terpisah:
1. `brankas1_pretrain.parquet`
2. `brankas2_finetune.parquet`

Tabel data di dalam brankas tersebut mencatat cuaca harian dan curah hujan dengan ketimpangan ekstrem di mana data badai (Siaga) luar biasa jarang terjadi dibandingkan hari cerah.

---

## 📐 LOGIKA KODE & REKAYASA STRUKTUR DATA

### 1. Eksekusi 4D Spatio-Temporal Windowing
Tabel datar dihancurkan dan dirakit ulang ke dalam susunan dimensi `[N_Samples, Window, Stations, Features]`:
- **N_Samples (Batch):** 15.300 matriks keping observasi (Brankas 1) & 3.560 matriks (Brankas 2).
- **Window (Waktu):** 14 hari cuaca mundur direkam utuh sebagai histori (*Temporal*).
- **Stations (Spasial):** 5 titik stasiun BMKG Jabodetabek (Kemayoran, Tj Priok, Halim, Citeko, Jabar).
- **Features (Karakteristik):** 17 Fitur Cuaca + 1 Indikator Stasiun *One-Hot* (Total 18 Fitur).

### 2. Penetapan Multi-Class (3 Level)
Selain melatih AI untuk meramal angka mutlak milimeter (Regresi), skrip pada fase ini mendidik AI untuk membunyikan alarm keselamatan (Klasifikasi) berdasarkan 3 level bahaya:
- **Kelas 0 (Aman):** Hujan < 20 mm
- **Kelas 1 (Waspada):** Hujan 20 mm - 50 mm
- **Kelas 2 (Siaga / Badai):** Hujan > 50 mm

### 3. Bug Fix Krusial: SMOTETomek Flatten Trick
Algoritma SMOTE secara *default* hanya bisa menangani data 2D dan buta pada kasus Regresi. Jika dipaksakan pada matriks 4D, nilai Regresi curah hujannya (*yr*) akan hancur teracak.
Oleh karenanya, dilakukan *Flatten Trick*:
1. Matriks 4D digepengkan ke 2D.
2. Label curah hujan (*yr*) **disuntikkan diam-diam** ke dalam ujung matriks.
3. SMOTE menggandakan fitur cuaca ekstrem (Kelas 2) sekaligus nilai *regresi* milimeter aslinya.
4. Matriks dirajut dan dikembalikan lagi ke wujud 4D.

Berkat trik ini, data cuaca buruk (*Siaga*) yang tadinya langka, kini memiliki jumlah setara (~5.630 sampel per kelas di Brankas 1), menghilangkan ketimpangan secara elegan.

---

## ➡️ OUTPUT UNTUK FASE SELANJUTNYA
Kepingan matriks berwujud 4D yang telah seimbang kelasnya ini dibagi menjadi formasi *Emas* **(70% Train, 15% Val, 15% Test)**, dan disimpan ke Google Drive dalam wujud tensor biner PyTorch (`.pt`).
Kepingan Tensor `.pt` ini langsung disuapkan ke dalam mulut **Fase 6 (The God-Tier Limit-Breaker Model)** dan **Fase 7 (Fair Baseline Comparison)** untuk dilakukan pertempuran komputasional antar *Artificial Intelligence*.
