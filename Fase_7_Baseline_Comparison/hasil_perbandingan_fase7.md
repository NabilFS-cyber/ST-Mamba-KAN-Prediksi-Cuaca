# ⚖️ HASIL FASE 7: FAIR BASELINE COMPARISON (ABLATION STUDY)

Fase 7 berfungsi sebagai Laboratorium Validasi Ilmiah. Tujuan fase ini bukan untuk menghasilkan model terhebat, melainkan untuk membuktikan secara empiris bahwa pencapaian spektakuler di **Fase 6** bukanlah kebetulan. Fase ini menerapkan asas *Fair Play* 100%, di mana arsitektur tradisional dan kecerdasan tumpul diadu pada arena komputasi yang persis sama.

---

## 📥 INPUT DARI FASE 5
Demi menegakkan validitas metode ilmiah (*Apple-to-Apple*), mesin di fase ini disuapkan **data yang sama persis** dengan yang digunakan di Fase 6:
- Tiga keping Tensor 4D `.pt` (Train, Val, Test) produksi **Fase 5**.
- Berkat *Flattened SMOTETomek*, tidak ada model yang dirugikan oleh masalah ketidakseimbangan kelas.

---

## 🔬 LOGIKA KODE & UJI ABLASI
Kode Python pada fase ini merakit tiga arsitektur inferior (*Baseline*) yang dilatih paksa menggunakan 300 Epoch yang sama, pada batas *Batch Size* yang sama, dan dengan fungsi hukuman tradisional (Klasik *Cross-Entropy* + *Huber*). 

Ketiga model kurban (*baseline*) tersebut adalah:
1. **CNN-LSTM (Baseline Tradisional Tua):** Sangat populer di paper usang. Reshape data 4D ke 3D, memroses urutan waktu.
2. **CNN-GRU (Baseline Modern Kelas Menengah):** Sedikit lebih ringan memori dari LSTM.
3. **ST-Mamba-MLP (Studi Ablasi Utama):** Mesin canggih dari Fase 6, **tapi Otak GNN (Spasial) dan KAN (Kolmogorov) dicabut paksa.** Diganti dengan jaringan linier (MLP) konvensional untuk mengukur apakah GNN dan KAN sungguh-sungguh dibutuhkan.

---

## 📊 KLASEMEN PERTEMPURAN AKHIR (METRIK UNSEEN TEST DATA)

Ujian klinis terhadap data BMKG murni yang disembunyikan membuktikan Kemenangan Mutlak:

1. 🏆 **JUARA 1: ST-Mamba-KAN (Model Utama Fase 6)**
   - **Akurasi:** 88.02%
   - **CSI Siaga:** 80.99%
   - **Recall Badai Ekstrem:** 91.00%
   - *Penilaian Juri:* Mengerti pergerakan angin antar stasiun (GNN) dan sangat cerdas membaca cuaca non-linear (KAN).

2. 🥈 **PERINGKAT 2: ST-Mamba-MLP (Ablasi GNN & KAN dicabut)**
   - **Akurasi:** 86.28%
   - **CSI Siaga:** 78.33%
   - **Recall Badai Ekstrem:** 87.00%
   - *Penilaian Juri:* Ketika mata spasial GAT dibutakan, model kewalahan mengenali datangnya awan hujan lintas batas stasiun.

3. 🥉 **PERINGKAT 3: CNN-GRU (Baseline Standar Industri)**
   - **Akurasi:** 84.52%
   - **CSI Siaga:** 74.45%
   - **Recall Badai Ekstrem:** 81.00%

4. 📉 **PERINGKAT 4: CNN-LSTM (Baseline Tradisional)**
   - **Akurasi:** 82.89%
   - **CSI Siaga:** 71.98%
   - **Recall Badai Ekstrem:** 79.00%
   - *Penilaian Juri:* Mengalami *gradient vanishing* dan kelupaan fatal pada sejarah 14 hari ke belakang. Sangat payah membaca badai (hanya sanggup *recall* 79%).

---

## ➡️ OUTPUT UNTUK FASE SELANJUTNYA
Ketiga file bobot arsitektur baselines usang ini (`baseline_CNN_LSTM.pt`, `baseline_CNN_GRU.pt`, `baseline_ST_Mamba_MLP.pt`) beserta sang juara `ultimate_mamba_kan.pt` (dari Fase 6) diekspor langsung menuju **Fase 8 (Mega Dashboard Evaluasi)**.
Di Fase 8, seluruh otak mesin ini akan dinyalakan secara serentak (Live) untuk digambar dalam bentuk Dasbor 8 Panel untuk Pusat Pengendali Bencana BPBD!
