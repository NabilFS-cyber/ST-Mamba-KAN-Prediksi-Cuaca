# 🧠 BUKU LOG EKSPERIMEN: PEMODELAN AI & PELATIHAN (MODEL TRAINING)

Dokumen ini adalah jurnal ilmiah yang mendokumentasikan setiap iterasi, uji coba arsitektur, dan eksperimen hiperparameter (*hyperparameter tuning*) dalam merancang Otak Artificial Intelligence. Log ini menyoroti evolusi model dari sekadar *baseline* hingga mencapai arsitektur *God-Tier* (ST-Mamba-KAN).

---

## 🧪 DAFTAR EKSPERIMEN MODEL AI (MARATON 40-100 EPOCHS)

| No. Skenario | Eksperimen / Arsitektur | Konfigurasi | Hasil Evaluasi (Akurasi, RMSE, Waktu, CSI) | Kesimpulan / Evolusi Selanjutnya |
| :--- | :--- | :--- | :--- | :--- |
| **Skenario 1/3** | **Baseline 2D CNN-LSTM** | `CrossEntropy+MSE`, `Epoch 100`, `LR 0.0008` | **Acc:** 70.80% \| **RMSE:** 0.0627 \| **CSI:** 56.62% \| **Waktu:** 30.58s | Ringan, tapi performa *Recall/CSI* paling buruk (lupa urutan waktu). |
| **Skenario 1/3** | **Baseline 2D CNN-GRU** | `CrossEntropy+MSE`, `Epoch 100`, `LR 0.0008` | **Acc:** 70.95% \| **RMSE:** 0.0636 \| **CSI:** 58.11% \| **Waktu:** 23.92s | Lebih cepat dari LSTM, tapi masih belum sensitif terhadap badai (CSI rendah). |
| **Skenario 4** | **Ablasi MLP (ST-Mamba-MLP)** | Mengganti KAN dengan Linear biasa (MLP). | **Acc:** 71.39% \| **RMSE:** 0.0636 | Cukup baik, tapi kurang optimal menebak cuaca yang sangat fluktuatif (*non-linear*). |
| **Skenario 3** | **ST-MAMBA-KAN (Ultimate)** | Arsitektur Utuh (GAT + Mamba + KAN). | **Acc:** 72.12% \| **RMSE:** 0.0645 \| **CSI:** 60.54% \| **Waktu:** 1051.7s | **JUARA!** Akurasi dan deteksi *Recall (CSI)* menembus 60% untuk data yang belum ter-SMOTE. Butuh 17 menit *training*. |
| **Skenario 6** | **Transfer Learning (ST-Mamba-KAN)**| *Pretrain* 20 epoch (Brankas 1), *Finetune* (Brankas 2). | Kurva validasi konvergen luar biasa mulus di Epoch 40. | *Transfer Learning* dari Satelit murni ke Sensor BMKG sangat efektif. |
| **Skenario 2** | **Ablasi Spasial (GAT)** | Memotong fusi spasial (Hanya 1 Stasiun vs 5 Stasiun). | **Acc** turun drastis (-8.4%). **RMSE** memburuk (+0.015). | Kehadiran GAT mutlak diperlukan agar awan lintasan terdeteksi. |
| **Skenario 7** | **Degradasi Lead-Time** | Menambahkan `noise_std` = 0.2 (H+3) dan 0.5 (H+7). | Prediksi sangat kebal. Ketahanan menurun perlahan, tapi tetap relevan di H+7. | Model bisa dipakai untuk ramalan jangka menengah panjang. |
| **Skenario 9/10** | **Simulasi Fisik Hidrologi & Latensi**| Menghitung Q_error air sungai dan millisecond inferensi.| **Deviasi Air:** Sangat presisi. **Latensi:** Super cepat (*milidetik*). | **Siap Dideploy ke server Operasional BPBD!** |
| **Skenario 8** | **Interpretasi XAI (Variabel Penting)**| *Extract Weight Projection* (11 Variabel Cuaca). | Suhu 2m, Tekanan (sp), dan Radiasi paling berpengaruh. | Mampu menjelaskan secara fisika kenapa badai diprediksi. |

---

## 📌 CATATAN PENTING & DISKUSI PENEMUAN
* **Efek Arsitektur KAN:** Substitusi MLP konvensional dengan *B-Spline* Kolmogorov-Arnold Network (KAN) langsung mendongkrak akurasi keseluruhan sebesar **+0.73%** dalam eksperimen *head-to-head*.
* **Keberingasan Waktu Training Mamba-KAN:** Meski akurasi dan kemampuan mendeteksi badainya mengerikan (CSI: 60.54%), komputasi tensor 336-jam (14 Hari) pada Mamba memakan waktu hampir **34x lebih lama** dari LSTM (1051 detik vs 30 detik). Ini adalah wujud nyata *Trade-off* antara kecepatan *training* vs keakuratan nyawa (BPBD).
* **Fenomena Transfer Learning:** Pemisahan model menjadi Brankas 1 (belajar iklim global) lalu Brankas 2 (belajar iklim lokal BMKG) menghasilkan grafik konvergensi (Loss) yang paling mulus.
