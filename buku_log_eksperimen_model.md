# 🧠 BUKU LOG EKSPERIMEN: PEMODELAN AI & PELATIHAN (MODEL TRAINING)

Dokumen ini adalah jurnal ilmiah yang mendokumentasikan setiap iterasi, uji coba arsitektur, dan eksperimen hiperparameter (*hyperparameter tuning*) dalam merancang Otak Artificial Intelligence. Log ini menyoroti evolusi model dari sekadar *baseline* hingga mencapai arsitektur *God-Tier* (ST-Mamba-KAN).

---

## 🧪 DAFTAR EKSPERIMEN MODEL AI

| No. Eksperimen | Tanggal | Arsitektur & Hipotesis | Konfigurasi / Hiperparameter (Loss, LR, Epoch) | Hasil (Akurasi, RMSE, Recall) | Kesimpulan / Evolusi Selanjutnya |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MDL-001** | *[Tanggal]* | *Contoh: Uji coba Baseline Klasik CNN-LSTM.* | *Loss: CrossEntropy, LR: 0.001, Epoch: 150* | *Acc: 82.89%, RMSE: 19.17mm, Recall Siaga: 79%* | *Gradient vanishing parah. Perlu coba Mamba.* |
| **MDL-002** | *[Tanggal]* | *Contoh: Transisi ke ST-Mamba-MLP.* | *Loss: Huber, Mamba Layers: 2* | *Acc: 86.28%, RMSE: 18.43mm, Recall Siaga: 87%* | *Akurasi naik, tapi MLP kurang peka spasial.* |
| **MDL-003** | *[Tanggal]* | *Contoh: Injeksi GNN & Elite Losses (PINN+EVT).* | *GAT + Mamba + Elite Losses* | *Acc: 88.02%, RMSE: 17.07mm, Recall Siaga: 91%* | *Hasil optimal! Ini akan jadi The Ultimate Model.* |
| **MDL-004** | ... | ... | ... | ... | ... |
| **MDL-005** | ... | ... | ... | ... | ... |

*(Silakan tambahkan baris tabel sesuai dengan puluhan percobaan arsitektur dan hiperparameter yang telah Anda bangun).*

---

## 📌 CATATAN PENTING & DISKUSI PENEMUAN
*Bagian ini dapat digunakan untuk membahas fenomena pelatihan seperti Overfitting, meledaknya loss (*Gradient Explosion*), efek dari penggunaan TTA (Test-Time Augmentation), hingga alasan mengapa fungsi kerugian (Loss Function) Ordinal Focal sangat memengaruhi Recall badai.*
