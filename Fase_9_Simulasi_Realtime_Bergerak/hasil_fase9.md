# Fase 9 – Simulasi Prediksi Real-Time Bergerak (Live Monitoring)

Fase ini merancang sistem monitoring cuaca interaktif berupa simulasi grafik bergerak secara *real-time* di Google Colab. Program ini memuat seluruh bobot model terlatih dari direktori penyimpanan Google Drive, melakukan inferensi dinamis pada *test set*, lalu menampilkan pergerakan nilai prediksi versus nilai aktual historis dari hari ke hari secara berurutan.

## Komponen Utama Simulasi

1. **Memuat Model Secara LIVE:**
   - Script secara dinamis memuat arsitektur dan bobot model utama **ST-Mamba-KAN** (dengan parameter optimal hasil tuning dan Exponential Moving Average/EMA).
   - Memuat 3 arsitektur baseline pembanding: **CNN-LSTM**, **CNN-GRU**, dan **ST-Mamba-MLP** (Ablation).

2. **Grafik Bergerak dengan Jendela Geser (Sliding Plot):**
   - Menggunakan jendela geser (*sliding window*) berukuran 50 hari pengamatan secara berurutan.
   - Grafik diperbarui secara interaktif menggunakan fungsi `IPython.display.clear_output()` untuk memberikan impresi animasi *real-time stream*.

3. **Status Peringatan BPBD Dinamis:**
   - Menghitung ambang batas status bahaya di setiap langkah:
     - **SIAGA EKSTREM (Merah):** Jika prediksi atau aktual menyentuh batas >= 50 mm/hari (Hujan Siaga).
     - **WASPADA (Oranye):** Jika intensitas berada di batas 20 - 50 mm/hari.
     - **AMAN (Hijau):** Jika cuaca normal di bawah 20 mm/hari.

## Keuntungan Bagi Reviewer Non-Informatika

- **Visualisasi Intuitif:** Dibandingkan hanya melihat tabel performa (RMSE/F1-score) yang membosankan dan kaku, grafik bergerak ini memberikan simulasi nyata bagaimana AI bekerja secara operasional layaknya layar monitoring di ruang kontrol BPBD DKI Jakarta.
- **Keterbacaan yang Nyata:** Reviewer dapat melihat langsung bagaimana garis merah (**ST-Mamba-KAN**) sangat responsif dan sigap menangkap lonjakan-lonjakan hujan ekstrem (di atas garis batas siaga 50mm) dibandingkan baseline lain yang cenderung terlambat atau mendatar (under-predict).

## Cara Menjalankan

Buka notebook Google Colab, impor file `fase9_simulasi_bergerak.py` dari repositori GitHub, lalu jalankan. Grafik pergerakan realtime akan langsung muncul menghiasi sel Colab Anda secara otomatis!
