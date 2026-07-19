# 🌩️ ST-Mamba-KAN: Prediksi Curah Hujan Ekstrem Spasio-Temporal di Jabodetabek

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![Platform](https://img.shields.io/badge/Platform-Google%20Colab-F9AB00.svg)](https://colab.research.google.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Spatio-Temporal Mamba - Kolmogorov-Arnold Network (ST-Mamba-KAN)** untuk Sistem Peringatan Dini Bencana Hidrometeorologi berbasis *Deep Learning* di wilayah Jabodetabek, Indonesia.

---

## 📖 Abstrak

Penelitian ini mengusulkan arsitektur jaringan saraf tiruan baru bernama **ST-Mamba-KAN** yang menggabungkan tiga komponen mutakhir:

1. **Dynamic GAT (Graph Attention Network):** Memodelkan relasi spasial dinamis antar 5 stasiun cuaca BMKG menggunakan mekanisme atensi multi-kepala.
2. **Mamba (Selective State Space Model):** Menangkap dependensi temporal jangka panjang dari data deret waktu cuaca 14 hari ke belakang secara efisien.
3. **KAN (Kolmogorov-Arnold Network):** Menggantikan lapisan MLP konvensional dengan fungsi aktivasi B-Spline (Grid 12) untuk menangkap anomali curah hujan ekstrem (*heavy-tail distribution*).

Model ini mampu melakukan **dual-task prediction** secara simultan:
- **Regresi:** Memprediksi volume curah hujan (mm/hari) dengan RMSE **17.07 mm**
- **Klasifikasi:** Mendeteksi tingkat bahaya cuaca (Aman/Waspada/Siaga) dengan akurasi **88.02%** dan Recall Siaga **91%**

---

## 🏗️ Arsitektur Model

```
Input Tensor 4D [Batch, 14 Hari, 5 Stasiun, 18 Fitur]
         │
         ▼
┌─────────────────────┐
│  Linear Projection  │  18 → hidden_dim
└─────────┬───────────┘
         │
         ▼
┌─────────────────────┐
│   Dynamic GAT       │  Multi-Head Attention (4 heads)
│   (Spatial Module)  │  Learns inter-station relationships
└─────────┬───────────┘
         │
         ▼
┌─────────────────────┐
│  Spatial Aggregation │  5×hidden → hidden
└─────────┬───────────┘
         │
         ▼
┌─────────────────────┐
│   MambaBlock × N    │  Selective State Space Model
│   (Temporal Module) │  d_state=16, d_conv=4, expand=2
└─────────┬───────────┘
         │
         ▼
┌─────────────────────┐
│  Mean + Max Pooling  │
└─────────┬───────────┘
         │
         ▼
┌─────────────────────┐
│   KANLinear Head    │  B-Spline Grid 12, Order 3
│   (Decision Module) │
└─────────┬───────────┘
         │
    ┌────┴────┐
    ▼         ▼
 Regresi   Klasifikasi
 (mm/hari)  (3 Kelas)
```

---

## 📊 Hasil Performa Final

### Perbandingan Model (300 Epoch, Patience 50)

| Model | RMSE (mm) | Akurasi | Macro F1 | Recall Siaga |
| :--- | :---: | :---: | :---: | :---: |
| CNN-LSTM (Baseline) | 19.17 | 82.89% | 0.740 | 79% |
| CNN-GRU (Baseline) | 19.97 | 84.52% | 0.763 | 81% |
| ST-Mamba-MLP (Ablasi) | 18.43 | 86.28% | 0.762 | 87% |
| **ST-Mamba-KAN** (Ours) | **17.07** | **88.02%** | **0.776** | **91%** |

### Matriks Konfusi (ST-Mamba-KAN)

| Asli \ Prediksi | Aman | Waspada | Siaga |
| :--- | :---: | :---: | :---: |
| **Aman** (<20mm) | 1564 | 64 | 27 |
| **Waspada** (20-50mm) | 90 | 138 | 82 |
| **Siaga** (≥50mm) | 31 | 45 | **788** |

---

## 📁 Struktur Proyek

```
Perancangan_Model_AI/
│
├── Fase_1_Dataset_Download/          # Download ERA5-Land dari ECMWF
│   ├── download_era5_land.py
│   └── hasil_fase1.md
│
├── Fase_2_Quality_Check_dan_Analisis/ # Quality Check & EDA
│   ├── fase2_quality_check.py
│   └── hasil_fase2.md
│
├── Fase_3_Data_Fusion_dan_Pembersihan/ # Fusi ERA5 + BMKG
│   ├── fase3_data_fusion.py
│   └── hasil_fase3.md
│
├── Fase_4_Pembuatan_Dual_Brankas/    # Train/Val/Test Split
│   ├── fase4_dual_brankas.py
│   └── hasil_fase4.md
│
├── Fase_5_Data_Windowing_4D/         # Tensor 4D + SMOTE
│   ├── fase5b_windowing_4d_smote.py
│   └── hasil_fase5.md
│
├── Fase_6_Pelatihan_Model_ST_Mamba/  # Model ST-Mamba Pertama
│   ├── fase6_mamba_pelatihan.py
│   └── hasil_fase6.md
│
├── Fase_7_ST_Mamba_GNN/              # Penambahan GNN Haversine
│   ├── fase7b_elite_gnn_mamba.py
│   ├── hasil_fase7_gnn_mamba.md
│   └── hasil_fase7b_elite.md
│
├── Fase_8_Baseline_Comparison/       # Uji Fair Play Baseline
│   ├── phase8_fair_baselines.py
│   └── hasil_perbandingan_fase8.md
│
├── Fase_9_Evaluasi_Mega_Dashboard/   # Dashboard 8-Panel + BPBD
│   ├── fase9_mega_dashboard.py
│   ├── fase9_mega_dashboard_live.py
│   └── hasil_fase9_mega_dashboard.md
│
├── Fase_10_Ultimate_ST_Mamba_KAN/    # Model Final (Limit-Breaker)
│   ├── fase10_optuna_gat_mamba_kan.py
│   └── hasil_fase10_ultimate.md
│
├── MASTER_CHANGELOG_EVOLUTION.md     # Rekam Jejak Evolusi
└── README.md                         # Dokumen ini
```

---

## 🔧 Cara Penggunaan

### Prasyarat
- Google Colab dengan GPU (T4/L4/A100)
- Google Drive untuk penyimpanan data dan model
- API key Copernicus CDS (untuk Fase 1)

### Urutan Eksekusi
```bash
# Fase 1: Download data ERA5-Land (memerlukan CDS API key)
# Fase 2: Quality check & analisis eksplorasi
# Fase 3: Fusi data ERA5 + BMKG
# Fase 4: Pembagian data (train/val/test) + dual brankas
# Fase 5: Konversi ke tensor 4D + SMOTE
# Fase 6: Pelatihan ST-Mamba (baseline internal)
# Fase 7: Penambahan GNN (Haversine spatial)
# Fase 8: Pelatihan model pembanding (CNN-LSTM, CNN-GRU, Ablasi)
# Fase 9: Visualisasi dashboard evaluasi
# Fase 10: Pelatihan model final ST-Mamba-KAN (Limit-Breaker)
```

Setiap fase dijalankan sebagai notebook/script terpisah di Google Colab. Baca file `hasil_*.md` di setiap folder untuk penjelasan lengkap input, proses, dan output.

---

## 🌐 Data

| Sumber | Deskripsi | Periode | Resolusi |
| :--- | :--- | :--- | :--- |
| **ERA5-Land** (ECMWF) | Data reanalisis satelit (11 variabel cuaca) | 2016 - Mei 2026 | 0.1° x 0.1° |
| **BMKG** | Data observasi darat (5 stasiun Jabodetabek) | Juni 2024 - Mei 2026 | Per stasiun |

### 5 Stasiun Cuaca BMKG
| Stasiun | Latitude | Longitude |
| :--- | :---: | :---: |
| Tanjung Priok | -6.1065 | 106.8810 |
| Kemayoran | -6.1554 | 106.8428 |
| Halim Perdanakusuma | -6.2665 | 106.8905 |
| Citeko (Bogor) | -6.7019 | 106.9330 |
| Pondok Betung | -6.2611 | 106.7572 |

---

## 📚 Referensi Kunci

- Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv:2312.00752
- Liu, Z., et al. (2025). *KAN: Kolmogorov-Arnold Networks*. arXiv:2404.19756
- Veličković, P., et al. (2018). *Graph Attention Networks*. ICLR 2018
- Lestari, N. D. P., & Djamal, E. C. (2022). *Rainfall Prediction Using Spatial CNN and RNN*. UNJANI
- Zou, et al. (2025). *Cloud Modeling with ST-Mamba Architecture*
- Hsu, et al. (2021). *Extreme Rainfall Anomaly Detection*

---

## 👥 Tim Peneliti

**Program Kreativitas Mahasiswa (PKM) - Riset Eksakta**

---

## 📄 Lisensi

Proyek ini dilisensikan di bawah [MIT License](LICENSE).
