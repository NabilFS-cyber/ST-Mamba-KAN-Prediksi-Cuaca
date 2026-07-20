# =====================================================================
# PHASE 4 – DUAL BRANKAS CREATION (PRE-TRAIN & FINE-TUNE)
# TARGET SAKRAL: HUJAN SANGAT LEBAT / EKSTREM (>= 50 MM)
# FINAL EDITION: RAM-SAFE + BILINEAR + WIB + SINKRONISASI + TITANIUM SHIELD
# =====================================================================
from google.colab import drive
import os, glob, warnings
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# 1️⃣ Mount Google Drive & Inisialisasi Path
drive.mount('/content/drive')
drive_folder = '/content/drive/MyDrive/Riset_ERA5_Land/'
OUTPUT_ROOT  = os.path.join(drive_folder, 'clean/')

# Koordinat Resmi Asli BMKG (Sesuai Situs Resmi)
stasiun_config = {
    "Stasiun Meteorologi Soekarno Hatta": {"lat": -6.12000, "lon": 106.65000},
    "Stasiun Meteorologi Maritim Tanjung Priok": {"lat": -6.10781, "lon": 106.88053},
    "Stasiun Meteorologi Kemayoran": {"lat": -6.15559, "lon": 106.84000},
    "Stasiun Meteorologi Citeko": {"lat": -6.70000, "lon": 106.85000},
    "Stasiun Klimatologi Jawa Barat": {"lat": -6.50000, "lon": 106.75000}
}

# 2️⃣ Load Master NetCDF Satelit
era5_path = os.path.join(OUTPUT_ROOT, 'era5_clean.nc')
if not os.path.isfile(era5_path):
    raise FileNotFoundError(f'File ERA5 bersih tidak ditemukan: {era5_path}')

ds_era5 = xr.open_dataset(era5_path, engine='h5netcdf')
print('Dimensi ERA5 Master:', ds_era5.sizes)

# =====================================================================
# 🔧 ARSITEKTUR RAM-SAFE: Ekstraksi Bilinear DULU per stasiun,
#    baru geser WIB dan resample harian di Pandas
# =====================================================================
station_era5_frames = []
print("\n📍 Mengekstrak nilai spasial satelit (Bilinear Interpolation) untuk 5 stasiun...")

var_akumulasi = [v for v in ds_era5.data_vars if v in ['tp', 'evabs', 'ssrd', 'ssr', 'cp']]
aturan_agregasi = {v: 'sum' if v in var_akumulasi else 'mean' for v in ds_era5.data_vars}

for name, info in stasiun_config.items():
    print(f"   → {name}")
    # 1. Tarik titik spasial dengan Bilinear Interpolation
    ds_interp = ds_era5.interp(latitude=info['lat'], longitude=info['lon'], method='linear')
    df_station = ds_interp.to_dataframe().reset_index()

    # 2. Geser Zona Waktu ke WIB (GMT+7) — SOP Resmi BMKG
    kolom_waktu = 'valid_time' if 'valid_time' in df_station.columns else 'time'
    df_station[kolom_waktu] = df_station[kolom_waktu] + pd.Timedelta(hours=7)
    df_station['DATE'] = pd.to_datetime(df_station[kolom_waktu]).dt.normalize()

    # 3. Resample Harian di Pandas (RAM-Safe)
    df_harian = df_station.groupby('DATE').agg(aturan_agregasi).reset_index()

    df_harian['STATION'] = name
    df_harian['LATITUDE'] = info['lat']
    df_harian['LONGITUDE'] = info['lon']
    station_era5_frames.append(df_harian)

era5_per_station = pd.concat(station_era5_frames, ignore_index=True)
print(f" ✅ Sukses! Total baris satelit harian WIB: {len(era5_per_station)}")

# =====================================================================
# 🔥 STRATEGI PEMBENTUKAN DUAL BRANKAS (TARGET EXTREME >= 50 mm)
# =====================================================================
CUTOFF_DATE = pd.to_datetime('2024-05-31')

print("\n📦 MEMBANGUN BRANKAS 1: PRE-TRAINING (Satelit Murni 2016 - Mei 2024)")
brankas1 = era5_per_station[era5_per_station['DATE'] <= CUTOFF_DATE].copy()
brankas1['RR'] = brankas1['tp'] * 1000.0

# ⚠️ ATURAN SAKRAL PROPOSAL: Hujan Ekstrem Siaga jika >= 50 mm/hari
brankas1['EXTREME'] = (brankas1['RR'] >= 50).astype(int)
brankas1 = brankas1.dropna(subset=['RR'])

# 🧠 Suntik kolom BMKG kosong (NaN) agar matriks fitur simetris
for col_bmkg in ['TX', 'RH_AVG', 'SS', 'FF_X']:
    brankas1[col_bmkg] = np.nan

print(f"   -> Sukses Mengunci Brankas 1: {brankas1.shape[0]} baris cuaca sejarah.")
print(f"   -> Rentang RR: {brankas1['RR'].min():.2f} - {brankas1['RR'].max():.2f} mm")

print("\n📦 MEMBANGUN BRANKAS 2: FINE-TUNING (FUSI DATA BMKG Juni 2024 - Mei 2026)")
csv_hybrid_path = os.path.join(OUTPUT_ROOT, "dataset_hybrid_clean_master.csv")
if not os.path.isfile(csv_hybrid_path):
    raise FileNotFoundError(f'File Master Fusi Phase 2 tidak ditemukan: {csv_hybrid_path}')

brankas2 = pd.read_csv(csv_hybrid_path)
brankas2['DATE'] = pd.to_datetime(brankas2['TANGGAL_FUSI']).dt.normalize()

# ⚠️ ATURAN SAKRAL PROPOSAL: Hujan Ekstrem Siaga jika >= 50 mm/hari
brankas2['EXTREME'] = (brankas2['RR'] >= 50).astype(int)

brankas2.rename(columns={'Nama_Stasiun': 'STATION'}, inplace=True)
for name, info in stasiun_config.items():
    mask = brankas2['STATION'] == name
    brankas2.loc[mask, 'LATITUDE'] = info["lat"]
    brankas2.loc[mask, 'LONGITUDE'] = info["lon"]

# Rapikan susunan kolom agar identik
kolom_fitur_final = [c for c in brankas1.columns if c != 'TANGGAL_FUSI']
brankas2 = brankas2[[c for c in kolom_fitur_final if c in brankas2.columns]]

print(f"   -> Sukses Mengunci Brankas 2: {brankas2.shape[0]} baris data fusi ground-truth.")
print(f"   -> Rentang RR: {brankas2['RR'].min():.2f} - {brankas2['RR'].max():.2f} mm")

# ---------------------------------------------------------------------
# 💾 PENYIMPANAN BERKAS PARQUET EMAS
# ---------------------------------------------------------------------
path_b1 = os.path.join(OUTPUT_ROOT, 'brankas1_pretrain.parquet')
path_b2 = os.path.join(OUTPUT_ROOT, 'brankas2_finetune.parquet')

# 🛡️ TITANIUM SHIELD: Paksa urutan kolom identik (sorted intersection)
kolom_seragam = sorted(list(set(brankas1.columns).intersection(set(brankas2.columns))))
brankas1 = brankas1[kolom_seragam]
brankas2 = brankas2[kolom_seragam]

brankas1.to_parquet(path_b1, index=False)
brankas2.to_parquet(path_b2, index=False)

# ---------------------------------------------------------------------
# 🚨 LAPORAN KETIMPANGAN DATA (IMBALANCE RATIO) & VALIDASI AKHIR
# ---------------------------------------------------------------------
print("\n" + "="*60)
print("🚨 LAPORAN KETIMPANGAN DATA (AMBANG BATAS >= 50mm) 🚨")
print("-" * 55)
print("BRANKAS 1 (Pre-Train: Jan 2016 - Mei 2024)")
b1_counts = brankas1['EXTREME'].value_counts()
b1_ratio = b1_counts.get(0, 0) / max(b1_counts.get(1, 1), 1)
print(f" -> Hari Aman (<50mm)   : {b1_counts.get(0, 0)} hari")
print(f" -> Hari Badai (>=50mm) : {b1_counts.get(1, 0)} hari")
print(f" -> Rasio Ketimpangan    : 1 : {b1_ratio:.0f}")

b2_counts = brankas2['EXTREME'].value_counts()
b2_ratio = b2_counts.get(0, 0) / max(b2_counts.get(1, 1), 1)
print("\nBRANKAS 2 (Fine-Tune: Jun 2024 - Mei 2026)")
b2_counts = brankas2['EXTREME'].value_counts()
print(f" -> Hari Aman (<50mm)   : {b2_counts.get(0, 0)} hari")
print(f" -> Hari Badai (>=50mm) : {b2_counts.get(1, 0)} hari")
print(f" -> Rasio Ketimpangan    : 1 : {b2_ratio:.0f}")

# Validasi Titanium Shield
print("\n🛡️ VALIDASI TITANIUM SHIELD:")
print(f" -> Kolom Brankas 1: {list(brankas1.columns)}")
print(f" -> Kolom Brankas 2: {list(brankas2.columns)}")
print(f" -> Kolom Identik?  : {'✅ YA, SEMPURNA!' if list(brankas1.columns) == list(brankas2.columns) else '❌ TIDAK SINKRON!'}")
print("="*60)
print(f"🎉 PHASE 4 SUKSES TOTAL!")
print(f" -> {os.path.basename(path_b1)} → {brankas1.shape[0]} baris ({brankas1.shape[1]} kolom)")
print(f" -> {os.path.basename(path_b2)} → {brankas2.shape[0]} baris ({brankas2.shape[1]} kolom)")

# ---------------------------------------------------------------------
# 📊 VISUALISASI SEBARAN HUJAN KEDUA BRANKAS
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(brankas1['RR'], bins=50, kde=True, ax=axes[0], color='royalblue')
axes[0].set_title('Distribusi Curah Hujan Brankas 1 (Satelit ERA5-Land)')
axes[0].set_xlabel('RR (mm)')
axes[0].axvline(50, color='red', linestyle='--', linewidth=2, label='Batas Ekstrem (50mm)')
axes[0].legend()

sns.histplot(brankas2['RR'], bins=50, kde=True, ax=axes[1], color='forestgreen')
axes[1].set_title('Distribusi Curah Hujan Brankas 2 (BMKG Ground-Truth)')
axes[1].set_xlabel('RR (mm)')
axes[1].axvline(50, color='red', linestyle='--', linewidth=2, label='Batas Ekstrem (50mm)')
axes[1].legend()

plt.suptitle('Analisis Distribusi Curah Hujan Dual Brankas (Ambang Ekstrem >= 50mm)', 
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

ds_era5.close()
