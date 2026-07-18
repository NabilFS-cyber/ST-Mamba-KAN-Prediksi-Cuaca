# =====================================================================
# PHASE 2 – DATASET ANALYSIS & QUALITY CHECK (ERA5-LAND & BMKG)
# =====================================================================
from google.colab import drive
import os, glob, pandas as pd, numpy as np, xarray as xr, warnings
warnings.filterwarnings('ignore')

# 1. Mount Google Drive & Set Root Folder Baru
drive.mount('/content/drive')
drive_folder = "/content/drive/MyDrive/Riset_ERA5_Land/"
BMKG_ROOT = os.path.join(drive_folder, "Data_BMKG/")
analysis_output_dir = os.path.join(drive_folder, "analysis")
os.makedirs(analysis_output_dir, exist_ok=True)

# 2. Deteksi Otomatis Seluruh 125 File .nc ERA5-Land
era5_files = sorted(glob.glob(os.path.join(drive_folder, "dataset_era5_land_*.nc")))
print(f'✅ Berhasil mendeteksi: {len(era5_files)} file NetCDF (.nc) bulanan.')

# 3. Deteksi File Excel BMKG
bmkg_files = sorted(glob.glob(os.path.join(BMKG_ROOT, '*.xlsx')))
print(f'✅ Berhasil mendeteksi: {len(bmkg_files)} file Excel BMKG Stasiun Bumi.\n')

# Fungsi Hitung Statistik Data Satelit
def compute_stats_da(da):
    return {
        'mean': float(da.mean().values),
        'median': float(da.median().values),
        'min': float(da.min().values),
        'max': float(da.max().values),
        'std': float(da.std().values),
        'missing_%': float(da.isnull().mean().values) * 100
    }

# =====================================================================
# A. SCANNING & AGGREGATE STATS UNTUK 125 FILE ERA5-LAND
# =====================================================================
# Menggunakan 11 Variabel Hidrometeorologi Daratan Utama Hasil Download
variables_era5_land = ['u10', 'v10', 'd2m', 't2m', 'sp', 'tp', 'ssrd', 'skt', 'swvl1', 'swvl2', 'evabs']
all_era5_stats = []

print("🔄 Memulai pemindaian data karakteristik 125 file Satelit...")
print("-" * 70)

for idx, f in enumerate(era5_files):
    try:
        # Buka file per bulan menggunakan engine h5netcdf bypass
        ds = xr.open_dataset(f, engine='h5netcdf')
        nama_file = os.path.basename(f)

        for var in variables_era5_land:
            if var in ds.variables:
                da = ds[var]
                stats = compute_stats_da(da)
                stats.update({'file': nama_file, 'variable': var})
                all_era5_stats.append(stats)
        ds.close()

        # Print progress singkat per 20 file biar tidak menemu penuhi layar log
        if (idx + 1) % 20 == 0 or (idx + 1) == len(era5_files):
            print(f" 🟩 Progress: {idx + 1}/{len(era5_files)} file satelit sukses diperiksa.")

    except Exception as e:
        print(f"❌ Gagal memproses file satelit {os.path.basename(f)}. Error: {e}")

if all_era5_stats:
    df_era5_stats = pd.DataFrame(all_era5_stats)
    df_era5_stats.to_csv(os.path.join(analysis_output_dir, 'era5_land_125files_stats.csv'), index=False)
    print('\n✅ STATS 125 FILE ERA5-LAND BERHASIL DIAMANKAN DI DRIVE!')
else:
    print('\n❌ Gagal mengekstrak statistik data satelit.')


# =====================================================================
# B. SCANNING & PENJINAKAN ANOMALI DATA 5 EXCEL BMKG
# =====================================================================
bmkg_stats = []
columns_to_analyze = ['TX', 'RH_AVG', 'RR', 'SS', 'FF_X'] # Memfokuskan pada kolom numerik utama

print("\n" + "="*70)
print("🔄 Memulai pembersihan & analisis data lapangan 5 Stasiun BMKG...")
print("-" * 70)

for f in bmkg_files:
    try:
        df = pd.read_excel(f)
        nama_stasiun = os.path.basename(f)
        print(f"📊 Menjinakkan data: {nama_stasiun}")

        for col in columns_to_analyze:
            if col in df.columns:
                # 🛠️ LANGKAH PENYELAMATAN 1: Koreksi Pemisah Desimal Koma Khas Indonesia
                if df[col].dtype == object:
                    df[col] = df[col].astype(str).str.replace(',', '.', regex=False)

                # Ubah paksa menjadi numerik murni
                series = pd.to_numeric(df[col], errors='coerce')

                # 🛠️ LANGKAH PENYELAMATAN 2: Netralisir Flag Error 8888 BMKG menjadi NaN
                series = series.replace(8888, np.nan)

                stats = {
                    'file': nama_stasiun,
                    'variable': col,
                    'mean': series.mean(),
                    'median': series.median(),
                    'min': series.min(),
                    'max': series.max(),
                    'std': series.std(),
                    'missing_%': series.isna().mean() * 100
                }
                bmkg_stats.append(stats)
                print(f"   -> Fitur '{col}' bersih. Cacat data/Missing: {stats['missing_%']:.2f}%")

    except Exception as e:
        print(f"❌ Gagal membaca stasiun {os.path.basename(f)}. Error: {e}")

if bmkg_stats:
    df_bmkg_stats = pd.DataFrame(bmkg_stats)
    df_bmkg_stats.to_csv(os.path.join(analysis_output_dir, 'bmkg_clean_stats.csv'), index=False)
    print('\n✅ STATS BERSIH DATA LAPANGAN BMKG BERHASIL DIKUNCI DI DRIVE!')
else:
    print('\n❌ Gagal mengekstrak statistik data BMKG.')

print("\n🎉 PHASE 1 SELESAI TOTAL! Seluruh data satelit dan data stasiun bumi Anda telah disterilkan.")
