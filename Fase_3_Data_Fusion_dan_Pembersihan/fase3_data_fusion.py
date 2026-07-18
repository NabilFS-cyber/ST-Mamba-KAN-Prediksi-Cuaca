# =====================================================================
# PHASE 3 – CLEANING, OUTLIER DETECTION & HYBRID FUSION
# =====================================================================
from google.colab import drive
import os, glob, warnings
import pandas as pd
import numpy as np
import xarray as xr
warnings.filterwarnings('ignore')

# 1️⃣ Mount Drive & Inisialisasi Struktur Folder Steril
drive.mount('/content/drive')
drive_folder = "/content/drive/MyDrive/Riset_ERA5_Land/"
BMKG_ROOT    = os.path.join(drive_folder, "Data_BMKG/")
OUTPUT_ROOT  = os.path.join(drive_folder, "clean/")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# 🔥 FIX: Koordinat Resmi 5 Stasiun BMKG (Sesuai Situs BMKG Asli)
stasiun_config = {
    "Stasiun Meteorologi Soekarno Hatta": {"lat": -6.12000, "lon": 106.65000, "pattern": "*Soekarno*"},
    "Stasiun Meteorologi Maritim Tanjung Priok": {"lat": -6.10781, "lon": 106.88053, "pattern": "*Tanjung*"},
    "Stasiun Meteorologi Kemayoran": {"lat": -6.15559, "lon": 106.84000, "pattern": "*Kemayoran*"},
    "Stasiun Meteorologi Citeko": {"lat": -6.70000, "lon": 106.85000, "pattern": "*Citeko*"},
    "Stasiun Klimatologi Jawa Barat": {"lat": -6.50000, "lon": 106.75000, "pattern": "*Jawa Barat*"}
}

# 2️⃣ Load & Jahit 125 Berkas ERA5-Land (Temporal Concatenation)
print("⏳ [STEP 1] Mendeteksi dan menjahit dimensi waktu 125 file NetCDF Satelit...")
era5_files = sorted(glob.glob(os.path.join(drive_folder, "dataset_era5_land_*.nc")))
print(f' -> Menemukan {len(era5_files)} berkas NetCDF bulanan.')

try:
    ds_master = xr.open_mfdataset(era5_files, engine='h5netcdf', combine='by_coords')
    print(' ✅ Sukses merakit master linimasa Satelit!')
except Exception as e:
    print(f' ❌ Gagal memuat kluster NetCDF. Detail: {e}')
    raise

# 3️⃣ Imputasi Data Kosong Satelit (Simple Forward-Fill)
print("\n⏳ [STEP 2] Melakukan temporal forward-fill pada grid data satelit...")
kolom_waktu_nc = 'valid_time' if 'valid_time' in ds_master.coords else 'time'
era5_filled = ds_master.ffill(kolom_waktu_nc)

# ✅ FITUR DIKEMBALIKAN: Simpan master data satelit bersih 'era5_clean.nc' ke Drive
print(" 💾 Menyimpan master data satelit bersih 'era5_clean.nc' ke Drive...")
print("    (Harap bersabar, proses rendering 125 file ke dalam 1 file ini memakan waktu beberapa menit...)")
era5_filled.to_netcdf(os.path.join(OUTPUT_ROOT, 'era5_clean.nc'), engine='h5netcdf')
print("  -> ✅ File 'era5_clean.nc' sukses dikunci dan tersimpan di Google Drive!")

# 4️⃣ Pembersihan Data Lapangan BMKG & Ekstraksi Koordinat Spasial
print("\n⏳ [STEP 3] Memulai pembersihan data stasiun bumi & ekstraksi spasial...")
print("-" * 85)

list_dataset_fusi = []
outlier_report = []

def detect_outliers(series):
    series = pd.to_numeric(series, errors='coerce').dropna()
    if series.empty: return pd.Series([], dtype=bool)
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    return (series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))

for nama_stasiun, info in stasiun_config.items():
    print(f"🔄 Memproses Stasiun: {nama_stasiun}")

    # A. Ekstraksi Spasial Satelit menggunakan titik BMKG Asli
    ds_lokal = era5_filled.sel(latitude=info["lat"], longitude=info["lon"], method="nearest")
    df_satelit = ds_lokal.to_dataframe().reset_index()

    kolom_waktu = 'valid_time' if 'valid_time' in df_satelit.columns else 'time'
    df_satelit['TANGGAL_FUSI'] = pd.to_datetime(df_satelit[kolom_waktu]).dt.normalize()

    var_akumulasi = [v for v in ds_master.data_vars if v in ['tp', 'evabs', 'ssrd', 'ssr']]
    aturan_agregasi = {v: 'sum' if v in var_akumulasi else 'mean' for v in ds_master.data_vars}
    df_sat_daily = df_satelit.groupby('TANGGAL_FUSI').agg(aturan_agregasi).reset_index()

    # B. Load & Jinakkan Data Excel BMKG
    file_search = glob.glob(os.path.join(BMKG_ROOT, info["pattern"]))
    if not file_search:
        print(f"  ⚠️ Warning: File Excel untuk {nama_stasiun} tidak ditemukan!")
        continue

    df_bmkg = pd.read_excel(file_search[0])
    df_bmkg.columns = df_bmkg.columns.str.strip().str.upper()

    date_col = [c for c in df_bmkg.columns if c in ['DATE', 'TANGGAL', 'TIME', 'DATETIME']][0]

    df_bmkg['TANGGAL_FUSI'] = pd.to_datetime(df_bmkg[date_col], dayfirst=True, errors='coerce').dt.normalize()

    df_bmkg = df_bmkg.dropna(subset=['TANGGAL_FUSI'])
    df_bmkg = df_bmkg.set_index('TANGGAL_FUSI').sort_index()

    kolom_iklim = [c for c in df_bmkg.columns if c in ['TX', 'RH_AVG', 'RR', 'SS', 'FF_X']]
    for col in kolom_iklim:
        if df_bmkg[col].dtype == object:
            df_bmkg[col] = df_bmkg[col].astype(str).str.replace(',', '.', regex=False)
        df_bmkg[col] = pd.to_numeric(df_bmkg[col], errors='coerce')
        df_bmkg[col] = df_bmkg[col].replace([8888, 9999], np.nan)

    # C. Imputasi Linear (Aman secara kronologis)
    df_bmkg[kolom_iklim] = df_bmkg[kolom_iklim].interpolate(method='linear', limit_direction='both', errors='ignore')
    if 'RR' in df_bmkg.columns:
        df_bmkg['RR'] = df_bmkg['RR'].fillna(0)
        mask_outlier = detect_outliers(df_bmkg['RR'])
        outlier_report.append({'station': nama_stasiun, 'variable': 'RR', 'outliers_count': int(mask_outlier.sum())})

    df_bmkg = df_bmkg.ffill().bfill().reset_index()
    df_bmkg.to_csv(os.path.join(OUTPUT_ROOT, f"{nama_stasiun}_clean.csv"), index=False)

    # D. Perkawinan Data
    df_hybrid_stasiun = pd.merge(df_sat_daily, df_bmkg[['TANGGAL_FUSI'] + kolom_iklim], on='TANGGAL_FUSI', how='inner')
    df_hybrid_stasiun['Nama_Stasiun'] = nama_stasiun
    list_dataset_fusi.append(df_hybrid_stasiun)
    print(f"   -> ✅ Stasiun '{nama_stasiun}' Bersih & Ter-fusi! ({len(df_hybrid_stasiun)} baris)")

# 5️⃣ Konsolidasi Akhir
print("\n" + "="*85)
if list_dataset_fusi:
    df_master_final = pd.concat(list_dataset_fusi, ignore_index=True)
    df_master_final = df_master_final.dropna(subset=['RR'])

    csv_master_path = os.path.join(OUTPUT_ROOT, "dataset_hybrid_clean_master.csv")
    df_master_final.to_csv(csv_master_path, index=False)

    df_outlier_report = pd.DataFrame(outlier_report)
    df_outlier_report.to_csv(os.path.join(OUTPUT_ROOT, 'bmkg_outliers.csv'), index=False)

    print(f"🎉 MASTERPIECE PHASE 3 SELESAI BULAT! SELURUH ASET MATRIKS EMAS TERBENTUK:")
    print(f" 👉 Master Fusi Tabel: {csv_master_path}")
    print(f"📊 Dimensi Matriks: {df_master_final.shape[0]} baris x {df_master_final.shape[1]} kolom.")
else:
    print("❌ Gagal total membangun fusi data hibrida riset.")

ds_master.close()
