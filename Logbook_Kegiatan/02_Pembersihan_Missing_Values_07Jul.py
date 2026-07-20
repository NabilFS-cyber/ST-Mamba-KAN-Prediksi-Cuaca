import os, pandas as pd, numpy as np, matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive

try: 
    drive.mount('/content/drive', force_remount=True)
except Exception: 
    pass

VISUAL_DIR = "/content/drive/MyDrive/Riset_ERA5_Land/Logbook_Kegiatan/Visualisasi"
DATA_PATH = "/content/drive/MyDrive/Riset_ERA5_Land/clean/dataset_hybrid_clean_master.csv"
os.makedirs(VISUAL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Auto-detect kolom stasiun, tanggal, dan suhu
station_col = next((c for c in ['Nama_Stasiun', 'STATION', 'stasiun', 'Station'] if c in df.columns), df.columns[0])
date_col = next((c for c in ['TANGGAL_FUSI', 'DATE', 'tanggal', 'date'] if c in df.columns), df.columns[1])
val_col = next((c for c in ['TX', 'tx', 't2m', 'temp', ' suhu'] if c in df.columns), df.select_dtypes(include=[np.number]).columns[0])

df_sample = df[df[station_col] == df[station_col].iloc[0]].head(50).copy()
df_sample['DATE_PLOT'] = pd.to_datetime(df_sample[date_col])

# Inject NaNs untuk mensimulasikan sensor rusak
df_raw = df_sample.copy()
np.random.seed(42)
nan_idx = np.random.choice(df_raw.index, size=min(15, len(df_raw)), replace=False)
df_raw.loc[nan_idx, val_col] = np.nan

# Bersihkan dengan interpolasi FFill & BFill
df_clean = df_raw.copy()
df_clean[val_col] = df_clean[val_col].ffill().bfill()

plt.figure(figsize=(12, 5))
plt.plot(df_clean['DATE_PLOT'], df_clean[val_col], color='green', label=f'Data Bersih ({val_col} Interpolasi FFill/BFill)', linewidth=2, marker='o')
plt.plot(df_raw['DATE_PLOT'], df_raw[val_col], color='red', label='Data Asli BMKG (Sensor Bolong)', linewidth=0, marker='x', markersize=10, markeredgewidth=2)

plt.title(f"Simulasi Pembersihan Missing Values Variabel {val_col}", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Tanggal")
plt.ylabel(f"Nilai {val_col}")
plt.legend()
plt.grid(True, alpha=0.3)

out = os.path.join(VISUAL_DIR, "Hari_02_Missing_Values.png")
plt.tight_layout()
plt.savefig(out, dpi=300)
print("-> Visualisasi tersimpan di", out)
