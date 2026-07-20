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
df_sample = df[df['STATION'] == 'Stasiun Meteorologi Kemayoran'].head(50).copy()
df_sample['DATE'] = pd.to_datetime(df_sample['TANGGAL_FUSI'])

# Inject NaNs untuk mensimulasikan sensor rusak
df_raw = df_sample.copy()
np.random.seed(42)
nan_idx = np.random.choice(df_raw.index, size=15, replace=False)
df_raw.loc[nan_idx, 'TX'] = np.nan

# Bersihkan dengan interpolasi FFill & BFill
df_clean = df_raw.copy()
df_clean['TX'] = df_clean['TX'].ffill().bfill()

plt.figure(figsize=(12, 5))
plt.plot(df_clean['DATE'], df_clean['TX'], color='green', label='Data Bersih (Interpolasi FFill/BFill)', linewidth=2, marker='o')
plt.plot(df_raw['DATE'], df_raw['TX'], color='red', label='Data Asli BMKG (Sensor Bolong)', linewidth=0, marker='x', markersize=10, markeredgewidth=2)

plt.title("Simulasi Pembersihan Missing Values Suhu Maksimum (TX)", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Tanggal")
plt.ylabel("Suhu Maksimum (°C)")
plt.legend()
plt.grid(True, alpha=0.3)

out = os.path.join(VISUAL_DIR, "Hari_02_Missing_Values.png")
plt.tight_layout()
plt.savefig(out, dpi=300)
print("-> Visualisasi tersimpan di", out)
