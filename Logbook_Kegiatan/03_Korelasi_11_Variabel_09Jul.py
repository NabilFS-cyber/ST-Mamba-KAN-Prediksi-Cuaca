import os, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from google.colab import drive

try: 
    drive.mount('/content/drive', force_remount=True)
except Exception: 
    pass

VISUAL_DIR = "/content/drive/MyDrive/Riset_ERA5_Land/Logbook_Kegiatan/Visualisasi"
DATA_PATH = "/content/drive/MyDrive/Riset_ERA5_Land/clean/dataset_hybrid_clean_master.csv"
if not os.path.exists(DATA_PATH):
    DATA_PATH = "/content/drive/MyDrive/Riset_ERA5_Land/clean/brankas2_finetune.parquet"

os.makedirs(VISUAL_DIR, exist_ok=True)

df = pd.read_parquet(DATA_PATH) if DATA_PATH.endswith('.parquet') else pd.read_csv(DATA_PATH)

# Pilih kolom numerik
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Filter out metadata/coords/labels jika ada
filter_out = ['EXTREME', 'CLASS', 'LATITUDE', 'LONGITUDE', 'lat', 'lon', 'index', 'Unnamed: 0']
cols_real = [c for c in num_cols if c not in filter_out][:11]

corr = df[cols_real].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, linewidths=.5)
plt.title(f"Matriks Korelasi Pearson {len(cols_real)} Variabel Cuaca (Sanity Check Logika Alam)", fontsize=13, fontweight='bold', pad=15)

out = os.path.join(VISUAL_DIR, "Hari_03_Korelasi_Variabel.png")
plt.tight_layout()
plt.savefig(out, dpi=300)
print("-> Visualisasi tersimpan di", out)
