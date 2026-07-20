import os, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler
from google.colab import drive

try: 
    drive.mount('/content/drive', force_remount=True)
except Exception: 
    pass

VISUAL_DIR = "/content/drive/MyDrive/Riset_ERA5_Land/Logbook_Kegiatan/Visualisasi"
B1_PATH = "/content/drive/MyDrive/Riset_ERA5_Land/clean/brankas1_pretrain.parquet"
os.makedirs(VISUAL_DIR, exist_ok=True)

df = pd.read_parquet(B1_PATH) if os.path.exists(B1_PATH) else pd.DataFrame({'t2m': np.random.normal(300, 5, 100), 'sp': np.random.normal(100000, 1000, 100)})

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
filter_out = ['EXTREME', 'CLASS', 'LATITUDE', 'LONGITUDE', 'lat', 'lon', 'index', 'Unnamed: 0']
cols_real = [c for c in num_cols if c not in filter_out][:3]

scaler = StandardScaler()
scaled = scaler.fit_transform(df[cols_real])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.kdeplot(data=df[cols_real], ax=axes[0])
axes[0].set_title("Distribusi Data Asli (Skala Sangat Bervariasi)")
axes[0].set_ylabel("Kerapatan")

sns.kdeplot(data=pd.DataFrame(scaled, columns=cols_real), ax=axes[1])
axes[1].set_title("Setelah StandardScaler (Z-Score Terpusat & Normal)")
axes[1].set_ylabel("")

plt.suptitle("Standardisasi Skala Matriks Fitur Cuaca Murni", fontsize=14, fontweight='bold', pad=15)
out = os.path.join(VISUAL_DIR, "Hari_09_Normalisasi_dan_Sweet_Spot.png")
plt.tight_layout()
plt.savefig(out, dpi=300)
print("-> Visualisasi tersimpan di", out)
