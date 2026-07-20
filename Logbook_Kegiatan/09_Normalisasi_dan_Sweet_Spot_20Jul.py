import os, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler
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

df = pd.read_csv(DATA_PATH) if DATA_PATH.endswith('.csv') else pd.read_parquet(DATA_PATH)

# Pilih 3 fitur penting yang terisi penuh dari fusi data BMKG & ERA5 untuk demo standardisasi
target_features = ['TX', 'RH_AVG', 'RR']
cols_real = [c for c in target_features if c in df.columns]

if len(cols_real) < 3:
    # Fallback ke kolom numerik acak jika kolom resmi tidak ditemukan
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    filter_out = ['EXTREME', 'CLASS', 'LATITUDE', 'LONGITUDE', 'lat', 'lon', 'index', 'Unnamed: 0']
    cols_real = [c for c in num_cols if c not in filter_out][:3]

scaler = StandardScaler()
scaled = scaler.fit_transform(df[cols_real])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot sebelum standardisasi (suhu puluhan, kelembapan puluhan/ratusan, hujan bervariasi)
sns.kdeplot(data=df[cols_real], ax=axes[0])
axes[0].set_title("Distribusi Data Asli (Skala Sangat Jomplang/Bervariasi)")
axes[0].set_ylabel("Kerapatan")
axes[0].grid(True, alpha=0.3)

# Plot sesudah standardisasi (seluruh fitur terpusat di mean=0 dengan std=1)
sns.kdeplot(data=pd.DataFrame(scaled, columns=cols_real), ax=axes[1])
axes[1].set_title("Setelah StandardScaler (Z-Score Terpusat & Sejajar)")
axes[1].set_ylabel("")
axes[1].grid(True, alpha=0.3)

plt.suptitle("Standardisasi Skala Matriks Fitur Fusi Iklim Hibrida", fontsize=14, fontweight='bold', y=0.98)
out = os.path.join(VISUAL_DIR, "Hari_09_Normalisasi_dan_Sweet_Spot.png")
plt.tight_layout()
plt.savefig(out, dpi=300)
print("-> Visualisasi tersimpan di", out)
