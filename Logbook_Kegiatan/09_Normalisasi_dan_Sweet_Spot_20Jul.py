import os, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler
from google.colab import drive

try: 
    drive.mount('/content/drive', force_remount=True)
except Exception: 
    pass

VISUAL_DIR = "/content/drive/MyDrive/Riset_ERA5_Land/Logbook_Kegiatan/Visualisasi"
B1_PATH = "/content/drive/MyDrive/Riset_ERA5_Land/clean/brankas1_pretrain.parquet"
os.makedirs(VISUAL_DIR, exist_ok=True)

df = pd.read_parquet(B1_PATH)
features = ['t2m', 'tp', 'sp']
cols_real = [c for c in features if c in df.columns]

if cols_real:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[cols_real])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.kdeplot(data=df[cols_real], ax=axes[0])
    axes[0].set_title("Distribusi Data Asli (Skala Sangat Bervariasi)")
    axes[0].set_ylabel("Kerapatan")

    sns.kdeplot(data=pd.DataFrame(scaled, columns=cols_real), ax=axes[1])
    axes[1].set_title("Setelah StandardScaler (Z-Score Terpusat & Normal)")
    axes[1].set_ylabel("")

    plt.suptitle("Standardisasi Skala Matriks Fitur Cuaca Murni", fontsize=14, fontweight='bold')
    out = os.path.join(VISUAL_DIR, "Hari_09_Normalisasi_dan_Sweet_Spot.png")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    print("-> Visualisasi tersimpan di", out)
else:
    print("Kolom tidak ditemukan.")
