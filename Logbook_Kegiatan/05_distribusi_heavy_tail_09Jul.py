import os, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')

from google.colab import drive
try:
    drive.mount('/content/drive', force_remount=True)
except Exception:
    pass

VISUAL_DIR = "/content/drive/MyDrive/Riset_ERA5_Land/Logbook_Kegiatan/Visualisasi"
os.makedirs(VISUAL_DIR, exist_ok=True)
HYBRID = "/content/drive/MyDrive/Riset_ERA5_Land/clean/dataset_hybrid_clean_master.csv"

print("Kalkulasi Distribusi Heavy-Tail")
if os.path.exists(HYBRID):
    df = pd.read_csv(HYBRID)
    rr = df['RR'][df['RR'] > 0]
    
    plt.figure(figsize=(10, 5))
    sns.histplot(rr, bins=50, kde=True, log_scale=(False, True))
    plt.axvline(20, color='orange', linestyle='--', label='Lebat (>20mm)')
    plt.axvline(50, color='red', linestyle='--', label='Sangat Lebat (>50mm)')
    plt.title("Distribusi Hujan Heavy-Tail")
    plt.legend()
    output_path = os.path.join(VISUAL_DIR, "Hari_05_Heavy_Tail.png")
    plt.savefig(output_path)
    print("-> Visualisasi disimpan di", output_path)
else:
    print(f"File hybrid tidak ditemukan di Google Drive: {HYBRID}")
