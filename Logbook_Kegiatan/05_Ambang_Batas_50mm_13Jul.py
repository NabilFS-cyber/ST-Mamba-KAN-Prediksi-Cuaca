import os, matplotlib.pyplot as plt, numpy as np, seaborn as sns

from google.colab import drive
try: drive.mount('/content/drive', force_remount=True)
except Exception: pass

VISUAL_DIR = "/content/drive/MyDrive/Riset_ERA5_Land/Logbook_Kegiatan/Visualisasi"
os.makedirs(VISUAL_DIR, exist_ok=True)

print("Kalkulasi Distribusi Heavy-Tail & Ambang Ekstrem (50mm)")
np.random.seed(42)
# Sintesis data hujan (Log-normal distribution -> heavy tail)
hujan = np.random.lognormal(mean=2.0, sigma=1.0, size=10000)
hujan = hujan[hujan > 0.1] # Buang hari cerah mutlak

plt.figure(figsize=(9, 5))
sns.histplot(hujan, bins=200, color='dodgerblue', stat='density', kde=True)
plt.axvline(50, color='red', linestyle='--', linewidth=2.5, label='Batas Kritis Siaga Banjir (50 mm)')
plt.xlim(0, 150)
plt.title("Distribusi Heavy-Tail Curah Hujan Hibrida (Jabodetabek)", fontsize=13)
plt.xlabel("Curah Hujan Harian (mm)")
plt.ylabel("Densitas Peluang")
plt.legend(loc='upper right', fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
out = os.path.join(VISUAL_DIR, "Hari_05_Ambang_50mm.png")
plt.savefig(out, dpi=300)
print("-> Visualisasi tersimpan di", out)
