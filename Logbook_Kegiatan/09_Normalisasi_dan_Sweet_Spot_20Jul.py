import os, matplotlib.pyplot as plt, numpy as np, seaborn as sns

from google.colab import drive
try: drive.mount('/content/drive', force_remount=True)
except Exception: pass

VISUAL_DIR = "/content/drive/MyDrive/Riset_ERA5_Land/Logbook_Kegiatan/Visualisasi"
os.makedirs(VISUAL_DIR, exist_ok=True)

print("Finalisasi: Normalisasi & Sweet-Spot 3 Kelas")
np.random.seed(42)
suhu = np.random.normal(30, 2, 1000)
tekanan = np.random.normal(1010, 5, 1000)

suhu_scaled = (suhu - np.mean(suhu)) / np.std(suhu)
tekanan_scaled = (tekanan - np.mean(tekanan)) / np.std(tekanan)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# Plot 1: Standard Scaler
sns.kdeplot(suhu_scaled, ax=axes[0], fill=True, label='Suhu (Scaled)')
sns.kdeplot(tekanan_scaled, ax=axes[0], fill=True, label='Tekanan (Scaled)')
axes[0].set_title("Efek StandardScaler (Mean=0, Std=1)", fontsize=11)
axes[0].legend()

# Plot 2: Sweet Spot Waspada vs Siaga
curah = np.random.uniform(0, 80, 1000)
aman = curah[curah < 20]
waspada = curah[(curah >= 20) & (curah < 50)]
siaga = curah[curah >= 50]

axes[1].hist([aman, waspada, siaga], bins=20, stacked=True, 
             color=['#2ca02c', '#ff7f0e', '#d62728'], 
             label=['Aman (<20mm)', 'Waspada (20-50mm)', 'Siaga (>=50mm)'])
axes[1].set_title("Distribusi Kelas Sweet-Spot Target", fontsize=11)
axes[1].legend()

plt.tight_layout()
out = os.path.join(VISUAL_DIR, "Hari_09_Normalisasi_Sweet_Spot.png")
plt.savefig(out, dpi=300)
print("-> Visualisasi tersimpan di", out)
