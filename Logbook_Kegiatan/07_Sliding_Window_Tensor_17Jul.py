import os, matplotlib.pyplot as plt, numpy as np

from google.colab import drive
try: drive.mount('/content/drive', force_remount=True)
except Exception: pass

VISUAL_DIR = "/content/drive/MyDrive/Riset_ERA5_Land/Logbook_Kegiatan/Visualisasi"
os.makedirs(VISUAL_DIR, exist_ok=True)

print("Perakitan Tensor Sliding Window 14 Hari")
t = np.arange(1, 21)
hujan = np.sin(t * 0.4) * 20 + 25 + np.random.randn(20) * 5

plt.figure(figsize=(10, 5))
plt.plot(t, hujan, marker='o', color='lightgray', linestyle='--', label='Deret Historis')
plt.plot(t[2:16], hujan[2:16], marker='o', color='mediumblue', linewidth=3, label='Memori Look-back (14 Hari / 336 Jam)')
plt.scatter(t[16], hujan[16], color='red', s=200, edgecolors='black', zorder=5, label='Target Prediksi (H+1)')

# Annotations
plt.axvspan(3, 16, color='blue', alpha=0.1)
plt.axvline(17, color='red', linestyle=':', alpha=0.5)
plt.title("Desain Arsitektur Tensor Sliding Window ST-Mamba", fontsize=13)
plt.xlabel("Indeks Waktu Berjalan (Harian)", fontsize=11)
plt.ylabel("Intensitas Curah Hujan (mm)", fontsize=11)
plt.xticks(t)
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
out = os.path.join(VISUAL_DIR, "Hari_07_Sliding_Window.png")
plt.savefig(out, dpi=300)
print("-> Visualisasi tersimpan di", out)
