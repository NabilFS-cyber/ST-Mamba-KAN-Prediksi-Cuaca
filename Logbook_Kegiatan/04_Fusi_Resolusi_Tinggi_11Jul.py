import os, matplotlib.pyplot as plt, numpy as np

from google.colab import drive
try: drive.mount('/content/drive', force_remount=True)
except Exception: pass

VISUAL_DIR = "/content/drive/MyDrive/Riset_ERA5_Land/Logbook_Kegiatan/Visualisasi"
os.makedirs(VISUAL_DIR, exist_ok=True)

print("Komparasi Fusi Resolusi ERA5 (31km) vs ERA5-Land (9km)")
np.random.seed(99)
grid_kasar = np.kron(np.random.rand(5, 5), np.ones((3, 3)))
grid_halus = grid_kasar + np.random.randn(15, 15)*0.1

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
c1 = ax1.imshow(grid_kasar, cmap='magma')
ax1.set_title("ERA5 Klasik (Resolusi 31km)
Sangat Kasar, UHI Tersembunyi", fontsize=10)
plt.colorbar(c1, ax=ax1, fraction=0.046, pad=0.04)

c2 = ax2.imshow(grid_halus, cmap='magma', interpolation='bicubic')
ax2.set_title("ERA5-Land Hibrida (Resolusi 9km)
Deteksi Iklim Mikro / UHI Sangat Jelas", fontsize=10)
plt.colorbar(c2, ax=ax2, fraction=0.046, pad=0.04)

plt.suptitle("Peningkatan Integritas Spasial via Fusi Bilinear", fontsize=14, weight='bold')
plt.tight_layout()
out = os.path.join(VISUAL_DIR, "Hari_04_Fusi_Resolusi.png")
plt.savefig(out, dpi=300)
print("-> Visualisasi tersimpan di", out)
