import os, numpy as np, matplotlib.pyplot as plt

from google.colab import drive
try:
    drive.mount('/content/drive', force_remount=True)
except Exception:
    pass

VISUAL_DIR = "/content/drive/MyDrive/Riset_ERA5_Land/Logbook_Kegiatan/Visualisasi"
os.makedirs(VISUAL_DIR, exist_ok=True)

print("Peningkatan Ketajaman Satelit (31km vs 9km)")
grid_31km = np.random.rand(5, 5)
grid_9km = np.random.rand(15, 15)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.imshow(grid_31km, cmap='hot', interpolation='nearest')
ax1.set_title("ERA5 Lama (31km) - Blur/Kasar")
ax2.imshow(grid_9km, cmap='hot', interpolation='nearest')
ax2.set_title("ERA5-Land (9km) - UHI Terdeteksi!")
plt.suptitle("Keunggulan Resolusi Spasial ERA5-Land")
output_path = os.path.join(VISUAL_DIR, "Satelit_Resolution.png")
plt.savefig(output_path)
print("-> Visualisasi disimpan di", output_path)
