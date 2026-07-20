import os, numpy as np, matplotlib.pyplot as plt
VISUAL_DIR = "Visualisasi"
os.makedirs(VISUAL_DIR, exist_ok=True)

print("[HARI 13] Peningkatan Ketajaman Satelit (31km vs 9km)")
# Simulasi heatmap pulau bahang perkotaan (UHI)
grid_31km = np.random.rand(5, 5)
grid_9km = np.random.rand(15, 15)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.imshow(grid_31km, cmap='hot', interpolation='nearest')
ax1.set_title("ERA5 Lama (31km) - Blur/Kasar")
ax2.imshow(grid_9km, cmap='hot', interpolation='nearest')
ax2.set_title("ERA5-Land (9km) - UHI Terdeteksi!")
plt.suptitle("Hari 13: Keunggulan Resolusi Spasial ERA5-Land")
plt.savefig(os.path.join(VISUAL_DIR, "Hari_13_Satelit_Resolution.png"))
print("-> Visualisasi disimpan di", os.path.join(VISUAL_DIR, "Hari_13_Satelit_Resolution.png"))
