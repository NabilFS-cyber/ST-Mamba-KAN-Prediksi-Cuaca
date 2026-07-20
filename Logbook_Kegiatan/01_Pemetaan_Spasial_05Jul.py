import os, numpy as np, matplotlib.pyplot as plt

from google.colab import drive
try: drive.mount('/content/drive', force_remount=True)
except Exception: pass

VISUAL_DIR = "/content/drive/MyDrive/Riset_ERA5_Land/Logbook_Kegiatan/Visualisasi"
os.makedirs(VISUAL_DIR, exist_ok=True)

print("Inventarisasi & Pemetaan Spasial")
# Simulasi Peta Topografi Sintetis Jabodetabek & Stasiun BMKG
x = np.linspace(106.5, 107.0, 100)
y = np.linspace(-6.8, -6.0, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(X*15) * np.cos(Y*15) + (Y*2)

plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, Z, levels=20, cmap='terrain')
plt.colorbar(contour, label='Elevasi/Anomali Topografi Relatif')

# Titik Stasiun Real
stations = {
    'Priok': (106.88, -6.10),
    'Soetta': (106.65, -6.12),
    'Kemayoran': (106.84, -6.15),
    'Citeko': (106.95, -6.70),
    'Jawa Barat': (106.75, -6.50)
}
for name, (lon, lat) in stations.items():
    plt.scatter(lon, lat, color='red', marker='^', s=150, edgecolors='black')
    plt.text(lon + 0.01, lat, name, fontsize=10, weight='bold', color='white', 
             bbox=dict(facecolor='black', alpha=0.5, pad=1))

plt.title("Pemetaan Grid Spasial ERA5-Land (9km) & 5 Stasiun BMKG", fontsize=12, pad=15)
plt.xlabel("Longitude (°BT)")
plt.ylabel("Latitude (°LS)")
plt.tight_layout()
out = os.path.join(VISUAL_DIR, "Hari_01_Pemetaan_Spasial.png")
plt.savefig(out, dpi=300)
print("-> Visualisasi tersimpan di", out)
