import os, matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive

try: 
    drive.mount('/content/drive', force_remount=True)
except Exception: 
    pass

VISUAL_DIR = "/content/drive/MyDrive/Riset_ERA5_Land/Logbook_Kegiatan/Visualisasi"
os.makedirs(VISUAL_DIR, exist_ok=True)

plt.figure(figsize=(10, 7))
stations = {
    'Stasiun Maritim Tg. Priok': (106.88053, -6.10781),
    'Stasiun Meteorologi Soekarno-Hatta': (106.65000, -6.12000),
    'Stasiun Meteorologi Kemayoran': (106.84000, -6.15559),
    'Stasiun Klimatologi Jawa Barat': (106.75000, -6.50000),
    'Stasiun Meteorologi Citeko': (106.95000, -6.70000)
}
sns.set_style("whitegrid")

for name, (lon, lat) in stations.items():
    plt.scatter(lon, lat, s=250, marker='^', color='red', edgecolor='black', zorder=5)
    plt.text(lon + 0.005, lat + 0.01, name, fontsize=11, fontweight='bold', 
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=4))

plt.xlim(106.5, 107.1)
plt.ylim(-6.8, -6.0)
plt.title("Pemetaan Spasial 5 Stasiun BMKG Jabodetabek (Grid ERA5-Land)", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Longitude (°BT)", fontsize=12)
plt.ylabel("Latitude (°LS)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

out = os.path.join(VISUAL_DIR, "Hari_01_Pemetaan_Spasial.png")
plt.tight_layout()
plt.savefig(out, dpi=300)
print("-> Visualisasi tersimpan di", out)
