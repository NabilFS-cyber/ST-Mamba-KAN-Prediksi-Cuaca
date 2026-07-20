import os, numpy as np, matplotlib.pyplot as plt

# Mount Google Drive untuk Google Colab
from google.colab import drive
try:
    drive.mount('/content/drive', force_remount=True)
except Exception:
    pass

VISUAL_DIR = "/content/drive/MyDrive/Riset_ERA5_Land/Logbook_Kegiatan/Visualisasi"
os.makedirs(VISUAL_DIR, exist_ok=True)

print("[HARI 7] Evaluasi Integritas Waktu Awan SMOTE")
x = np.linspace(0, 10, 50)
asli = np.sin(x)
smote = np.linspace(asli[0], asli[-1], 50)

plt.figure(figsize=(8, 4))
plt.plot(x, asli, label="Dinamika Hukum Alam", color='blue')
plt.plot(x, smote, label="Interpolasi SMOTE (Awan Hantu)", color='red', linestyle='--')
plt.title("Hari 7: Mengapa SMOTE Biasa Dibatalkan")
plt.legend()
output_path = os.path.join(VISUAL_DIR, "Hari_07_Awan_Hantu_SMOTE.png")
plt.savefig(output_path)
print("-> Visualisasi disimpan di", output_path)
