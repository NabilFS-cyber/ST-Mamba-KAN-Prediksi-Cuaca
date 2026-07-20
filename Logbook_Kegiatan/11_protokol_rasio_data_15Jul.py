import os, matplotlib.pyplot as plt

from google.colab import drive
try:
    drive.mount('/content/drive', force_remount=True)
except Exception:
    pass

VISUAL_DIR = "/content/drive/MyDrive/Riset_ERA5_Land/Logbook_Kegiatan/Visualisasi"
os.makedirs(VISUAL_DIR, exist_ok=True)

print("Protokol Rasio Himpunan Data Anti-Kebocoran")
plt.figure(figsize=(6, 6))
plt.pie([70, 15, 15], labels=['Latih (70%)', 'Validasi (15%)', 'Ujian (15%)'], autopct='%1.0f%%', colors=['green', 'gold', 'red'])
plt.title("Distribusi Dataset Bebas Kebocoran Ujian")
output_path = os.path.join(VISUAL_DIR, "Rasio_Data.png")
plt.savefig(output_path)
print("-> Visualisasi disimpan di", output_path)
