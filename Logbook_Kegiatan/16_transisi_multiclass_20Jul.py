import os, matplotlib.pyplot as plt

# Mount Google Drive untuk Google Colab
from google.colab import drive
try:
    drive.mount('/content/drive', force_remount=True)
except Exception:
    pass

VISUAL_DIR = "/content/drive/MyDrive/Riset_ERA5_Land/Logbook_Kegiatan/Visualisasi"
os.makedirs(VISUAL_DIR, exist_ok=True)

print("[HARI 16] Transisi Peringatan Kewaspadaan 3-Level (Multi-Class)")
plt.figure(figsize=(7, 4))
bars = plt.bar(['Aman (<20mm)', 'Waspada (20-100mm)', 'Siaga (>100mm)'], [5632, 5632, 5632], color=['green', 'orange', 'red'])
plt.title("Hari 16: Distribusi Kelas Terkunci Pasca SMOTETomek (Multi-Class)")
plt.ylabel("Jumlah Sampel Latih")
output_path = os.path.join(VISUAL_DIR, "Hari_16_Distribusi_Multiclass.png")
plt.savefig(output_path)
print("-> Visualisasi disimpan di", output_path)
print("TUGAS SELESAI 100%!")
