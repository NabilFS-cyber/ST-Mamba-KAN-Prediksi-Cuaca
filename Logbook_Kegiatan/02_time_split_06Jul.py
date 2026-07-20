import os, matplotlib.pyplot as plt

from google.colab import drive
try:
    drive.mount('/content/drive', force_remount=True)
except Exception:
    pass

VISUAL_DIR = "/content/drive/MyDrive/Riset_ERA5_Land/Logbook_Kegiatan/Visualisasi"
os.makedirs(VISUAL_DIR, exist_ok=True)

print("Simulasi Time-Split Dual Brankas")
sizes = [368880, 68976]
labels = [f'Brankas 1 (Pra-Pelatihan)\n{sizes[0]} Baris', f'Brankas 2 (Penyesuaian)\n{sizes[1]} Baris']
plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, colors=['#1f77b4', '#ff7f0e'], autopct='%1.1f%%', explode=(0.1, 0))
plt.title("Rasio Time-Split Historis")
output_path = os.path.join(VISUAL_DIR, "Hari_02_Time_Split.png")
plt.savefig(output_path)
print("-> Visualisasi disimpan di", output_path)
