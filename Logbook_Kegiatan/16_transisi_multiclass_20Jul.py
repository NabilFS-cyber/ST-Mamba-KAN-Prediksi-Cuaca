import os, matplotlib.pyplot as plt
VISUAL_DIR = "Visualisasi"
os.makedirs(VISUAL_DIR, exist_ok=True)

print("[HARI 16] Transisi Peringatan Kewaspadaan 3-Level (Multi-Class)")
plt.figure(figsize=(7, 4))
bars = plt.bar(['Aman (<20mm)', 'Waspada (20-100mm)', 'Siaga (>100mm)'], [5632, 5632, 5632], color=['green', 'orange', 'red'])
plt.title("Hari 16: Distribusi Kelas Terkunci Pasca SMOTETomek (Multi-Class)")
plt.ylabel("Jumlah Sampel Latih")
plt.savefig(os.path.join(VISUAL_DIR, "Hari_16_Distribusi_Multiclass.png"))
print("-> Visualisasi disimpan di", os.path.join(VISUAL_DIR, "Hari_16_Distribusi_Multiclass.png"))
print("TUGAS SELESAI 100%!")
