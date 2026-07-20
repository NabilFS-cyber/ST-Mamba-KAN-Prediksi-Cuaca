import os, numpy as np, matplotlib.pyplot as plt
VISUAL_DIR = "Visualisasi"
os.makedirs(VISUAL_DIR, exist_ok=True)

print("[HARI 7] Evaluasi Integritas Waktu Awan SMOTE")
# Simulasi deret waktu: Garis lurus patah-patah = interpolasi linear SMOTE (awan hantu)
x = np.linspace(0, 10, 50)
asli = np.sin(x) # Dinamika cuaca asli
smote = np.linspace(asli[0], asli[-1], 50) # Hantu linear

plt.figure(figsize=(8, 4))
plt.plot(x, asli, label="Dinamika Hukum Alam", color='blue')
plt.plot(x, smote, label="Interpolasi SMOTE (Awan Hantu)", color='red', linestyle='--')
plt.title("Hari 7: Mengapa SMOTE Biasa Dibatalkan")
plt.legend()
plt.savefig(os.path.join(VISUAL_DIR, "Hari_07_Awan_Hantu_SMOTE.png"))
print("-> Visualisasi disimpan di", os.path.join(VISUAL_DIR, "Hari_07_Awan_Hantu_SMOTE.png"))
