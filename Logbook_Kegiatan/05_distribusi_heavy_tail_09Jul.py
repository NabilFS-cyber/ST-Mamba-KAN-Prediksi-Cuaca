import os, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
VISUAL_DIR = "Visualisasi"
os.makedirs(VISUAL_DIR, exist_ok=True)
HYBRID = r"C:\kuliah nabil\DLL\PKM\PENDANAAN\Perancangan_Model_AI\Fase_3_Data_Fusion_dan_Pembersihan\Dataset\dataset_hybrid_clean_master.csv"

print("[HARI 5] Kalkulasi Distribusi Heavy-Tail")
if os.path.exists(HYBRID):
    df = pd.read_csv(HYBRID)
    rr = df['RR'][df['RR'] > 0]
    
    plt.figure(figsize=(10, 5))
    sns.histplot(rr, bins=50, kde=True, log_scale=(False, True))
    plt.axvline(20, color='orange', linestyle='--', label='Lebat (>20mm)')
    plt.axvline(50, color='red', linestyle='--', label='Sangat Lebat (>50mm)')
    plt.title("Hari 5: Distribusi Hujan Heavy-Tail")
    plt.legend()
    plt.savefig(os.path.join(VISUAL_DIR, "Hari_05_Heavy_Tail.png"))
    print("-> Visualisasi disimpan di", os.path.join(VISUAL_DIR, "Hari_05_Heavy_Tail.png"))
else:
    print("File hybrid tidak ditemukan.")
