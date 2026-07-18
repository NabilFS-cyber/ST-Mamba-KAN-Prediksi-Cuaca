# =====================================================================
# PHASE 5B – 4D SPATIO-TEMPORAL WINDOWING & FLATTENED SMOTETOMEK
# =====================================================================
# !pip -q install imbalanced-learn joblib

import os, warnings
import pandas as pd
import numpy as np
import joblib
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
warnings.filterwarnings('ignore')

from google.colab import drive
drive.mount('/content/drive')

DRIVE_FOLDER = '/content/drive/MyDrive/Riset_ERA5_Land/'
CLEAN_ROOT   = os.path.join(DRIVE_FOLDER, 'clean')
TENSOR_ROOT  = os.path.join(DRIVE_FOLDER, 'tensors_mamba')
os.makedirs(TENSOR_ROOT, exist_ok=True)

PATH_B1 = os.path.join(CLEAN_ROOT, 'brankas1_pretrain.parquet')
PATH_B2 = os.path.join(CLEAN_ROOT, 'brankas2_finetune.parquet')

print("⏳ [STEP 1] Memuat Berkas Parquet dari Brankas Emas...")
df_b1 = pd.read_parquet(PATH_B1)
df_b2 = pd.read_parquet(PATH_B2)

# ============================================================
# ⚠️ KRUSIAL: Urutan stasiun harus selalu konsisten untuk GNN!
# ============================================================
STATIONS = [
    "Stasiun Meteorologi Soekarno Hatta",
    "Stasiun Meteorologi Maritim Tanjung Priok",
    "Stasiun Meteorologi Kemayoran",
    "Stasiun Meteorologi Citeko",
    "Stasiun Klimatologi Jawa Barat"
]

for df in [df_b1, df_b2]:
    df['STATION'] = pd.Categorical(df['STATION'], categories=STATIONS, ordered=True)

df_b1 = df_b1.sort_values(by=['DATE', 'STATION']).reset_index(drop=True)
df_b2 = df_b2.sort_values(by=['DATE', 'STATION']).reset_index(drop=True)

EXCLUDE = ['DATE', 'STATION', 'EXTREME', 'RR']
FEATURES = [c for c in df_b1.columns if c not in EXCLUDE]
N_WEATHER_FEATURES = len(FEATURES) # 17

print("\n⏳ [STEP 2] Smart Imputation NaN & Fallback...")
bmkg_cols = ['TX', 'RH_AVG', 'SS', 'FF_X']
for col in bmkg_cols:
    if col in df_b2.columns and col in df_b1.columns:
        median_val = df_b2[col].median()
        if pd.isna(median_val): median_val = 0
        df_b1[col] = df_b1[col].fillna(median_val)
        df_b2[col] = df_b2[col].fillna(median_val)

# Cegah error Categorical dengan hanya mengisi NaN pada fitur numerik
for c in FEATURES + ['RR']:
    df_b1[c] = df_b1[c].fillna(0)
    df_b2[c] = df_b2[c].fillna(0)

# 🔥 KUNCI UTAMA 1: MULTI-CLASS (3 LEVEL)
def get_class(mm):
    if mm < 20.0: return 0     # KELAS 0: Ringan / Aman
    elif mm < 50.0: return 1   # KELAS 1: Sedang / Waspada
    else: return 2             # KELAS 2: Lebat & Ekstrem / Siaga

# 🔥 KUNCI UTAMA 2: 4D SPATIO-TEMPORAL WINDOWING
WINDOW = 14

def make_4d_sequences(df):
    # Paksa sinkronisasi spasial: Pastikan setiap tanggal memiliki tepat 5 stasiun!
    unique_dates = df['DATE'].unique()
    idx = pd.MultiIndex.from_product([unique_dates, STATIONS], names=['DATE', 'STATION'])
    
    # Hapus duplikat dan paksa bentuk matriks utuh (jika ada hari yang bolong di satu stasiun)
    df = df.drop_duplicates(subset=['DATE', 'STATION']).set_index(['DATE', 'STATION']).reindex(idx).reset_index()
    
    # Cerdas: Jangan isi dengan 0! Kita gunakan Forward Fill lalu Backward Fill per stasiun
    # agar pola cuaca (seperti suhu/kelembapan) tidak anjlok menjadi 0.
    for c in FEATURES + ['RR']:
        df[c] = df.groupby('STATION')[c].ffill().bfill().fillna(0)
        
    n_dates = len(unique_dates)
    
    # [N_Dates, 5 Stasiun, 17 Fitur Cuaca]
    all_features = df[FEATURES].values.reshape(n_dates, 5, N_WEATHER_FEATURES)
    # [N_Dates, 5 Stasiun]
    all_targets_r = df['RR'].values.reshape(n_dates, 5)
    
    X, y_reg, y_cls = [], [], []
    
    for i in range(n_dates - WINDOW):
        # [14 Hari, 5 Stasiun, 17 Fitur] (Ini adalah "Graf Cuaca Jabodetabek" utuh!)
        window_graph = all_features[i : i + WINDOW]
        
        # Kita membuat 5 sampel terpisah untuk hari yang sama.
        # Pada setiap sampel, kita menyuntikkan (one-hot) yang memberi tahu AI
        # stasiun mana yang harus dia prediksi.
        for target_idx in range(5):
            # One-hot indikator stasiun (14, 5, 1)
            one_hot = np.zeros((WINDOW, 5, 1))
            one_hot[:, target_idx, 0] = 1.0 
            
            # Gabungkan fitur cuaca (17) dengan indikator (1). Total 18 fitur!
            # Shape sampel X_sample: [14, 5, 18]
            x_sample = np.concatenate([window_graph, one_hot], axis=-1)
            
            X.append(x_sample)
            
            # Ambil target Hujan (RR) khusus untuk stasiun 'target_idx'
            rr_val = all_targets_r[i + WINDOW, target_idx]
            y_reg.append(rr_val)
            y_cls.append(get_class(rr_val))

    return np.array(X), np.array(y_reg), np.array(y_cls)

print(f"\n⏳ [STEP 3] Merakit Tensor 4D Mutlak (Window = {WINDOW} Hari)...")
# Menghasilkan Tensor 4D!
X_b1, yr_b1, yc_b1 = make_4d_sequences(df_b1)
X_b2, yr_b2, yc_b2 = make_4d_sequences(df_b2)
print(f"   -> Brankas 1: {X_b1.shape} sampel.")
print(f"   -> Brankas 2: {X_b2.shape} sampel.")

print("\n⏳ [STEP 4] Memecah Data menjadi Train / Val / Test (70-15-15)...")
def safe_stratified_split(X, yr, yc, test_size, rs=42):
    try: return train_test_split(X, yr, yc, test_size=test_size, random_state=rs, stratify=yc)
    except: return train_test_split(X, yr, yc, test_size=test_size, random_state=rs)

X_tr1, X_tmp1, yr_tr1, yr_tmp1, yc_tr1, yc_tmp1 = safe_stratified_split(X_b1, yr_b1, yc_b1, 0.30)
X_val1, X_ts1, yr_val1, yr_ts1, yc_val1, yc_ts1 = safe_stratified_split(X_tmp1, yr_tmp1, yc_tmp1, 0.50)

X_tr2, X_tmp2, yr_tr2, yr_tmp2, yc_tr2, yc_tmp2 = safe_stratified_split(X_b2, yr_b2, yc_b2, 0.30)
X_val2, X_ts2, yr_val2, yr_ts2, yc_val2, yc_ts2 = safe_stratified_split(X_tmp2, yr_tmp2, yc_tmp2, 0.50)

print("\n📐 [STEP 5] Fitting Master StandardScaler (Hanya pada Fitur Cuaca)...")
# PENTING: Jangan me-scaling One-Hot indikator!
def scale_4d(X, sc, fit=False):
    # X shape: [N_Samples, 14, 5, 18]
    # Kita hanya me-scaling fitur indeks 0 sampai 16 (17 fitur cuaca)
    n, w, s, f = X.shape
    weather_part = X[:, :, :, :-1].reshape(-1, f-1)
    
    if fit: sc.fit(weather_part)
    
    scaled_weather = sc.transform(weather_part).reshape(n, w, s, f-1)
    
    X_scaled = X.copy()
    X_scaled[:, :, :, :-1] = scaled_weather
    return X_scaled

scaler = StandardScaler()
# Gabungkan Train B1 dan B2 untuk fitting
X_tr_combined = np.concatenate([X_tr1, X_tr2], axis=0)
scale_4d(X_tr_combined, scaler, fit=True)
joblib.dump(scaler, os.path.join(TENSOR_ROOT, 'scaler_master_4d.pkl'))

X_tr1 = scale_4d(X_tr1, scaler); X_val1 = scale_4d(X_val1, scaler); X_ts1 = scale_4d(X_ts1, scaler)
X_tr2 = scale_4d(X_tr2, scaler); X_val2 = scale_4d(X_val2, scaler); X_ts2 = scale_4d(X_ts2, scaler)

print("\n⚖️ [STEP 6] Menjalankan SMOTETomek 4D (Flatten Trick)...")
def apply_smotetomek_4d(X, yr, yc, name):
    print(f"   Memproses {name}...")
    n_samples, n_window, n_stations, n_features = X.shape
    
    # GEPENGKAN TENSOR 4D MENJADI 2D UNTUK SMOTE
    # 14 * 5 * 18 = 1260 kolom
    X_flat = X.reshape(n_samples, -1)
    
    # [BUG FIX KRUSIAL] Gabungkan yr ke dalam X_flat agar SMOTE juga menginterpolasi
    # nilai curah hujan regresi (yr) secara bersamaan! Jika tidak, yr akan teracak
    # dan model regresi akan hancur (RMSE 106 mm).
    X_yr_flat = np.column_stack([X_flat, yr])
    
    unique, counts = np.unique(yc, return_counts=True)
    min_count = counts.min()
    k = min(5, min_count - 1)

    if k < 1:
        print(f"   ℹ️ Skip SMOTE: Sampel minoritas terlalu sedikit.")
        return X, yr, yc

    smt = SMOTETomek(random_state=42, smote=SMOTE(k_neighbors=k))
    X_yr_res_flat, yc_res = smt.fit_resample(X_yr_flat, yc)
    
    # Pisahkan kembali X_res dan yr_res
    X_res_flat = X_yr_res_flat[:, :-1]
    yr_res = X_yr_res_flat[:, -1].astype(np.float32)
    
    # RAJUT KEMBALI KE 4D TENSOR!
    X_res = X_res_flat.reshape(-1, n_window, n_stations, n_features)
    
    print(f"   ✅ Distribusi kelas baru: {np.bincount(yc_res)}")
    return X_res, yr_res, yc_res

X_tr1, yr_tr1, yc_tr1 = apply_smotetomek_4d(X_tr1, yr_tr1, yc_tr1, "BRANKAS 1 TRAIN")
X_tr2, yr_tr2, yc_tr2 = apply_smotetomek_4d(X_tr2, yr_tr2, yc_tr2, "BRANKAS 2 TRAIN")

print("\n💾 [STEP 7] Menyimpan Tensor PyTorch (Format 4D)...")
def save_pt(X, yr, yc, prefix):
    torch.save(torch.tensor(X, dtype=torch.float32), os.path.join(TENSOR_ROOT, f'{prefix}_X_4d.pt'))
    torch.save(torch.tensor(yr, dtype=torch.float32), os.path.join(TENSOR_ROOT, f'{prefix}_yr_4d.pt'))
    torch.save(torch.tensor(yc, dtype=torch.long), os.path.join(TENSOR_ROOT, f'{prefix}_yc_4d.pt'))

save_pt(X_tr1, yr_tr1, yc_tr1, 'b1_train'); save_pt(X_val1, yr_val1, yc_val1, 'b1_val'); save_pt(X_ts1, yr_ts1, yc_ts1, 'b1_test')
save_pt(X_tr2, yr_tr2, yc_tr2, 'b2_train'); save_pt(X_val2, yr_val2, yc_val2, 'b2_val'); save_pt(X_ts2, yr_ts2, yc_ts2, 'b2_test')
print("🎉 PHASE 5B SELESAI TOTAL! Tensor 4D siap untuk GNN Sejati!")
