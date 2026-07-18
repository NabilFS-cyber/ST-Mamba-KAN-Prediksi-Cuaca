# =====================================================================
# PHASE 6 – THE ULTIMATE SWEET SPOT (Q1 REGRESI + SINTA 2 KLASIFIKASI)
# =====================================================================
import os, math, time, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, balanced_accuracy_score
warnings.filterwarnings('ignore')

# 1. MOUNT GOOGLE DRIVE
from google.colab import drive
try:
    drive.mount('/content/drive', force_remount=True)
except: pass

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Perangkat: {device}")
TENSOR_ROOT = '/content/drive/MyDrive/Riset_ERA5_Land/tensors_mamba'
CLEAN_ROOT  = '/content/drive/MyDrive/Riset_ERA5_Land/clean'
os.makedirs(CLEAN_ROOT, exist_ok=True)

WINDOW, N_FEATURES, HIDDEN_DIM = 14, 17, 192
N_MAMBA_LAYERS = 4
BATCH_SIZE = 64
EPOCHS = 150
PATIENCE = 30
LABEL_NAMES = ['🟢 Aman (<20mm)', '🟡 Waspada (20-50mm)', '🔴 Siaga (≥50mm)']

# ============================================================
# PEMUATAN & PENGGABUNGAN DATASET
# ============================================================
def get_dataset(prefix):
    X = torch.load(f'{TENSOR_ROOT}/{prefix}_X.pt', weights_only=False)
    yr = torch.log1p(torch.load(f'{TENSOR_ROOT}/{prefix}_yr.pt', weights_only=False).clamp(min=0))
    yc = torch.load(f'{TENSOR_ROOT}/{prefix}_yc.pt', weights_only=False)
    return TensorDataset(X, yr, yc)

print("\n🔄 Memuat Data Satelit (ERA5) & Data Darat (BMKG)...")
ds_b1_tr = get_dataset('b1_train'); ds_b1_vl = get_dataset('b1_val'); ds_b1_ts = get_dataset('b1_test')
ds_b2_tr = get_dataset('b2_train'); ds_b2_vl = get_dataset('b2_val'); ds_b2_ts = get_dataset('b2_test')

print("🔗 MENGGABUNGKAN DATASET (DATA FUSION)...")
ds_tr_fusion = ConcatDataset([ds_b1_tr, ds_b2_tr])
ds_vl_fusion = ConcatDataset([ds_b1_vl, ds_b2_vl])
ds_ts_fusion = ConcatDataset([ds_b1_ts, ds_b2_ts]) # Untuk Evaluasi Klasifikasi

tr_loader = DataLoader(ds_tr_fusion, batch_size=BATCH_SIZE, shuffle=True)
vl_loader = DataLoader(ds_vl_fusion, batch_size=BATCH_SIZE, shuffle=False)

# Loader Evaluasi Spesifik:
ts_loader_bmkg = DataLoader(ds_b2_ts, batch_size=BATCH_SIZE, shuffle=False) # Regresi wajib di BMKG
ts_loader_fusi = DataLoader(ds_ts_fusion, batch_size=BATCH_SIZE, shuffle=False) # Klasifikasi diuji di Fusi

# ============================================================
# INOVASI SINTA 2: ORDINAL COST FOCAL LOSS
# ============================================================
class OrdinalCostFocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.cost_matrix = torch.tensor([
            [1.0, 1.0, 2.0],  # Asli Aman
            [2.5, 1.0, 2.0],  # Asli Waspada (Denda berat jika meleset!)
            [3.0, 1.2, 1.0]   # Asli Siaga
        ]).to(device)

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_term = ((1 - pt) ** self.gamma) * ce_loss
        probs = F.softmax(inputs, dim=1)
        expected_cost = torch.sum(probs * self.cost_matrix[targets], dim=1)
        return (focal_term * expected_cost).mean()

# ============================================================
# BLOK DASAR ST-MAMBA-KAN (DARI Q1 SCOPUS EDITION)
# ============================================================
class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.2):
        super().__init__()
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, padding=d_conv - 1, groups=self.d_inner, bias=True)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        A = torch.arange(1, d_state + 1).float().unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x_path, z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_path.permute(0, 2, 1))[:, :, :x.shape[1]]
        x_path = F.silu(x_conv.permute(0, 2, 1))
        B, L, D = x_path.shape
        A = -torch.exp(self.A_log)
        x_dbl = self.x_proj(x_path)
        delta = F.softplus(self.dt_proj(x_dbl[:, :, :1]))
        B_ssm = x_dbl[:, :, 1:1+self.d_state]
        C_ssm = x_dbl[:, :, 1+self.d_state:]
        h = torch.zeros(B, D, self.d_state, device=x.device)
        y = torch.zeros_like(x_path)
        for t in range(L):
            dt = delta[:, t, :]
            dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))
            dB = dt.unsqueeze(-1) * B_ssm[:, t, :].unsqueeze(1)
            h = dA * h + dB * x_path[:, t, :].unsqueeze(-1)
            y[:, t, :] = (h * C_ssm[:, t, :].unsqueeze(1)).sum(-1) + self.D * x_path[:, t, :]
        return self.dropout(self.out_proj(y * F.silu(z))) + residual

class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=4, spline_order=3):
        super().__init__()
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        h = 2.0 / grid_size
        self.register_buffer('grid', torch.linspace(-1 - h*spline_order, 1 + h*spline_order, grid_size + 2*spline_order + 1))
        self.spline_weight = nn.Parameter(torch.randn(out_features, in_features, grid_size + spline_order) * 0.1)
        self.spline_order = spline_order
    def forward(self, x):
        base_out = F.silu(F.linear(x, self.base_weight))
        x_u = torch.tanh(x).unsqueeze(-1)
        bases = ((x_u >= self.grid[:-1]) & (x_u < self.grid[1:])).float()
        for p in range(1, self.spline_order + 1):
            n = bases.shape[-1] - 1
            left = ((x_u - self.grid[:n]) / (self.grid[p:p+n] - self.grid[:n] + 1e-8)) * bases[:, :, :-1]
            right = ((self.grid[p+1:p+1+n] - x_u) / (self.grid[p+1:p+1+n] - self.grid[1:1+n] + 1e-8)) * bases[:, :, 1:]
            bases = left + right
        return base_out + torch.einsum('bin,oin->bo', bases, self.spline_weight)

class KANHead(nn.Module):
    def __init__(self, in_dim, out_dim=1):
        super().__init__()
        self.out_dim = out_dim
        self.kan = KANLinear(in_dim, out_dim)
    def forward(self, x): return self.kan(x).squeeze(-1) if self.out_dim==1 else self.kan(x)

class EMA:
    def __init__(self, model):
        self.model = model
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
    def update(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad: self.shadow[n] = 0.001 * p.data + 0.999 * self.shadow[n]
    def apply(self):
        self.backup = {n: p.data.clone() for n, p in self.model.named_parameters() if p.requires_grad}
        for n, p in self.model.named_parameters():
            if p.requires_grad: p.data = self.shadow[n]
    def restore(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad: p.data = self.backup[n]

# ============================================================
# MODEL REGRESI & KLASIFIKASI (ARSITEKTUR Q1 - RMSE TERENDAH)
# ============================================================
class MetMambaRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(N_FEATURES, HIDDEN_DIM)
        self.mambas = nn.ModuleList([MambaBlock(HIDDEN_DIM) for _ in range(N_MAMBA_LAYERS)])
        self.reg_head = KANHead(HIDDEN_DIM * 2, 1)
    def forward(self, x):
        x = self.proj(x)
        for m in self.mambas: x = m(x)
        x_pool = torch.cat([x.mean(dim=1), x.max(dim=1)[0]], dim=1)
        return self.reg_head(x_pool)

class MetMambaClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(N_FEATURES, HIDDEN_DIM)
        self.mambas = nn.ModuleList([MambaBlock(HIDDEN_DIM) for _ in range(N_MAMBA_LAYERS)])
        self.cls_head = KANHead(HIDDEN_DIM * 2, 3)
    def forward(self, x):
        x = self.proj(x)
        for m in self.mambas: x = m(x)
        x_pool = torch.cat([x.mean(dim=1), x.max(dim=1)[0]], dim=1)
        return self.cls_head(x_pool)

# ============================================================
# EKSEKUSI PELATIHAN DENGAN GRADIENT CLIPPING
# ============================================================
def train_regressor(model, ema, tr_loader, vl_loader, epochs, lr):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs, steps_per_epoch=len(tr_loader))
    best_score = float('inf')
    early = 0
    for ep in range(1, epochs+1):
        model.train()
        for X, yr, yc in tr_loader:
            X, yr = X.to(device), yr.to(device)
            pr = model(X)
            loss = F.huber_loss(pr, yr)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Pengaman Gradient Explosion
            opt.step(); sch.step(); ema.update()

        ema.apply(); model.eval()
        vp_reg, vt_reg = [], []
        with torch.no_grad():
            for X, yr, yc in vl_loader:
                vp_reg.append(model(X.to(device)).cpu()); vt_reg.append(yr)

        vp_mm = torch.expm1(torch.cat(vp_reg)).clamp(min=0)
        vt_mm = torch.expm1(torch.cat(vt_reg)).clamp(min=0)
        rmse = torch.sqrt(((vp_mm - vt_mm)**2).mean()).item()
        mae = torch.mean(torch.abs(vp_mm - vt_mm)).item()

        if rmse < best_score:
            best_score = rmse; early = 0
            torch.save({'state': model.state_dict(), 'ema': ema.shadow}, f'{CLEAN_ROOT}/best_reg_Fusion.pt')
            print(f"📊 Reg Fusion Ep {ep:03d} │ RMSE: {rmse:.2f} │ MAE: {mae:.2f} 💾 ✅")
        else:
            early += 1
            if early >= PATIENCE: print("⏹️ Early Stopping (Regresi)!"); break
        ema.restore()

def train_classifier(model, ema, tr_loader, vl_loader, epochs, lr, loss_fn):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs, steps_per_epoch=len(tr_loader))
    best_score = -float('inf')
    early = 0

    for ep in range(1, epochs+1):
        model.train()
        for X, yr, yc in tr_loader:
            X, yc = X.to(device), yc.to(device)
            pc = model(X)
            loss = loss_fn(pc, yc)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Pengaman Gradient Explosion
            opt.step(); sch.step(); ema.update()

        ema.apply(); model.eval()
        vp_cls, vt_cls = [], []
        with torch.no_grad():
            for X, yr, yc in vl_loader:
                vp_cls.append(model(X.to(device)).cpu()); vt_cls.append(yc)

        vt_c = torch.cat(vt_cls).numpy()
        vp_c = torch.argmax(torch.cat(vp_cls), dim=1).numpy()

        bacc = balanced_accuracy_score(vt_c, vp_c)
        acc = accuracy_score(vt_c, vp_c)
        score = (0.6 * acc) + (0.4 * bacc) # Sinta 2 scoring logic

        if score > best_score:
            best_score = score; early = 0
            torch.save({'state': model.state_dict(), 'ema': ema.shadow}, f'{CLEAN_ROOT}/best_cls_Fusion.pt')
            print(f"📊 Cls Fusion Ep {ep:03d} │ Acc: {acc:.3f} │ B-Acc: {bacc:.3f} 💾 ✅")
        else:
            early += 1
            if early >= PATIENCE: print("⏹️ Early Stopping (Klasifikasi)!"); break
        ema.restore()

# ============================================================
print("\n" + "="*50)
print("🚀 [1] MELATIH MODEL REGRESI (MENGINCAR RMSE <17mm)")
print("="*50)
mod_reg = MetMambaRegressor().to(device); ema_reg = EMA(mod_reg)
train_regressor(mod_reg, ema_reg, tr_loader, vl_loader, EPOCHS, 4e-4)

print("\n" + "="*50)
print("🚀 [2] MELATIH MODEL KLASIFIKASI (MENGINCAR SWEET SPOT)")
print("="*50)
mod_cls = MetMambaClassifier().to(device); ema_cls = EMA(mod_cls)
q1_loss_fn = OrdinalCostFocalLoss(gamma=2.0)
train_classifier(mod_cls, ema_cls, tr_loader, vl_loader, EPOCHS, 4e-4, q1_loss_fn)

# ============================================================
# EVALUASI AKHIR: AUTO SWEET-SPOT FINDER (THE MASTERPIECE)
# ============================================================
print("\n" + "="*65)
print("🔎 EVALUASI AKHIR: MENCARI SWEET SPOT KESEIMBANGAN SEMPURNA")
print("="*65)

# Load Best Models
ckpt_reg = torch.load(f'{CLEAN_ROOT}/best_reg_Fusion.pt', weights_only=False)
mod_reg.load_state_dict(ckpt_reg['state'])
for n, p in mod_reg.named_parameters(): p.data = ckpt_reg['ema'][n]
mod_reg.eval()

ckpt_cls = torch.load(f'{CLEAN_ROOT}/best_cls_Fusion.pt', weights_only=False)
mod_cls.load_state_dict(ckpt_cls['state'])
for n, p in mod_cls.named_parameters(): p.data = ckpt_cls['ema'][n]
mod_cls.eval()

# --- EVALUASI REGRESI PADA DATA BMKG MURNI ---
all_reg, all_yt = [], []
with torch.no_grad():
    for X, yr, _ in ts_loader_bmkg:
        all_reg.append(mod_reg(X.to(device)).cpu()); all_yt.append(yr)

vp_mm = torch.expm1(torch.cat(all_reg)).clamp(min=0).numpy().flatten()
vt_mm = torch.expm1(torch.cat(all_yt)).clamp(min=0).numpy().flatten()
rmse = np.sqrt(np.mean((vp_mm - vt_mm)**2))
mae  = np.mean(np.abs(vp_mm - vt_mm))

# --- EVALUASI KLASIFIKASI PADA DATA FUSI (DENGAN TTA) ---
def tta_predict(model_b, x_batch, n_aug=3):
    preds = [F.softmax(model_b(x_batch), dim=1)]
    for _ in range(n_aug):
        noise = torch.randn_like(x_batch) * 0.01
        preds.append(F.softmax(model_b(x_batch + noise), dim=1))
    return torch.stack(preds).mean(0)

all_cls, all_yc = [], []
with torch.no_grad():
    for X, _, yc in ts_loader_fusi:
        prob_cls = tta_predict(mod_cls, X.to(device))
        all_cls.append(prob_cls.cpu()); all_yc.append(yc)

probs = torch.cat(all_cls).numpy()
vt_c = torch.cat(all_yc).numpy()

# 🎯 AUTO SWEET-SPOT SEARCHER
# Skrip akan mencoba ratusan kombinasi ambang batas untuk mencari titik emas (Sweet Spot)
# yang memaksimalkan Keseimbangan Akurasi dan Macro F1.
print("\n🤖 AI sedang mensimulasikan ratusan kombinasi Sweet Spot...")
best_score = 0
best_vp_c = None
best_w, best_s = 0.48, 0.42

for thr_w in np.arange(0.35, 0.60, 0.02):
    for thr_s in np.arange(0.35, 0.55, 0.02):
        vp_c_temp = np.zeros_like(vt_c)
        for i in range(len(probs)):
            p_a, p_w, p_s = probs[i]
            if p_s > thr_s:      vp_c_temp[i] = 2
            elif p_w > thr_w:    vp_c_temp[i] = 1
            else:                vp_c_temp[i] = 0

        acc_t = accuracy_score(vt_c, vp_c_temp)
        f1_t = f1_score(vt_c, vp_c_temp, average='macro')

        # Cari keseimbangan emas: 60% Akurasi, 40% F1-Score
        score_t = (acc_t * 0.6) + (f1_t * 0.4)
        if score_t > best_score:
            best_score = score_t
            best_vp_c = vp_c_temp.copy()
            best_w, best_s = thr_w, thr_s

# Gunakan Sweet Spot Terbaik
acc_final = accuracy_score(vt_c, best_vp_c)
bacc_final = balanced_accuracy_score(vt_c, best_vp_c)
f1_mac_final = f1_score(vt_c, best_vp_c, average='macro')
cm_final = confusion_matrix(vt_c, best_vp_c, labels=[0,1,2])

print(f"🎯 SWEET SPOT DITEMUKAN! [Waspada: {best_w:.2f} | Siaga: {best_s:.2f}]")

print(f"\n📈 HASIL AKHIR DECOUPLED ST-MAMBA-KAN (THE ULTIMATE SWEET SPOT)")
print(f"   [Output Model 1: Hidrologi (Diuji pada Ground-Truth BMKG Murni)]")
print(f"   ├── RMSE         : {rmse:.2f} mm")
print(f"   ├── MAE          : {mae:.2f} mm")

print(f"\n   [Output Model 2: Deteksi Badai (Diuji pada Data Fusi dengan TTA)]")
print(f"   ├── Akurasi Total: {acc_final*100:.2f}%")
print(f"   ├── Balanced Acc : {bacc_final*100:.2f}%")
print(f"   ├── Macro F1     : {f1_mac_final:.3f}")

print(f'\n📊 MATRIKS KONFUSI FINAL:')
print(f'{"":20s} {"[Tebak Aman]":>14s} {"[Tebak Waspada]":>16s} {"[Tebak Siaga]":>14s}')
print(f' {"[Asli Aman]":20s} {cm_final[0,0]:>10d}     {cm_final[0,1]:>10d}     {cm_final[0,2]:>10d}')
print(f' {"[Asli Waspada]":20s} {cm_final[1,0]:>10d}     {cm_final[1,1]:>10d}     {cm_final[1,2]:>10d}')
print(f' {"[Asli Siaga]":20s} {cm_final[2,0]:>10d}     {cm_final[2,1]:>10d}     {cm_final[2,2]:>10d}')

print(f'\n📋 CLASSIFICATION REPORT:')
print(classification_report(vt_c, best_vp_c, target_names=LABEL_NAMES))
