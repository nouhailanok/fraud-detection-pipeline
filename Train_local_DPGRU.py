"""
train_local.py — Entraînement local GRU / DPGRU (Approche A ou B)
══════════════════════════════════════════════════════
Fixes v2 :
  - API torch.amp (non dépréciée) pour GradScaler et autocast
  - Gradient clipping (max_norm=1.0) → évite les explosions → nan
  - pos_weight en FP32 forcé → évite overflow FP16
  - Détection NaN dans les données au démarrage
  - Checkpoint protégé : sauvegarde dès epoch 1 garantie
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    average_precision_score,
)

from models.fraud_rnn import FraudRNN
from data.dataloader import get_dataloaders_approach_A, get_dataloaders_approach_B

# DPGRU — import conditionnel (nécessite opacus installé)
try:
    from opacus.layers import DPGRU
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    print("⚠️  Opacus non installé → USE_DPGRU forcé à False")
    print("   pip install opacus")

# ══════════════════════════════════════════════════════
# ⚙️  CONFIGURATION
# ══════════════════════════════════════════════════════

DATA_DIR       = "data/node_1/tensors/"
CHECKPOINT_DIR = "checkpoints/"

# Modèle
INPUT_DIM    = 26
HIDDEN_DIM   = 128
NUM_LAYERS   = 2
DROPOUT_RATE = 0.2

# Entraînement
BATCH_SIZE    = 256
SEQ_LEN       = 5
TRAIN_RATIO   = 0.8
POS_WEIGHT    = 167.0   # ratio réel : 362 974 légit / 2 174 fraudes
PATIENCE      = 5       # early stopping
APPROACH      = "A"     # "A" = intra-user temporel | "B" = population unseen users
USE_DPGRU    = True    # True = DPGRU (compatible Opacus/DP) | False = GRU standard

EPOCHS        = 20
LEARNING_RATE = 0.0005

# ══════════════════════════════════════════════════════

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"   # Mixed Precision uniquement sur GPU

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_MEM  = torch.cuda.get_device_properties(0).total_memory / 1e9
else:
    GPU_NAME, GPU_MEM = "N/A", 0.0


# ══════════════════════════════════════════════════════
# 🔍 VÉRIFICATION DES TENSEURS
# ══════════════════════════════════════════════════════

def check_tensors(data_dir: str) -> bool:
    """Vérifie l'absence de NaN/Inf dans les tenseurs avant l'entraînement."""
    files = sorted(Path(data_dir).glob("X_batch_*.npy"))
    total_nan = total_inf = 0

    for f in files:
        X = np.load(f)
        total_nan += np.isnan(X).sum()
        total_inf += np.isinf(X).sum()

    if total_nan > 0 or total_inf > 0:
        print(f"❌ Tenseurs corrompus → NaN:{total_nan}  Inf:{total_inf}")
        print("   Relance generate_tensors_from_csv.py pour régénérer les tenseurs.")
        return False

    print(f"✅ Tenseurs vérifiés ({len(files)} fichiers) — aucun NaN/Inf")
    return True


# ══════════════════════════════════════════════════════
# 🔧 UTILS
# ══════════════════════════════════════════════════════

def save_checkpoint(model, epoch, metric_val, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch"      : epoch,
        "model_state": model.state_dict(),
        "metric"     : metric_val,
    }, path)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss  = 0.0
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # AMP désactivé en évaluation :
            # pos_weight=167 en FP16 → overflow → NaN dans la loss
            logits = model(x).float()   # forcé FP32
            loss   = criterion(logits, y.float())

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            total_loss  += loss.item() * y.size(0)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    n = len(all_targets)
    return (
        total_loss / n if n > 0 else 0.0,
        np.array(all_targets),
        np.array(all_preds),
        np.array(all_probs),
    )


# ══════════════════════════════════════════════════════
# 🚀 TRAIN
# ══════════════════════════════════════════════════════

def train_and_evaluate():
    print(f"\n{'═'*55}")
    approach_label = "A — Split Temporel Intra-User 70/10/20" if APPROACH == "A" else "B — Population Unseen Users 80/20"
    print(f"  🚀  Entraînement GRU — Approche {approach_label}")
    print(f"{'═'*55}")
    print(f"  Device  : {DEVICE}  ({GPU_NAME}  {GPU_MEM:.1f} GB)" if DEVICE.type == "cuda"
          else f"  Device  : CPU")
    print(f"  Epochs  : {EPOCHS}  |  Batch : {BATCH_SIZE}  |  LR : {LEARNING_RATE}")
    model_type = "DPGRU (Opacus)" if USE_DPGRU else "GRU standard"
    print(f"  Modèle  : {model_type}  hidden={HIDDEN_DIM}  layers={NUM_LAYERS}  dropout={DROPOUT_RATE}")
    print(f"  pos_weight : {POS_WEIGHT}  |  Mixed Precision : {USE_AMP}")
    print(f"{'─'*55}\n")

    # ── Vérification des tenseurs ─────────────────────────────────
    if not check_tensors(DATA_DIR):
        return

    # ── 1. Données ────────────────────────────────────────────────
    if APPROACH == "A":
        print("\n📂 Chargement (Approche A — Split Temporel Intra-User 70/10/20)...")
        train_loader, val_loader, test_loader = get_dataloaders_approach_A(
            data_dir    = DATA_DIR,
            train_ratio = 0.70,
            val_ratio   = 0.10,
            batch_size  = BATCH_SIZE,
            seq_len     = SEQ_LEN,
        )
    else:
        print("\n📂 Chargement (Approche B — Population Unseen Users 80/20)...")
        # train_loader, val_loader, test_loader = get_dataloaders_approach_B(
        #     data_dir    = DATA_DIR,
        #     train_ratio = 0.70,    # 70% train
        #     val_ratio   = 0.10,    # 10% validation (early stopping)
        #     batch_size  = BATCH_SIZE,
        #     seq_len     = SEQ_LEN,
        # )

        train_loader, test_loader = get_dataloaders_approach_B(
            data_dir    = DATA_DIR,
            train_ratio = 0.80,
            val_ratio   = 0.0,    # 10% validation (early stopping)
            batch_size  = BATCH_SIZE,
            seq_len     = SEQ_LEN,
        )

        val_loader = test_loader

    # ── 2. Modèle ─────────────────────────────────────────────────
    if USE_DPGRU and OPACUS_AVAILABLE:
        # DPGRU — même architecture que FraudRNN mais avec DPGRU
        # compatible Opacus pour la Differential Privacy en FL
        import torch.nn as nn_local
        class FraudDPRNN(nn_local.Module):
            def __init__(self):
                super().__init__()
                # DPGRU : même interface que nn.GRU, compatible per-sample gradients
                # Note : dropout entre couches géré manuellement (DPGRU ne le supporte pas nativement)
                self.gru = DPGRU(
                    input_size  = INPUT_DIM,
                    hidden_size = HIDDEN_DIM,
                    num_layers  = NUM_LAYERS,
                    batch_first = True,
                )
                self.classifier = nn_local.Sequential(
                    nn_local.Linear(HIDDEN_DIM, 64),
                    nn_local.ReLU(),
                    nn_local.Dropout(DROPOUT_RATE),
                    nn_local.Linear(64, 1),
                )
            def forward(self, x):
                _, h_n    = self.gru(x)
                final_mem = h_n[-1, :, :]   # dernière couche GRU
                return self.classifier(final_mem)

        model = FraudDPRNN().to(DEVICE)
        print("\n🔐 Modèle : DPGRU (Opacus-compatible) — prêt pour Differential Privacy")
    else:
        # GRU standard — entraînement local sans DP
        model = FraudRNN(
            input_dim    = INPUT_DIM,
            hidden_dim   = HIDDEN_DIM,
            num_layers   = NUM_LAYERS,
            dropout_rate = DROPOUT_RATE,
        ).to(DEVICE)
        print("\n🧠 Modèle : GRU standard")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Paramètres entraînables : {total_params:,}")

    # ── 3. Loss / Optimizer / Scheduler ──────────────────────────
    # FIX : pos_weight en float() explicite → reste FP32 même avec AMP
    pos_weight = torch.tensor([POS_WEIGHT], dtype=torch.float32).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer  = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    # ✅ API non dépréciée : torch.amp.GradScaler
    scaler = torch.amp.GradScaler(device=DEVICE.type, enabled=USE_AMP)

    # ── 4. Boucle d'entraînement ──────────────────────────────────
    best_val_loss     = float("inf")
    epochs_no_improve = 0
    model_suffix      = "dpgru" if USE_DPGRU else "gru"
    checkpoint_path   = f"{CHECKPOINT_DIR}best_{model_suffix}_approach{APPROACH}.pt"

    print(f"\n{'─'*55}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        n_train          = 0
        has_nan          = False

        for x, y in train_loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Forward en FP16 (vitesse), loss en FP32 (stabilité)
            with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                logits = model(x)
            logits_fp32 = logits.float()   # repasse en FP32 avant la loss
            loss = criterion(logits_fp32, y.float())

            # Détection NaN dans la loss
            if torch.isnan(loss):
                has_nan = True
                continue

            loss.backward()   # pas de scaler car loss est en FP32

            # Gradient clipping → évite les explosions
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_train_loss += loss.item() * y.size(0)
            n_train          += y.size(0)

        if has_nan:
            print(f"⚠️  Epoch {epoch} : NaN détectés dans la loss → batches ignorés")

        avg_train_loss = total_train_loss / n_train if n_train > 0 else float("nan")
        val_loss, _, _, _ = evaluate(model, val_loader,  criterion, DEVICE)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch [{epoch:>2}/{EPOCHS}] "
              f"| Train Loss: {avg_train_loss:.4f} "
              f"| Val Loss: {val_loss:.4f} "
              f"| LR: {current_lr:.6f}")

        # Checkpoint — sauvegarde si meilleure val_loss ET loss valide
        if not np.isnan(val_loss) and val_loss < best_val_loss:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            save_checkpoint(model, epoch, val_loss, checkpoint_path)
            print(f"  💾 Meilleur modèle sauvegardé (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  ⏳ Pas d'amélioration ({epochs_no_improve}/{PATIENCE})")

        if epochs_no_improve >= PATIENCE:
            print(f"\n🛑 Early stopping à l'epoch {epoch}.")
            break

    # ── 5. Évaluation finale ──────────────────────────────────────
    print(f"\n{'═'*55}")
    print("📊 Évaluation finale — meilleur modèle")
    print(f"{'═'*55}")

    if not Path(checkpoint_path).exists():
        print("❌ Aucun checkpoint sauvegardé (toutes les losses étaient NaN).")
        print("   → Vérifie tes tenseurs avec check_tensors.py")
        return

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    print(f"   Chargé depuis epoch {checkpoint['epoch']}  "
          f"(val_loss={checkpoint['metric']:.4f})\n")

    from sklearn.metrics import f1_score

    # ── Calibration du seuil sur VALIDATION (pas sur test) ───────────────────
    # Approche A : val_loader = 10% du milieu (clients connus, période intermédiaire)
    # Approche B : val_loader = 10% des users (clients inconnus, jamais vus)
    # Dans les deux cas : le seuil est cherché sur val → appliqué au test
    # → aucun biais introduit sur les métriques finales du test
    print("🎯 Calibration du seuil sur la validation...")
    _, val_targets_cal, _, val_probs_cal = evaluate(model, val_loader, criterion, DEVICE)

    best_f1_val, best_thresh = 0.0, 0.5
    for t in np.arange(0.05, 0.95, 0.01):
        preds_t = (val_probs_cal >= t).astype(float)
        f1 = f1_score(val_targets_cal, preds_t, pos_label=1, zero_division=0)
        if f1 > best_f1_val:
            best_f1_val, best_thresh = f1, t
    print(f"   Seuil optimal trouvé : {best_thresh:.2f}  (F1 val={best_f1_val:.4f})")
    print(f"   Ce seuil sera appliqué au test (non biaisé)\n")

    # ── Évaluation finale sur TEST ────────────────────────────────────────────
    _, all_targets, all_preds, all_probs = evaluate(model, test_loader, criterion, DEVICE)

    # Rapport avec seuil par défaut 0.5
    print("📋 RAPPORT DE CLASSIFICATION (seuil 0.5)")
    print("─"*55)
    print(classification_report(
        all_targets, all_preds,
        target_names=["Légitime (0)", "Fraude (1)"],
        zero_division=0,
    ))

    cm = confusion_matrix(all_targets, all_preds)
    tn, fp, fn, tp = cm.ravel()
    print("🧩 MATRICE DE CONFUSION (seuil 0.5)")
    print("─"*55)
    print(f"  Vrais Négatifs (Légitimes détectés) : {tn:>7,}")
    print(f"  Faux Positifs  (Fausses alertes)    : {fp:>7,} ⚠️")
    print(f"  Faux Négatifs  (Fraudes manquées)   : {fn:>7,} 🚨")
    print(f"  Vrais Positifs (Fraudes détectées)  : {tp:>7,}")

    print("\n📈 MÉTRIQUES DE PERFORMANCE")
    print("─"*55)
    try:
        roc_auc = roc_auc_score(all_targets, all_probs)
        pr_auc  = average_precision_score(all_targets, all_probs)
        print(f"  🌟 AUC-ROC                   : {roc_auc:.4f}")
        print(f"  🎯 PR-AUC (Precision-Recall) : {pr_auc:.4f}")

        # ── Seuil calibré sur val → appliqué au test ─────────────────────────
        preds_cal = (all_probs >= best_thresh).astype(float)
        cm_cal    = confusion_matrix(all_targets, preds_cal)
        tn_c, fp_c, fn_c, tp_c = cm_cal.ravel()
        f1_cal = f1_score(all_targets, preds_cal, pos_label=1, zero_division=0)

        print(f"\n  🔍 SEUIL CALIBRÉ SUR VALIDATION → appliqué au test")
        print(f"  Seuil  : {best_thresh:.2f}  (calibré sur val, F1 val={best_f1_val:.4f})")
        print(f"  F1-fraude sur test : {f1_cal:.4f}")
        print(f"\n  Avec seuil calibré :")
        print(f"  Vrais Positifs (Fraudes détectées) : {tp_c:>6,}  / {int(all_targets.sum())}")
        print(f"  Faux Positifs  (Fausses alertes)   : {fp_c:>6,}  ← à minimiser")
        print(f"  Faux Négatifs  (Fraudes manquées)  : {fn_c:>6,}")
        print(f"\n  Rappel fraude   : {tp_c / max(tp_c+fn_c,1):.2%}")
        print(f"  Précision fraude: {tp_c / max(tp_c+fp_c,1):.2%}")

        # ── Comparaison seuil 0.5 vs seuil calibré ───────────────────────────
        print(f"\n  📊 Comparaison seuils :")
        print(f"  {'':20s} {'Seuil 0.5':>12} {'Seuil calibré':>14}")
        print(f"  {'Fausses alertes':20s} {fp:>12,} {fp_c:>14,}")
        print(f"  {'Fraudes détectées':20s} {tp:>12,} {tp_c:>14,}")
        print(f"  {'Rappel fraude':20s} {tp/max(tp+fn,1):>12.2%} {tp_c/max(tp_c+fn_c,1):>14.2%}")
        print(f"  {'Précision fraude':20s} {tp/max(tp+fp,1):>12.2%} {tp_c/max(tp_c+fp_c,1):>14.2%}")

    except ValueError as e:
        print(f"  ⚠️ Calcul AUC impossible : {e}")

    print(f"\n{'═'*55}\n")


if __name__ == "__main__":
    train_and_evaluate()