"""
train_central.py — Entraînement centralisé (sans FL) pour comparaison
══════════════════════════════════════════════════════════════════════
Objectif : baseline "sans FL" — entraîne le même modèle DPGRU 3 couches
           sur TOUTES les données combinées des 4 nœuds.

Stratégie :
  1. Copie les tenseurs des 4 nodes dans data/central/tensors/
  2. Utilise get_dataloaders_approach_B (même dataloader que FL)
  3. Entraîne avec les mêmes hyperparamètres que FL
  4. Sauvegarde les résultats dans logs/central/

Comparaison équitable FL vs Centralisé :
  - Architecture       : GRU standard 3 couches, hidden=128, dropout=0.3 (sans DP)
  - Même approche      : B (split par population d'utilisateurs)
  - Même hyperparams   : LR, batch, epochs, pos_weight
  - Même dataloader    : get_dataloaders_approach_B
  - Différence unique  : centralisé sans DP vs fédéré avec FL+DP
"""



import json
import time
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, average_precision_score, f1_score,
)
import sys
sys.path.insert(0, str(Path(__file__).parent))

from models.fraud_rnn import build_model
from data.dataloader import get_dataloaders_approach_B

# ══════════════════════════════════════════════════════
# ⚙️  CONFIGURATION — identique au FL pour comparaison
# ══════════════════════════════════════════════════════

print("Démarrage...")
print(f"CWD : {Path.cwd()}")

NODE_DIRS = [
    "data/node_1/tensors",
    "data/node_2/tensors",
    "data/node_3/tensors",
    "data/node_4/tensors",
]

CENTRAL_DIR    = Path("data/central/tensors")      # dossier combiné temporaire (train)
TEST_DIR       = Path("data/tensors_FraudTest")    # tenseurs FraudTest.csv (test externe)
LOGS_DIR       = Path("logs/central")
CHECKPOINT_DIR = Path("checkpoints/central")

# Hyperparamètres — identiques au FL
SEQ_LEN       = 5
BATCH_SIZE    = 256
EPOCHS        = 20
LEARNING_RATE = 0.0005
# TRAIN_RATIO : sépare train et validation dans CENTRAL_DIR
# (le test vient de data/tensors_FraudTest/ — dossier séparé)
# 90% des users centraux → train | 10% → validation (early stopping)
TRAIN_RATIO   = 0.90
POS_WEIGHT    = 1.0
PATIENCE      = 5

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42) 

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"




# ══════════════════════════════════════════════════════
# 📂 COMBINAISON DES TENSEURS
# ══════════════════════════════════════════════════════

def combine_tensors(node_dirs: list, out_dir: Path) -> bool:
    """
    Copie les tenseurs X_batch_*.npy / y_batch_*.npy de tous les nodes
    dans un dossier central unique, en renommant pour éviter les conflits.

    Le dataloader existant (get_dataloaders_approach_B) peut ensuite
    les lire comme si c'était un seul nœud centralisé.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Nettoyer l'ancien contenu si existant
    for f in out_dir.glob("*.npy"):
        f.unlink()

    total_x = total_y = 0

    for node_id, node_dir in enumerate(node_dirs, 1):
        path = Path(node_dir)
        x_files = sorted(path.glob("X_batch_*.npy"))
        y_files = sorted(path.glob("y_batch_*.npy"))

        if not x_files:
            print(f"  ⚠️  Aucun tenseur dans {node_dir}")
            continue

        n_node = 0
        for x_file, y_file in zip(x_files, y_files):
            # Renommer pour éviter les conflits entre nodes
            stem   = x_file.stem.replace("X_batch_", f"X_batch_n{node_id}_")
            y_stem = y_file.stem.replace("y_batch_", f"y_batch_n{node_id}_")
            shutil.copy2(x_file, out_dir / f"{stem}.npy")
            shutil.copy2(y_file, out_dir / f"{y_stem}.npy")
            n_node += 1

        X_node = np.concatenate([np.load(f) for f in x_files], axis=0)
        y_node = np.concatenate([np.load(f) for f in y_files], axis=0)
        print(f"  ✅ Node {node_id} : {len(X_node):,} séquences | "
              f"fraudes={int(y_node.sum())} ({y_node.mean()*100:.2f}%)")
        total_x += len(X_node)
        total_y += int(y_node.sum())

    print(f"\n  Total combiné : {total_x:,} séquences | fraudes={total_y:,}")
    return total_x > 0


# ══════════════════════════════════════════════════════
# 📊 ÉVALUATION
# ══════════════════════════════════════════════════════

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x, y    = x.to(device), y.to(device)
            logits   = model(x).float()
            loss     = criterion(logits, y.float())
            probs    = torch.sigmoid(logits)
            preds    = (probs >= 0.5).float()
            total_loss  += loss.item() * y.size(0)
            all_probs.extend(probs.cpu().numpy().flatten())
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(y.cpu().numpy().flatten())

    n = len(all_targets)
    return (
        total_loss / n if n > 0 else 0.0,
        np.array(all_targets),
        np.array(all_preds),
        np.array(all_probs),
    )


# ══════════════════════════════════════════════════════
# 🚀 MAIN
# ══════════════════════════════════════════════════════

def train_central():
    run_id   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"central_{run_id}"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═'*60}")
    print(f"  🏛️  ENTRAÎNEMENT CENTRALISÉ (sans FL) — {run_id}")
    print(f"{'═'*60}")
    print(f"  Device     : {DEVICE}")
    print(f"  Modèle     : GRU standard 3 couches | hidden=128 | dropout=0.3 (sans DP)")
    print(f"  Dataloader : get_dataloaders_approach_B (identique au FL)")
    print(f"  Epochs     : {EPOCHS} | Batch : {BATCH_SIZE} | LR : {LEARNING_RATE}")
    print(f"  Nodes      : {len(NODE_DIRS)} combinés dans {CENTRAL_DIR}")
    print(f"{'─'*60}\n")

    # ── 1. Combiner les tenseurs ──────────────────────────────────
    print("📂 Combinaison des tenseurs...")
    if not combine_tensors(NODE_DIRS, CENTRAL_DIR):
        print("❌ Aucun tenseur trouvé. Vérifier NODE_DIRS.")
        return

    # ── 2. Dataloader train+val — depuis les tenseurs centraux ──
    # 90% users → train | 10% users → val (early stopping)
    # Le test vient entièrement de data/tensors_FraudTest/
    print(f"\n📊 Chargement train/val depuis {CENTRAL_DIR}...")
    print(f"  Split : {int(TRAIN_RATIO*100)}% users train / {int((1-TRAIN_RATIO)*100)}% users val")
    train_loader, val_loader = get_dataloaders_approach_B(
        data_dir    = CENTRAL_DIR,
        train_ratio = TRAIN_RATIO,
        batch_size  = BATCH_SIZE,
        seq_len     = SEQ_LEN,
        # random_seed = 42,
    )

    # ── Dataloader test — depuis data/tensors_FraudTest/ ─────────
    if TEST_DIR.exists():
        print(f"\n📊 Chargement test depuis {TEST_DIR}...")
        from data.dataloader import FraudSequenceDataset
        from torch.utils.data import DataLoader as _DL

        # Calculer scaler depuis tenseurs train centraux
        print(f"  📐 Calcul scaler depuis {CENTRAL_DIR}...")
        x_files_c     = sorted(CENTRAL_DIR.glob("X_batch_*.npy"))
        X_central_all = np.concatenate(
            [np.load(f)[:, 1:] for f in x_files_c], axis=0
        ).astype(np.float32)

        median_c = np.median(X_central_all, axis=0)
        q75, q25 = np.percentile(X_central_all, [75, 25], axis=0)
        iqr_c    = np.where((q75 - q25) == 0, 1.0, q75 - q25)
        del X_central_all

        # Charger FraudTest + appliquer même scaler
        test_ds          = FraudSequenceDataset(TEST_DIR, sequence_length=SEQ_LEN)
        test_ds.features = (test_ds.features - median_c) / iqr_c
        test_ds.features = np.clip(test_ds.features, -5.0, 5.0).astype(np.float32)
        print(f"  📐 FraudTest normalisé : min={test_ds.features.min():.3f}  max={test_ds.features.max():.3f}")

        test_loader  = _DL(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=True)
        n_fraud_test = int(test_ds.y.sum())
        print(f"  ✅ {len(test_ds):,} séquences | fraudes={n_fraud_test:,}")
    else:
        print(f"\n  ⚠️  {TEST_DIR} introuvable → test sur val_loader")
        test_loader = val_loader

    # ── 3. Modèle GRU standard (sans DP) ────────────────────────
    # use_dpgru=False → GRU standard avec dropout inter-couches (0.3)
    # + post_gru_dropout (0.3) + classifier dropout (0.3)
    # 3 points de dropout — MIEUX que DPGRU qui perd le dropout inter-couches
    print(f"\n🧠 Construction du modèle GRU standard (sans DP)...")
    model = build_model(use_dpgru=False).to(DEVICE)
    print(f"  {model.info()}")

    # ── 4. Loss / Optimizer / Scheduler ──────────────────────────
    pos_weight = torch.tensor([POS_WEIGHT], dtype=torch.float32).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    # ── 5. Entraînement ───────────────────────────────────────────
    best_val_loss     = float("inf")
    best_val_f1       = 0.0
    epochs_no_improve = 0
    checkpoint_path   = CHECKPOINT_DIR / f"{run_name}_best.pt"
    metrics_log       = []

    print(f"\n{'─'*60}")
    print("  🚀 Entraînement...\n")
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, n_samples = 0.0, 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                logits = model(x)
            loss = criterion(logits.float(), y.float())

            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            n_samples  += y.size(0)

        avg_train_loss = total_loss / n_samples if n_samples > 0 else float("nan")
        val_loss, val_targets, val_preds, val_probs = evaluate(
            model, val_loader, criterion, DEVICE
        )
        val_recall = float(np.sum((val_targets == 1) & (val_preds == 1))) / max(float(np.sum(val_targets == 1)), 1)
        val_f1     = f1_score(val_targets, val_preds, pos_label=1, zero_division=0)

        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        print(f"  Epoch [{epoch:>2}/{EPOCHS}]"
              f" | Train={avg_train_loss:.4f}"
              f" | Val={val_loss:.4f}"
              f" | Recall={val_recall:.3f}"
              f" | F1={val_f1:.3f}"
              f" | LR={lr:.6f}"
              f" | {time.time()-t_start:.0f}s")

        metrics_log.append({
            "epoch"       : epoch,
            "train_loss"  : round(avg_train_loss, 4),
            "val_loss"    : round(val_loss, 4),
            "val_recall"  : round(val_recall, 4),
            "val_f1"      : round(val_f1, 4),
            "lr"          : lr,
        })

        # Early stopping basé sur F1 (comme le serveur FL)
        if val_f1 > best_val_f1:
            best_val_f1       = val_f1
            best_val_loss     = val_loss
            epochs_no_improve = 0
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "val_loss"   : val_loss,
                "val_f1"     : val_f1,
            }, checkpoint_path)
            print(f"    💾 Meilleur modèle sauvegardé (F1={val_f1:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\n  🛑 Early stopping epoch {epoch}")
                break

    # ── 6. Évaluation finale ──────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  📊 ÉVALUATION FINALE")
    print(f"{'═'*60}")

    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Checkpoint epoch {ckpt['epoch']} | F1={ckpt['val_f1']:.4f}\n")

    _, all_targets, all_preds, all_probs = evaluate(
        model, test_loader, criterion, DEVICE
    )

    # Calibration seuil sur les prédictions
    best_f1_cal, best_thresh = 0.0, 0.5
    for t in np.arange(0.05, 0.95, 0.01):
        preds_t = (all_probs >= t).astype(float)
        f1 = f1_score(all_targets, preds_t, pos_label=1, zero_division=0)
        if f1 > best_f1_cal:
            best_f1_cal, best_thresh = f1, t

    preds_cal = (all_probs >= best_thresh).astype(float)
    cm        = confusion_matrix(all_targets, preds_cal)
    tn, fp, fn, tp = cm.ravel()
    recall    = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    f1_final  = f1_score(all_targets, preds_cal, pos_label=1, zero_division=0)
    roc_auc   = roc_auc_score(all_targets, all_probs)
    pr_auc    = average_precision_score(all_targets, all_probs)

    print(f"  {'Métrique':<25} {'Centralisé':>12}  {'FL (meilleur run)':>18}")
    print(f"  {'─'*57}")
    print(f"  {'Recall (fraude)':<25} {recall:>12.4f}  {'0.6196':>18}")
    print(f"  {'Precision':<25} {precision:>12.4f}  {'~0.78':>18}")
    print(f"  {'F1 (fraude)':<25} {f1_final:>12.4f}  {'0.7041':>18}")
    print(f"  {'AUC-ROC':<25} {roc_auc:>12.4f}  {'—':>18}")
    print(f"  {'PR-AUC':<25} {pr_auc:>12.4f}  {'—':>18}")
    print(f"  {'Fraudes détectées':<25} {int(tp):>9,}/{int(tp+fn):,}  {'':>18}")
    print(f"  {'Fausses alertes':<25} {int(fp):>12,}  {'':>18}")
    print(f"  {'Seuil':<25} {best_thresh:>12.2f}  {'':>18}")

    # ── 7. Sauvegarder les résultats ──────────────────────────────
    results = {
        "run_id"          : run_name,
        "mode"            : "centralisé (sans FL, sans DP)",
        "timestamp"       : run_id,
        "architecture"    : "GRU standard 3 couches hidden=128 dropout=0.3 (sans DP)",
        "dataloader"      : "get_dataloaders_approach_B",
        "approach"        : "B",
        "n_nodes_combined": len(NODE_DIRS),
        "threshold"       : best_thresh,
        "recall"          : round(recall,    4),
        "precision"       : round(precision, 4),
        "f1"              : round(f1_final,  4),
        "roc_auc"         : round(roc_auc,   4),
        "pr_auc"          : round(pr_auc,    4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "metrics_per_epoch": metrics_log,
    }

    out_json = LOGS_DIR / f"{run_name}_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n  💾 Résultats → {out_json}")
    print(f"  💾 Checkpoint → {checkpoint_path}")
    print(f"\n{'═'*60}")
    print("  ✅ Entraînement centralisé terminé")
    print("  Compare avec : logs/runs/*/fl_metrics.json")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    train_central()