"""
prepare_csv.py — Prépare FraudTest.csv (ou tout CSV fraude) pour generate_tensors_from_csv.py
══════════════════════════════════════════════════════════════════════════════════════════════
Ce script :
  1. Lit le CSV brut (FraudTrain.csv ou FraudTest.csv)
  2. Trie par utilisateur (cc_num) puis par temps (unix_time)
  3. Supprime les doublons éventuels
  4. Regroupe proprement les transactions par user
  5. Sauvegarde le CSV préparé prêt pour generate_tensors_from_csv.py

Usage :
  python prepare_csv.py --input data/FraudTest.csv --output data/FraudTest_sorted.csv
  python prepare_csv.py --input data/FraudTrain.csv --output data/FraudTrain_sorted.csv
"""

import argparse
import pandas as pd
from pathlib import Path


def prepare_csv(input_path: str, output_path: str) -> None:

    input_path  = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        print(f"❌ Fichier introuvable : {input_path}")
        return

    print(f"\n{'═'*55}")
    print(f"  📂 Lecture de {input_path.name}...")
    print(f"{'═'*55}")

    df = pd.read_csv(input_path)

    # Supprimer colonne index si elle existe (Unnamed: 0)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    print(f"  Shape initiale    : {df.shape}")
    print(f"  Colonnes          : {list(df.columns)}")
    print(f"  Utilisateurs      : {df['cc_num'].nunique():,}")
    print(f"  Transactions      : {len(df):,}")
    print(f"  Fraudes           : {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.2f}%)")

    # ── 1. Vérifier les colonnes obligatoires ────────────────────
    required = ["cc_num", "unix_time", "amt", "is_fraud"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        print(f"\n❌ Colonnes manquantes : {missing}")
        return

    # ── 2. Convertir les types ───────────────────────────────────
    df["cc_num"]    = df["cc_num"].astype(str).str.strip()
    df["unix_time"] = pd.to_numeric(df["unix_time"], errors="coerce")
    df["amt"]       = pd.to_numeric(df["amt"],       errors="coerce")
    df["is_fraud"]  = pd.to_numeric(df["is_fraud"],  errors="coerce").fillna(0).astype(int)

    # ── 3. Supprimer les lignes avec valeurs critiques manquantes ─
    n_before = len(df)
    df = df.dropna(subset=["cc_num", "unix_time", "amt"])
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"\n  ⚠️  {n_dropped} lignes supprimées (valeurs manquantes)")

    # ── 4. Supprimer les doublons ─────────────────────────────────
    n_before = len(df)
    df = df.drop_duplicates(subset=["cc_num", "unix_time"], keep="first")
    n_dup = n_before - len(df)
    if n_dup > 0:
        print(f"  ⚠️  {n_dup} doublons supprimés")

    # ── 5. TRI : par user puis par temps ─────────────────────────
    print(f"\n  🔀 Tri par cc_num → unix_time...")
    df = df.sort_values(["cc_num", "unix_time"], ascending=[True, True])
    df = df.reset_index(drop=True)

    # ── 6. Rapport sur le résultat ───────────────────────────────
    print(f"\n{'─'*55}")
    print(f"  ✅ CSV préparé :")
    print(f"     Transactions      : {len(df):,}")
    print(f"     Utilisateurs      : {df['cc_num'].nunique():,}")
    print(f"     Fraudes           : {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.2f}%)")

    # Vérifier le tri par user
    users_sorted = df.groupby("cc_num")["unix_time"].is_monotonic_increasing.all()
    print(f"     Tri chronologique : {'✅ OK' if users_sorted else '❌ ERREUR'}")

    # Stats sur les transactions par user
    txn_per_user = df.groupby("cc_num").size()
    print(f"     Txns/user : min={txn_per_user.min()} | "
          f"mean={txn_per_user.mean():.1f} | "
          f"max={txn_per_user.max()}")

    # Exemple des 3 premiers users
    print(f"\n  Exemple (3 premiers users) :")
    first_3 = df["cc_num"].unique()[:3]
    for uid in first_3:
        u = df[df["cc_num"] == uid]
        print(f"    User {uid} : {len(u)} txns | "
              f"fraudes={u['is_fraud'].sum()} | "
              f"period={u['unix_time'].min():.0f}→{u['unix_time'].max():.0f}")

    # ── 7. Sauvegarde ─────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n  💾 Sauvegardé → {output_path}")
    print(f"{'═'*55}\n")
    print(f"  ➡️  Prochaine étape : generate_tensors_from_csv.py --csv {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prépare un CSV fraude pour generate_tensors_from_csv.py"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Chemin du CSV brut (ex: data/FraudTest.csv)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Chemin du CSV préparé (ex: data/FraudTest_sorted.csv)"
    )
    args = parser.parse_args()
    prepare_csv(args.input, args.output)