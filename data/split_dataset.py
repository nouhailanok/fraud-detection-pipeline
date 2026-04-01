import pandas as pd
import numpy as np
import os
from pathlib import Path

def partition_nodes(input_csv="data/fraudTrain.csv", output_base_dir="data"):
    # 1. Chargement et Tri
    print(f"📖 Chargement de {input_csv}...")
    # On garde toutes les colonnes pour que les scripts d'ingestion aient tout ce qu'il faut
    df = pd.read_csv(input_csv)
    
    print("⏳ Tri par utilisateur et temps...")
    df = df.sort_values(['cc_num', 'unix_time']).reset_index(drop=True)

    # 2. Répartition des utilisateurs (Config 1, 2, 3, 4)
    unique_users = df['cc_num'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_users)

    node_configs = [
        {'name': 'node_1', 'ratio': 0.30},
        {'name': 'node_2', 'ratio': 0.25},
        {'name': 'node_3', 'ratio': 0.25},
        {'name': 'node_4', 'ratio': 0.20},
    ]

    last_idx = 0
    for config in node_configs:
        n_users = int(len(unique_users) * config['ratio'])
        user_group = unique_users[last_idx : last_idx + n_users]
        last_idx += n_users

        # Filtrage des données du noeud
        node_df = df[df['cc_num'].isin(user_group)]
        
        # Sauvegarde (Uniquement CSV maintenant)
        node_dir = Path(output_base_dir) / config['name']
        node_dir.mkdir(parents=True, exist_ok=True)
        
        # On sauvegarde le CSV local
        csv_path = node_dir / "local_data.csv"
        node_df.to_csv(csv_path, index=False)
        
        print(f"✅ {config['name']} prêt : {len(node_df)} lignes sauvegardées dans {csv_path}")

if __name__ == "__main__":
    partition_nodes()
