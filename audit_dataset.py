import pandas as pd

def analyze_user_distribution(csv_path, user_column='cc_num'):
    print(f"🔍 Démarrage de l'audit sur : {csv_path}...")
    
    try:
        # Chargement rapide du CSV avec Pandas
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ Erreur : Le fichier '{csv_path}' est introuvable.")
        return

    # 1. Calculs globaux
    total_transactions = len(df)
    total_users = df[user_column].nunique()
    
    print("\n" + "="*40)
    print("📊 RÉSUMÉ GLOBAL DU DATASET")
    print("="*40)
    print(f"👥 Nombre total d'utilisateurs uniques : {total_users}")
    print(f"💳 Nombre total de transactions       : {total_transactions}")
    print(f"📈 Moyenne de transactions par user   : {total_transactions / total_users:.2f}")
    
    # 2. Comptage par utilisateur
    # value_counts() trie automatiquement du plus grand au plus petit
    tx_per_user = df[user_column].value_counts()
    
    # 3. Calcul du pourcentage
    tx_percentage = (tx_per_user / total_transactions) * 100
    
    # 4. Création d'un tableau récapitulatif (DataFrame)
    summary_df = pd.DataFrame({
        'Nombre_Transactions': tx_per_user,
        'Pourcentage_Dataset_%': tx_percentage.round(4) # Arrondi à 4 décimales
    })
    
    # Pour nommer la colonne de l'index (l'ID de l'utilisateur)
    summary_df.index.name = 'User_ID'

    # --- AFFICHAGE DES RÉSULTATS CLÉS ---
    print("\n" + "="*40)
    print("🏆 TOP 5 (Les plus actifs)")
    print("="*40)
    print(summary_df.head(5))
    
    print("\n" + "="*40)
    print("⚠️ FLOP 5 (Les moins actifs - Cause probable du 'Test: 0')")
    print("="*40)
    print(summary_df.tail(5))
    
    # Combien d'utilisateurs ont moins de 5 transactions ?
    users_less_than_5 = (summary_df['Nombre_Transactions'] < 5).sum()
    print("\n" + "="*40)
    print("🚨 ALERTE DISTRIBUTION")
    print("="*40)
    print(f"Utilisateurs avec moins de 5 transactions : {users_less_than_5} "
          f"({(users_less_than_5/total_users)*100:.1f}% des utilisateurs)")

    # 5. Sauvegarde des résultats complets
    output_filename = "audit_utilisateurs_resultat.csv"
    summary_df.to_csv(output_filename)
    print(f"\n💾 Le rapport détaillé pour CHAQUE utilisateur a été sauvegardé dans : {output_filename}")


if __name__ == "__main__":
    # D'après la structure de ton projet, on pointe vers ton fichier brut
    chemin_csv = "data/node_1/local_data.csv" 
    
    # La colonne qui identifie l'utilisateur dans ton map_to_iso était "cc_num" (le PAN)
    colonne_utilisateur = "cc_num" 
    
    analyze_user_distribution(chemin_csv, colonne_utilisateur)