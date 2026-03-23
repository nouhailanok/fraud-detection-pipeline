import numpy as np

class TransactionVectorizer:
    def __init__(self):
        # 1. La liste fixe des catégories (14 dimensions pour le One-Hot Encoding)
        # L'ordre est strict pour que l'IA lise toujours les données de la même façon.
        self.categories = [
            'entertainment', 'food_dining', 'gas_transport', 'grocery_net', 
            'grocery_pos', 'health_fitness', 'home', 'kids_pets', 
            'misc_net', 'misc_pos', 'personal_care', 'shopping_net', 
            'shopping_pos', 'travel'
        ]
        self.num_categories = len(self.categories)

    # --- MÉTHODES DE TRANSFORMATION MATHÉMATIQUE ---

    def _log_scale(self, amount):
        """Applique la transformation logarithmique (Log1p) pour gérer les valeurs extrêmes."""
        return np.log1p(float(amount))

    def _one_hot_encode(self, category):
        """Transforme la catégorie en un vecteur binaire (0 et 1)."""
        ohe = [0] * self.num_categories
        if category in self.categories:
            index = self.categories.index(category)
            ohe[index] = 1
        return ohe

    # --- L'ORCHESTRATEUR DE VECTORISATION ---

    def vectorize(self, enriched_json):
        """
        Prend le Grand JSON et retourne un tuple: (Vecteur_X_26D, Label_y)
        """
        # 1. Transformations spécifiques
        amount_scaled = self._log_scale(enriched_json["Original_Amount"])
        mcc_encoded = self._one_hot_encode(enriched_json["Original_MCC"])
        
        # 2. Rassemblement des 12 features numériques
        numerical_features = [
            amount_scaled,
            float(enriched_json["Feature_Client_Age"]),
            float(enriched_json["Feature_Dist_Client_Merch_Km"]),
            float(enriched_json["Feature_Time_Sin"]),
            float(enriched_json["Feature_Time_Cos"]),
            float(enriched_json["Time_Delta_Sec"]),
            float(enriched_json["Velocity_Count_24H"]),
            float(enriched_json["Velocity_Sum_24H"]),
            float(enriched_json["Prev_Txn_Distance_Km"]),
            float(enriched_json["Apparent_Travel_Speed_Kmh"]),
            float(enriched_json["Amount_vs_Avg_Ratio"]),
            float(enriched_json["Is_New_Merchant"])
        ]
        
        # 3. Concaténation : 12 Nombres + 14 Catégories = 26 dimensions
        raw_vector = numerical_features + mcc_encoded
        
        # 4. Conversion en tableau NumPy (float32 optimisé pour GPU/Deep Learning)
        X = np.array(raw_vector, dtype=np.float32)
        
        # 5. Extraction du label cible (y)
        y = np.array([int(enriched_json["Label_is_fraud"])], dtype=np.float32)
        
        return X, y