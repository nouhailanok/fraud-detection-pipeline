import math
import json
from datetime import datetime, timedelta

class TransactionEnricher:
    def __init__(self, time_window_hours=24):
        self.time_window = timedelta(hours=time_window_hours)
        self.earth_radius_km = 6371.0
        
        # NOUVEAU FORMAT DU STATE :
        # { "pan_hash": { "history_24h": [...], "seen_merchants": set() } }
        self.client_state = {}

    # --- MÉTHODES MATHÉMATIQUES & PARSING ---

    def _haversine(self, lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        return round(self.earth_radius_km * c, 2)

    def _encode_cyclical_time(self, dt_obj):
        hours_decimal = dt_obj.hour + (dt_obj.minute / 60.0) + (dt_obj.second / 3600.0)
        sin_time = math.sin(2 * math.pi * hours_decimal / 24.0)
        cos_time = math.cos(2 * math.pi * hours_decimal / 24.0)
        return round(sin_time, 4), round(cos_time, 4)

    def _parse_custom_datetime(self, dt_str):
        # Format MMDDhhmmss (Année fixée à 2019 pour coller à ton CSV)
        year = 2019
        month, day = int(dt_str[0:2]), int(dt_str[2:4])
        hour, minute, sec = int(dt_str[4:6]), int(dt_str[6:8]), int(dt_str[8:10])
        return datetime(year, month, day, hour, minute, sec)

    def _calculate_age(self, dob_str, current_dt):
        """Calcule l'âge exact du client au moment de la transaction."""
        # Format du CSV : DD/MM/YYYY (ex: 21/08/1947)
        # dob = datetime.strptime(dob_str, "%d/%m/%Y")
        dob = datetime.strptime(dob_str, "%Y-%m-%d")
        # On soustrait les années, et on retire 1 si l'anniversaire n'est pas encore passé cette année
        age = current_dt.year - dob.year - ((current_dt.month, current_dt.day) < (dob.month, dob.day))
        return age

    # --- MÉTHODE DE VÉLOCITÉ ET GESTION DE MÉMOIRE (Mise à jour) ---

    def _update_and_get_velocity(self, pan, current_dt, amount, client_lat, client_lon, merchant_name):
        # Initialisation du client s'il est nouveau
        if pan not in self.client_state:
            self.client_state[pan] = {
                "history_24h": [], 
                "seen_merchants": set() # Un Set est ultra rapide pour vérifier si un élément existe
            }

        state = self.client_state[pan]
        history = state["history_24h"]
        seen_merchants = state["seen_merchants"]
        
        # 1. Nettoyage de la mémoire glissante (24h)
        cutoff_time = current_dt - self.time_window
        history = [txn for txn in history if txn["dt"] > cutoff_time]
        
        # 2. Variables de base
        time_delta_sec = 0
        velocity_count_24h = len(history)
        velocity_sum_24h = sum(txn["amount"] for txn in history)
        prev_txn_distance_km = 0.0
        apparent_travel_speed_kmh = 0.0

        # --- NOUVELLE FEATURE : Le Ratio Comportemental ---
        if velocity_count_24h > 0:
            avg_amount_24h = velocity_sum_24h / velocity_count_24h
            # +1 au dénominateur pour éviter la division par zéro si avg_amount est 0
            amount_vs_avg_ratio = round(amount / (avg_amount_24h + 1), 2)
        else:
            amount_vs_avg_ratio = 1.0 # S'il n'y a pas d'historique, le ratio est neutre (1.0)

        # --- NOUVELLE FEATURE : Nouveau Marchand ---
        is_new_merchant = 0 if merchant_name in seen_merchants else 1

        # 3. Calculs Spatiaux/Temporels par rapport à la DERNIÈRE transaction
        if velocity_count_24h > 0:
            last_txn = history[-1]
            time_delta_sec = (current_dt - last_txn["dt"]).total_seconds()
            prev_txn_distance_km = self._haversine(client_lat, client_lon, last_txn["lat"], last_txn["lon"])
            
            if time_delta_sec > 0:
                apparent_travel_speed_kmh = round(prev_txn_distance_km / (time_delta_sec / 3600.0), 2)

        # 4. Mise à jour de la mémoire
        history.append({"dt": current_dt, "amount": amount, "lat": client_lat, "lon": client_lon})
        seen_merchants.add(merchant_name) # Ajoute le marchand au Set (sans doublon)

        # Sauvegarde du nouvel état
        self.client_state[pan] = {"history_24h": history, "seen_merchants": seen_merchants}

        return {
            "Time_Delta_Sec": time_delta_sec,
            "Velocity_Count_24H": velocity_count_24h,
            "Velocity_Sum_24H": round(velocity_sum_24h, 2),
            "Prev_Txn_Distance_Km": prev_txn_distance_km,
            "Apparent_Travel_Speed_Kmh": apparent_travel_speed_kmh,
            "Amount_vs_Avg_Ratio": amount_vs_avg_ratio,
            "Is_New_Merchant": is_new_merchant
        }

    # --- L'ORCHESTRATEUR PUBLIC ---

    def enrich(self, raw_json):
        pan = raw_json["DE002_PAN_HASH"]
        amount = float(raw_json["DE004_Amount"])
        client_lat = float(raw_json["Client_Lat"])
        client_lon = float(raw_json["Client_Long"])
        merch_lat = float(raw_json["Merch_Lat"])
        merch_lon = float(raw_json["Merch_Long"])
        merchant_name = raw_json["Merchant_Name"] # Nouvelle donnée en entrée
        dob_str = raw_json["DOB"] # Nouvelle donnée en entrée
        
        dt_obj = self._parse_custom_datetime(raw_json["DE007_DateTime"])
        
        # Calculs métiers
        dist_client_merch = self._haversine(client_lat, client_lon, merch_lat, merch_lon)
        sin_t, cos_t = self._encode_cyclical_time(dt_obj)
        client_age = self._calculate_age(dob_str, dt_obj) # Nouvelle Feature
        
        # Calculs avec State
        velocity_features = self._update_and_get_velocity(pan, dt_obj, amount, client_lat, client_lon, merchant_name)

        # Assemblage du Grand JSON
        enriched_json = {
            "PAN_HASH": pan,
            "Original_Amount": amount,
            "Original_MCC": raw_json["DE018_MCC"],
            "Label_is_fraud": raw_json["Metadata_is_fraud"],
            "Feature_Client_Age": client_age,
            "Feature_Dist_Client_Merch_Km": dist_client_merch,
            "Feature_Time_Sin": sin_t,
            "Feature_Time_Cos": cos_t,
        }
        enriched_json.update(velocity_features)
        return enriched_json