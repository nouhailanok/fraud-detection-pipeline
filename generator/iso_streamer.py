import csv
import time
from datetime import datetime

#  variable global 
SERVER_ID = "01"
stan_counter = 1
RRN_COUNTER = 1
used_DE37 = set()


# Format montant → DE004 (12 digits, centimes)
def format_amount(amount):
    amount = float(amount)
    cents = int(amount * 100)
    return str(cents).zfill(12)


# Format date → DE007 (MMDDhhmmss)
def format_datetime(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    return dt.strftime("%m%d%H%M%S")


# Génération STAN (6 digits)
def generate_stan():
    global stan_counter
    stan = stan_counter % 1000000  # rollover at 999999
    stan_counter += 1
    return SERVER_ID + str(stan).zfill(6)  

# Génération DE037 using the first 12 digits of trans_num + collision handling

def get_de37_stream(trans_num):
    de37 = str(trans_num)[:12]

    while de37 in used_DE37:
        last_char = de37[-1]
        new_char = format((int(last_char, 16) + 1) % 16, 'x')
        de37 = de37[:-1] + new_char

    used_DE37.add(de37)
    return de37

# Mapping CSV → ISO 8583 JSON
def map_to_iso(row):
    iso_msg = {
        "message_type": "0200",
        "DE002_PAN": row["cc_num"],
        "DE003_ProcessingCode": "000000",
        "DE004_Amount": format_amount(row["amt"]),
        "DE007_DateTime": format_datetime(row["trans_date_trans_time"]),
        "DE011_STAN": generate_stan(),
        "DE018_MCC": row["category"],
        # "DE037_RRN": generate_de037(),
        # "DE037_RRN": generate_unique_DE37(row["trans_num"]),
        "DE037_RRN": get_de37_stream(row["trans_num"]),
        "DE043_MerchantLoc": f"{row['merchant']} {row['city']} {row['state']}",
        "DE049_Currency": "840",

        # Data utile pour ML / debug
        "DE123_CustomData": {
            "trans_num": row["trans_num"],
            "unix_time": int(row["unix_time"]),
            "is_fraud": int(row["is_fraud"]),
            "lat": float(row["lat"]),
            "long": float(row["long"]),
            "merch_lat": float(row["merch_lat"]),
            "merch_long": float(row["merch_long"]),
        }
    }

    return iso_msg


# Streamer principal
def stream_csv(file_path, delay=0.1):
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)

        for row in reader:
            try:
                iso_msg = map_to_iso(row)
                print(iso_msg)  # 👉 plus tard: envoyer à Kafka / API
                time.sleep(delay)

            except Exception as e:
                print(f"Erreur sur ligne : {e}")
                continue


# Lancement
if __name__ == "__main__":
    stream_csv("data/fraudTrain.csv", delay=0.1)