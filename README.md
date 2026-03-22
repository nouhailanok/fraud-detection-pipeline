# fraud-detection-pipeline

# ISO 8583 Transaction Streamer

This module simulates real-time financial transactions and converts them into an ISO 8583-like JSON format.
It is designed for use in fraud detection (AML) pipelines and real-time data streaming systems.

---

## 📌 Overview

The script reads transactions from a CSV dataset and streams them one by one (every 0.1s), mimicking a real banking transaction flow.
Each transaction is mapped to ISO 8583 fields and enriched with additional data for machine learning purposes.

---

## ⚙️ Key Components

### 1. Amount Formatting (DE004)

```python
format_amount(amount)
```

* Converts the transaction amount into **12-digit numeric format**
* Expressed in **cents**
* Example: `12.5 → 000000001250`

---

### 2. Date Formatting (DE007)

```python
format_datetime(date_str)
```

* Converts datetime into ISO format: `MMDDhhmmss`
* Example: `2020-01-01 12:30:45 → 0101123045`

---

### 3. STAN Generation (DE011)

```python
generate_stan()
```

* Generates a **6-digit System Trace Audit Number**
* Auto-incremented counter
* Rolls over after `999999`

---

### 4. RRN Generation (DE037)

```python
get_de37_stream(trans_num)
```

* Uses the **first 12 characters of `trans_num`**
* Ensures ISO compliance (**AN12 format**)
* Handles collisions:

  * If duplicate detected → modifies last character (hex increment)
  * Guarantees uniqueness during streaming

---

### 5. ISO Mapping

```python
map_to_iso(row)
```

Maps CSV fields to ISO 8583 structure:

| Field | Description                |
| ----- | -------------------------- |
| DE002 | Card number (PAN)          |
| DE003 | Processing code            |
| DE004 | Amount                     |
| DE007 | Transaction date/time      |
| DE011 | STAN                       |
| DE018 | Merchant category          |
| DE037 | Retrieval Reference Number |
| DE043 | Merchant location          |
| DE049 | Currency (USD = 840)       |

---

### 6. Custom Data (DE123)

Additional transaction metadata used for:

* Fraud detection
* Machine learning models
* Debugging

Includes:

* `trans_num`
* `unix_time`
* `is_fraud`
* Geolocation (lat/long)

---

## 🔄 Streaming Logic

```python
stream_csv(file_path, delay=0.1)
```

* Reads transactions from CSV
* Emits one transaction every **0.1 seconds**
* Simulates real-time transaction processing
* Currently prints JSON output (can be replaced by Kafka/API)

---

## ▶️ How to Run

```bash
python generator/iso_streamer.py
```

---

## 🚀 Future Improvements

* Integrate with **Kafka** for real-time pipelines
* Send data to a **REST API**
* Connect to **fraud detection models**
* Add logging & monitoring

---

## 🎯 Use Cases

* Fraud Detection (AML systems)
* Real-time data streaming simulation
* ISO 8583 message generation
* Backend testing for fintech systems

---

## 👥 Notes for Team

* Ensure the CSV file exists at: `data/fraudTrain.csv`
* Avoid modifying global counters unless necessary
* DE037 uniqueness is handled in-memory (reset if script restarts)
* Can be extended easily for Kafka or API integration

---
