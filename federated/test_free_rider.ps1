# test_free_rider.ps1 — Test Behavioral Analysis : FREE RIDER (nœud 4)
# Nœud 4 entraîne normalement les 2 premiers rounds, puis devient free rider
# Le BehavioralAnalyzer devrait détecter l'anomalie à partir du round 3

$DIR = "C:\Users\medam\OneDrive\Documents\Projects\fraud-detection-pipeline"
$PY  = "C:\Users\medam\AppData\Local\Programs\Python\Python312\python.exe"

# ── SERVEUR ────────────────────────────────────────────────────────────────────
$ServerScript = {
    param($DIR, $PY)
    Set-Location $DIR
    $env:PYTHONPATH                = $DIR
    $env:NOM_USER                  = "Mohamed"
    $env:FL_ROUNDS                 = "8"
    $env:FL_MIN_CLIENTS            = "4"
    $env:FL_LOCAL_EPOCHS           = "3"
    $env:FLOWER_PORT               = "8080"
    $env:FL_PATIENCE               = "10"
    $env:FL_RESUME                 = "false"
    $env:BA_ACTIVATION_ROUND       = "3"
    $env:BA_CONTAMINATION          = "0.25"
    $env:FL_LOGS_DIR               = "logs/runs/test_free_rider"
    $env:FL_CHECKPOINT_DIR         = "logs/runs/test_free_rider/checkpoints"
    Write-Output "=== SERVEUR FL — Test FREE RIDER ==="
    & $PY federated/server.py
    Read-Host "Appuyer sur Entree pour fermer"
}

# ── BANQUES LÉGITIMES (1, 2, 3) ───────────────────────────────────────────────
$Client1 = {
    param($DIR, $PY)
    Set-Location $DIR
    $env:PYTHONPATH           = $DIR
    $env:CLIENT_ID            = "1"
    $env:FLOWER_SERVER_HOST   = "127.0.0.1"
    $env:FLOWER_SERVER_PORT   = "8080"
    $env:FL_CLIENT_CONTINUOUS = "false"
    $env:DP_NOISE             = "1.0"
    $env:DP_EPSILON_TARGET    = "1.0"
    $env:FL_LR                = "0.0005"
    $env:FL_BATCH_SIZE        = "64"
    $env:FL_POS_WEIGHT        = "167.0"
    Write-Output "BANQUE 1 — Légitime"
    & $PY federated/client.py
    Read-Host "Entree pour fermer"
}

$Client2 = {
    param($DIR, $PY)
    Set-Location $DIR
    $env:PYTHONPATH           = $DIR
    $env:CLIENT_ID            = "2"
    $env:FLOWER_SERVER_HOST   = "127.0.0.1"
    $env:FLOWER_SERVER_PORT   = "8080"
    $env:FL_CLIENT_CONTINUOUS = "false"
    $env:DP_NOISE             = "1.0"
    $env:DP_EPSILON_TARGET    = "1.0"
    $env:FL_LR                = "0.0005"
    $env:FL_BATCH_SIZE        = "64"
    $env:FL_POS_WEIGHT        = "163.0"
    Write-Output "BANQUE 2 — Légitime"
    & $PY federated/client.py
    Read-Host "Entree pour fermer"
}

$Client3 = {
    param($DIR, $PY)
    Set-Location $DIR
    $env:PYTHONPATH           = $DIR
    $env:CLIENT_ID            = "3"
    $env:FLOWER_SERVER_HOST   = "127.0.0.1"
    $env:FLOWER_SERVER_PORT   = "8080"
    $env:FL_CLIENT_CONTINUOUS = "false"
    $env:DP_NOISE             = "1.0"
    $env:DP_EPSILON_TARGET    = "1.0"
    $env:FL_LR                = "0.0005"
    $env:FL_BATCH_SIZE        = "64"
    $env:FL_POS_WEIGHT        = "194.0"
    Write-Output "BANQUE 3 — Légitime"
    & $PY federated/client.py
    Read-Host "Entree pour fermer"
}

# ── NŒUD 4 : FREE RIDER (actif dès le round 1) ───────────────────────────────
$Client4_FreeRider = {
    param($DIR, $PY)
    Set-Location $DIR
    $env:PYTHONPATH                = $DIR
    $env:CLIENT_ID                 = "4"
    $env:FLOWER_SERVER_HOST        = "127.0.0.1"
    $env:FLOWER_SERVER_PORT        = "8080"
    $env:FL_CLIENT_CONTINUOUS      = "false"
    $env:FREE_RIDER_MODE           = "zero"
    $env:FREE_RIDER_START_ROUND    = "1"
    $env:FL_BATCH_SIZE             = "64"
    $env:FL_POS_WEIGHT             = "166.0"
    Write-Output "BANQUE 4 — FREE RIDER (mode=zero, dès round 1)"
    & $PY attack_sim/free_rider_client.py
    Read-Host "Entree pour fermer"
}

# ── LANCEMENT ─────────────────────────────────────────────────────────────────
Write-Output "Lancement test FREE RIDER — 4 nœuds, 8 rounds"
Write-Output "Behavioral Analysis activé au round 3 (contamination=0.25)"
Write-Output ""

Start-Process powershell -ArgumentList "-NoExit", "-Command", $ServerScript, $DIR, $PY
Start-Sleep -Seconds 4
Start-Process powershell -ArgumentList "-NoExit", "-Command", $Client1, $DIR, $PY
Start-Process powershell -ArgumentList "-NoExit", "-Command", $Client2, $DIR, $PY
Start-Process powershell -ArgumentList "-NoExit", "-Command", $Client3, $DIR, $PY
Start-Process powershell -ArgumentList "-NoExit", "-Command", $Client4_FreeRider, $DIR, $PY

Write-Output "5 fenetres ouvertes — surveillez les alertes BA dans la fenetre serveur"
