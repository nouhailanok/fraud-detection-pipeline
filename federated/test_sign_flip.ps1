# test_sign_flip.ps1 — Test Behavioral Analysis : SIGN FLIP (nœud 3)
# Utilise poison_client.py — nœud 3 inverse ses gradients à partir du round 3

$DIR = "C:\Users\medam\OneDrive\Documents\Projects\fraud-detection-pipeline"
$PY  = "C:\Users\medam\AppData\Local\Programs\Python\Python312\python.exe"

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
    $env:FL_LOGS_DIR               = "logs/runs/test_sign_flip"
    $env:FL_CHECKPOINT_DIR         = "logs/runs/test_sign_flip/checkpoints"
    Write-Output "=== SERVEUR FL — Test SIGN FLIP ==="
    & $PY federated/server.py
    Read-Host "Appuyer sur Entree pour fermer"
}

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

$Client4 = {
    param($DIR, $PY)
    Set-Location $DIR
    $env:PYTHONPATH           = $DIR
    $env:CLIENT_ID            = "4"
    $env:FLOWER_SERVER_HOST   = "127.0.0.1"
    $env:FLOWER_SERVER_PORT   = "8080"
    $env:FL_CLIENT_CONTINUOUS = "false"
    $env:DP_NOISE             = "1.0"
    $env:DP_EPSILON_TARGET    = "1.0"
    $env:FL_LR                = "0.0005"
    $env:FL_BATCH_SIZE        = "64"
    $env:FL_POS_WEIGHT        = "166.0"
    Write-Output "BANQUE 4 — Légitime"
    & $PY federated/client.py
    Read-Host "Entree pour fermer"
}

# ── NŒUD 3 : SIGN FLIP via poison_client.py ───────────────────────────────────
$Client3_SignFlip = {
    param($DIR, $PY)
    Set-Location $DIR
    $env:PYTHONPATH                = $DIR
    $env:CLIENT_ID                 = "3"
    $env:FLOWER_SERVER_HOST        = "127.0.0.1"
    $env:FLOWER_SERVER_PORT        = "8080"
    $env:FL_CLIENT_CONTINUOUS      = "false"
    $env:ATTACK_MODE               = "SIGN_FLIP"
    $env:ATTACK_INTENSITY          = "1.0"
    $env:ATTACK_START_ROUND        = "3"
    $env:DP_NOISE                  = "1.0"
    $env:DP_EPSILON_TARGET         = "1.0"
    $env:FL_BATCH_SIZE             = "64"
    $env:FL_POS_WEIGHT             = "194.0"
    Write-Output "BANQUE 3 — SIGN FLIP (à partir du round 3)"
    & $PY attack_sim/poison_client.py
    Read-Host "Entree pour fermer"
}

Write-Output "Lancement test SIGN FLIP — 4 nœuds, 8 rounds"
Write-Output "Nœud 3 : SIGN_FLIP actif dès round 3"
Write-Output "Behavioral Analysis activé au round 3 (contamination=0.25)"
Write-Output ""

Start-Process powershell -ArgumentList "-NoExit", "-Command", $ServerScript, $DIR, $PY
Start-Sleep -Seconds 4
Start-Process powershell -ArgumentList "-NoExit", "-Command", $Client1, $DIR, $PY
Start-Process powershell -ArgumentList "-NoExit", "-Command", $Client2, $DIR, $PY
Start-Process powershell -ArgumentList "-NoExit", "-Command", $Client3_SignFlip, $DIR, $PY
Start-Process powershell -ArgumentList "-NoExit", "-Command", $Client4, $DIR, $PY

Write-Output "5 fenetres ouvertes"
