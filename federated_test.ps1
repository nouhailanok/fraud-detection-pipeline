# On définit la racine du projet comme chemin de recherche Python
$env:PYTHONPATH = $PWD.Path

# --- CONFIGURATION DU SERVEUR ---
$ServerBlock = {
    $env:PYTHONPATH = $PWD.Path
    $env:FL_ROUNDS = "3"
    $env:FL_MIN_CLIENTS = "2"
    $env:FLOWER_PORT = "8080"
    $env:FLOWER_TLS_REQUIRE_CLIENT_CERT = $false
    Write-Host "🚀 Lancement du SERVEUR..." -ForegroundColor Cyan
    python federated/server.py
}

# --- CONFIGURATION BANQUE 1 ---
$Client1Block = {
    $env:PYTHONPATH = $PWD.Path
    $env:CLIENT_ID = "1"
    $env:FLOWER_SERVER_ADDRESS = '127.0.0.1:8080'
    $env:FLOWER_SERVER_HOST = '127.0.0.1'
    $env:FLOWER_SERVER_PORT = "8080"
    $env:FL_CLIENT_CONTINUOUS = $true
    Write-Host "🏦 Lancement de la BANQUE 1..." -ForegroundColor Green
    python federated/client.py
}

# --- CONFIGURATION BANQUE 2 ---
$Client2Block = {
    $env:PYTHONPATH = $PWD.Path
    $env:CLIENT_ID = "2"
    $env:FLOWER_SERVER_ADDRESS = '127.0.0.1:8080'
    $env:FLOWER_SERVER_HOST = '127.0.0.1'
    $env:FLOWER_SERVER_PORT = "8080"
    $env:FL_CLIENT_CONTINUOUS = $true
    Write-Host "🏦 Lancement de la BANQUE 2..." -ForegroundColor Yellow
    python federated/client.py
}

# Lancement
Start-Process powershell -ArgumentList "-NoExit", "-Command", $ServerBlock
Start-Sleep -Seconds 2
Start-Process powershell -ArgumentList "-NoExit", "-Command", $Client1Block
Start-Process powershell -ArgumentList "-NoExit", "-Command", $Client2Block