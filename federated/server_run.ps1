# Set-Location "C:\Users\medam\OneDrive\Documents\Projects\fraud-detection-pipeline"
# $env:PYTHONPATH = $PWD.Path
# $env:FL_ROUNDS = "10"
# $env:FL_MIN_CLIENTS = "2"
# $env:FL_LOCAL_EPOCHS = "5"
# $env:FLOWER_PORT = "8090"
# $env:FLOWER_TLS_REQUIRE_CLIENT_CERT = "false"
# Write-Output "=== SERVEUR FL Python 3.12 (2 noeuds RTX 3050 Ti) ==="
# & "C:\Users\medam\AppData\Local\Programs\Python\Python312\python.exe" federated/server.py
# Write-Output "Serveur termine"
# Read-Host "Appuyer sur Entree pour fermer"



# Set-Location "C:\Users\medam\OneDrive\Documents\Projects\fraud-detection-pipeline"
# $env:PYTHONPATH = $PWD.Path
# $env:FL_ROUNDS = "15"
# $env:FL_MIN_CLIENTS = "4"
# $env:FL_LOCAL_EPOCHS = "3"
# $env:FLOWER_PORT = "8080"
# $env:FLOWER_TLS_REQUIRE_CLIENT_CERT = "false"
# $env:FL_PATIENCE = "5"
# $env:FL_RESUME = "false"
# Write-Output "SERVEUR FL - 15 rounds - 3 epochs - 4 clients"
# & "C:\Users\medam\AppData\Local\Programs\Python\Python312\python.exe" federated/server.py
# Write-Output "Serveur termine"
# Read-Host "Appuyer sur Entree pour fermer"




Set-Location "C:\Users\medam\OneDrive\Documents\Projects\fraud-detection-pipeline"
$env:PYTHONPATH                     = $PWD.Path
$env:NOM_USER                       = "Mohamed"
$env:FL_LOGS_DIR                    = "logs/runs/V4_noise1.00_batch128"
$env:FL_CHECKPOINT_DIR              = "logs/runs/V4_noise1.00_batch128/checkpoints"
$env:FL_ROUNDS                      = "15"
$env:FL_MIN_CLIENTS                 = "4"
$env:FL_LOCAL_EPOCHS                = "3"
$env:FLOWER_PORT                    = "8080"
$env:FLOWER_TLS_REQUIRE_CLIENT_CERT = "false"
$env:FL_PATIENCE                    = "5"
$env:FL_RESUME                      = "false"
Write-Output "SERVEUR FL V4 - noise=1.00 batch=128"
& "C:\Users\medam\AppData\Local\Programs\Python\Python312\python.exe" federated/server.py
Write-Output "Serveur termine"
Read-Host "Appuyer sur Entree pour fermer"