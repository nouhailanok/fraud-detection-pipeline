# Set-Location "C:\Users\medam\OneDrive\Documents\Projects\fraud-detection-pipeline"
# $env:PYTHONPATH = $PWD.Path
# $env:CLIENT_ID = "2"
# $env:FLOWER_SERVER_HOST = "127.0.0.1"
# $env:FLOWER_SERVER_PORT = "8090"
# $env:FL_CLIENT_CONTINUOUS = "true"
# $env:DP_NOISE = "1.5"
# $env:FL_LR = "0.0005"
# $env:FL_BATCH_SIZE = "256"
# $env:FL_POS_WEIGHT = "167.0"
# Write-Output "=== BANQUE 2 Python 3.12 RTX 3050 Ti ==="
# & "C:\Users\medam\AppData\Local\Programs\Python\Python312\python.exe" federated/client.py
# Write-Output "Client 2 termine"
# Read-Host "Appuyer sur Entree pour fermer"




# Set-Location "C:\Users\medam\OneDrive\Documents\Projects\fraud-detection-pipeline"
# $env:PYTHONPATH = $PWD.Path
# $env:CLIENT_ID = "2"
# $env:FLOWER_SERVER_HOST = "127.0.0.1"
# $env:FLOWER_SERVER_PORT = "8080"
# $env:FL_CLIENT_CONTINUOUS = "true"
# $env:DP_NOISE = "1.00"
# $env:DP_EPSILON_TARGET = "1.0"
# $env:FL_LR = "0.0005"
# $env:FL_BATCH_SIZE = "128"
# $env:FL_POS_WEIGHT = "163.0"
# Write-Output "BANQUE 2 Python 3.12 RTX 3050 Ti"
# & "C:\Users\medam\AppData\Local\Programs\Python\Python312\python.exe" federated/client.py
# Write-Output "Client 2 termine"
# Read-Host "Appuyer sur Entree pour fermer"










Set-Location "C:\Users\medam\OneDrive\Documents\Projects\fraud-detection-pipeline"
$env:PYTHONPATH           = $PWD.Path
$env:CLIENT_ID            = "2"
$env:FLOWER_SERVER_HOST   = "127.0.0.1"
$env:FLOWER_SERVER_PORT   = "8080"
$env:FL_CLIENT_CONTINUOUS = "false"
$env:DP_NOISE             = "1.00"
$env:DP_EPSILON_TARGET    = "1.0"
$env:FL_LR                = "0.0005"
$env:FL_BATCH_SIZE        = "64"
$env:FL_POS_WEIGHT        = "163.0"
Write-Output "BANQUE 2 - V4 noise=1.00 batch=64"
& "C:\Users\medam\AppData\Local\Programs\Python\Python312\python.exe" federated/client.py
Write-Output "Client 2 termine"
Read-Host "Appuyer sur Entree pour fermer"