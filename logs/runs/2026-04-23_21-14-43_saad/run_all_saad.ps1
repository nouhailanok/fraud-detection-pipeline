param (
    [string]$Role = "Launcher"
)

# Common Directory and Python Paths
$DIR = "C:\Users\SAAD\OneDrive\Desktop\CSCC_S4\Projet metier\fraud-detection-pipeline"
$PY  = "$DIR\venv\Scripts\python.exe"

# ---------------------------------------------------------
# 1. THE LAUNCHER: Opens Windows Terminal with 5 tabs
# ---------------------------------------------------------
if ($Role -eq "Launcher") {
    $ScriptPath = $MyInvocation.MyCommand.Path

    Write-Output "Launching Federated Learning Pipeline..."
    
    # Build the arguments for Windows Terminal. 
    # Semicolons tell wt.exe to split tabs, not PowerShell.
    $wtArgs = "new-tab --title `"Serveur`" powershell -NoExit -Command `"& '$ScriptPath' -Role Server`" `; " +
              "new-tab --title `"Client 1`" powershell -NoExit -Command `"& '$ScriptPath' -Role Client1`" `; " +
              "new-tab --title `"Client 2`" powershell -NoExit -Command `"& '$ScriptPath' -Role Client2`" `; " +
              "new-tab --title `"Client 3`" powershell -NoExit -Command `"& '$ScriptPath' -Role Client3`" `; " +
              "new-tab --title `"Client 4`" powershell -NoExit -Command `"& '$ScriptPath' -Role Client4`""

    # Use Start-Process so PowerShell safely passes the whole string to wt.exe
    Start-Process wt.exe -ArgumentList $wtArgs
    exit
}

# ---------------------------------------------------------
# 2. THE EXECUTOR: Runs specific code based on the tab's role
# ---------------------------------------------------------
Set-Location $DIR
$env:PYTHONPATH = $DIR

switch ($Role) {
    "Server" {
        # ── Identité du run (sera incluse dans le nom du dossier) ──
        $env:NOM_USER = "saad"

        # ── Paramètres serveur / training ──
        $env:PYTHONPATH = $DIR
        $env:FL_ROUNDS = "45"
        $env:FL_MIN_CLIENTS = "4"
        $env:FL_LOCAL_EPOCHS = "3"
        $env:FLOWER_PORT = "8080" # Kept at 8080 to match the clients
        $env:FLOWER_TLS_REQUIRE_CLIENT_CERT = "false"
        $env:FL_PATIENCE = "5"

        # ── OPTIONNEL : reprise depuis checkpoint (dir ou .npz) ──
        $env:FL_RESUME_FROM = "logs\runs\2026-04-21_13-04-17_saad\checkpoints"
        
        if ($env:FL_RESUME_FROM -and $env:FL_RESUME_FROM.Trim().Length -gt 0) {
            $env:FL_RESUME = "true"
        } else {
            $env:FL_RESUME = "false"
        }

        # ── Création automatique d'un dossier par run ──
        $timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
        $runBase   = "logs\runs"
        $runName   = "${timestamp}_$($env:NOM_USER)"
        $runDir    = Join-Path $runBase $runName
        $ckptDir   = Join-Path $runDir "checkpoints"

        New-Item -ItemType Directory -Force -Path $ckptDir | Out-Null

        # On force le serveur à écrire logs/checkpoints dans ce run
        $env:FL_LOGS_DIR = $runDir
        $env:FL_CHECKPOINT_DIR = $ckptDir

        # ── Snapshot des variables du run + copie du script ──
        $envFile = Join-Path $runDir "server_run_env.txt"
        @(
            "NOM_USER=$env:NOM_USER",
            "FL_ROUNDS=$env:FL_ROUNDS",
            "FL_MIN_CLIENTS=$env:FL_MIN_CLIENTS",
            "FL_LOCAL_EPOCHS=$env:FL_LOCAL_EPOCHS",
            "FLOWER_PORT=$env:FLOWER_PORT",
            "FLOWER_TLS_REQUIRE_CLIENT_CERT=$env:FLOWER_TLS_REQUIRE_CLIENT_CERT",
            "FL_PATIENCE=$env:FL_PATIENCE",
            "FL_RESUME=$env:FL_RESUME",
            "FL_RESUME_FROM=$env:FL_RESUME_FROM",
            "FL_LOGS_DIR=$env:FL_LOGS_DIR",
            "FL_CHECKPOINT_DIR=$env:FL_CHECKPOINT_DIR"
        ) | Set-Content -Path $envFile -Encoding UTF8

        # Copies this exact launcher script into the run directory for logging purposes
        Copy-Item -Path $MyInvocation.MyCommand.Path -Destination (Join-Path $runDir "run_all_saad.ps1") -Force

        Write-Output "=== SERVEUR FL | run: $runName ==="
        & $PY federated/server.py
        
        Write-Output "Serveur termine"
        Read-Host "Appuyer sur Entree pour fermer"
    }
    
    "Client1" {
        Start-Sleep -Seconds 2 # Short delay to let the server bind to the port
        $env:CLIENT_ID = "1"
        $env:FLOWER_SERVER_HOST = "127.0.0.1"
        $env:FLOWER_SERVER_PORT = "8080"
        $env:FL_CLIENT_CONTINUOUS = "true"
        $env:DP_NOISE = "1.20"
        $env:FL_LR = "0.0005"
        $env:FL_BATCH_SIZE = "64"
        $env:FL_POS_WEIGHT = "167.0"
        
        Write-Output "=== BANQUE 1 Python 3.11.9 RTX 4060 ==="
        & $PY federated/client.py
        
        Write-Output "Client 1 termine"
        Read-Host "Appuyer sur Entree pour fermer"
    }

    "Client2" {
        Start-Sleep -Seconds 2
        $env:CLIENT_ID = "2"
        $env:FLOWER_SERVER_HOST = "127.0.0.1"
        $env:FLOWER_SERVER_PORT = "8080"
        $env:FL_CLIENT_CONTINUOUS = "true"
        $env:DP_NOISE = "1.20"
        $env:FL_LR = "0.0005"
        $env:FL_BATCH_SIZE = "64"
        $env:FL_POS_WEIGHT = "163.0"
        
        Write-Output "=== BANQUE 2 Python 3.11.9 RTX 4060 ==="
        & $PY federated/client.py
        
        Write-Output "Client 2 termine"
        Read-Host "Appuyer sur Entree pour fermer"
    }

    "Client3" {
        Start-Sleep -Seconds 2
        $env:CLIENT_ID = "3"
        $env:FLOWER_SERVER_HOST = "127.0.0.1"
        $env:FLOWER_SERVER_PORT = "8080"
        $env:FL_CLIENT_CONTINUOUS = "true"
        $env:DP_NOISE = "1.20"
        $env:FL_LR = "0.0005"
        $env:FL_BATCH_SIZE = "64"
        $env:FL_POS_WEIGHT = "193.8"
        
        Write-Output "=== BANQUE 3 Python 3.11.9 RTX 4060 ==="
        & $PY federated/client.py
        
        Write-Output "Client 3 termine"
        Read-Host "Appuyer sur Entree pour fermer"
    }

    "Client4" {
        Start-Sleep -Seconds 2
        $env:CLIENT_ID = "4"
        $env:FLOWER_SERVER_HOST = "127.0.0.1"
        $env:FLOWER_SERVER_PORT = "8080"
        $env:FL_CLIENT_CONTINUOUS = "true"
        $env:DP_NOISE = "1.20"
        $env:FL_LR = "0.0005"
        $env:FL_BATCH_SIZE = "64"
        $env:FL_POS_WEIGHT = "165.8"
        
        Write-Output "=== BANQUE 4 Python 3.11.9 RTX 4060 ==="
        & $PY federated/client.py
        
        Write-Output "Client 4 termine"
        Read-Host "Appuyer sur Entree pour fermer"
    }
}