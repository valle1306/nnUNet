# Fetch selected files from Amarel and place them into your local nnUNet workspace.
# Usage: Open PowerShell in Windows and run: .\scripts\fetch_from_amarel.ps1

# --- Configuration - edit if needed ---
$UserAtHost = "hpl14@amarel.rutgers.edu"
$RemoteBase = "/scratch/hpl14/nnunet_v2_valerie"
$LocalNNUNet = "C:\\Users\\lpnhu\\Downloads\\nnUNet\\nnunetv2"

# Mapping: remote -> local destination
$Mappings = @(
    @{ remote = "$RemoteBase/code/predict_with_mc_dropout_edited.py"; local = Join-Path $LocalNNUNet "inference\\predict_with_mc_dropout_edited.py" },
    @{ remote = "$RemoteBase/code/predict_with_mc_dropout_ver4.py"; local = Join-Path $LocalNNUNet "inference\\predict_with_mc_dropout_ver4_from_amarel.py" },
    @{ remote = "$RemoteBase/code/visualize_uncertainty_overlay_amarel.py"; local = Join-Path $LocalNNUNet "visualization\\visualize_uncertainty_overlay_amarel.py" },
    @{ remote = "$RemoteBase/code/visualize_seg_and_uncertainty.py"; local = Join-Path $LocalNNUNet "visualization\\visualize_seg_and_uncertainty.py" },
    @{ remote = "$RemoteBase/code/create_proper_uncertainty_overlay.py"; local = Join-Path $LocalNNUNet "visualization\\create_proper_uncertainty_overlay.py" },
    @{ remote = "$RemoteBase/code/run_visualization.sh"; local = Join-Path $LocalNNUNet "visualization\\run_visualization.sh" },
    @{ remote = "$RemoteBase/code/run_mc_dropout_inference.py"; local = Join-Path $LocalNNUNet "inference\\run_mc_dropout_inference.py" }
)

Write-Host ("Will copy these files from " + $UserAtHost + ":" + $RemoteBase + " to local: " + $LocalNNUNet) -ForegroundColor Cyan

foreach ($m in $Mappings) {
    $remote = $m.remote
    $local = $m.local
    $destDir = Split-Path -Path $local -Parent

    if (-not (Test-Path -Path $destDir)) {
        Write-Host "Creating directory: $destDir" -ForegroundColor Yellow
        New-Item -ItemType Directory -Path $destDir -Force | Out-Null
    }

    Write-Host ("Copying: " + $UserAtHost + ":" + $remote + " -> " + $local) -ForegroundColor Green
    try {
        # scp will prompt for password if needed. If you use key-based auth this will be passwordless.
        $src = $UserAtHost + ":" + $remote
        scp $src $local
        if ($LASTEXITCODE -ne 0) {
            Write-Host "scp exited with code $LASTEXITCODE for $remote" -ForegroundColor Red
        } else {
            Write-Host "Copied: $local" -ForegroundColor Green
        }
    } catch {
        Write-Host ("Error copying " + $remote + ": " + $_) -ForegroundColor Red
    }
}

Write-Host "Done. If scp isn't available on this machine, use WinSCP or PSCP (PuTTY) or run the individual commands from WSL." -ForegroundColor Cyan

Write-Host "Next steps:\n 1) Run Get-FileHash -Algorithm SHA256 on copies and compare with remote sha256sum output (recommended).\n 2) Run git --no-pager diff --no-index to inspect file differences before committing." -ForegroundColor Magenta
