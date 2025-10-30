# Move obsolete inference python files into a timestamped backup folder under nnunetv2/inference/obsolete_backup
# Usage: run from repo root in PowerShell: .\scripts\cleanup_inference.ps1

$InferenceDir = "C:\Users\lpnhu\Downloads\nnUNet\nnunetv2\inference"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$BackupDir = Join-Path $InferenceDir ("obsolete_backup_$timestamp")

# Files to move (update this list if you want different files kept/removed)
$FilesToMove = @(
    "JHU_inference.py",
    "predict_with_mc_dropout_org.py",
    "predict_with_mc_dropout_ver1.py",
    "predict_with_mc_dropout_ver2.py",
    "predict_with_mc_dropout_ver3.py",
    "predict_with_mc_dropout_ver4.py"
)

if (-not (Test-Path -Path $BackupDir)) {
    New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null
    Write-Host "Created backup directory: $BackupDir" -ForegroundColor Green
}

foreach ($f in $FilesToMove) {
    $src = Join-Path $InferenceDir $f
    if (Test-Path -Path $src) {
        $dst = Join-Path $BackupDir $f
        try {
            Move-Item -Path $src -Destination $dst -Force
            Write-Host "Moved $f -> $BackupDir" -ForegroundColor Cyan
        } catch {
            Write-Host ("Failed to move " + $f + ": " + $_) -ForegroundColor Red
        }
    } else {
        Write-Host "Not found (skipping): $f" -ForegroundColor Yellow
    }
}

Write-Host "Cleanup complete. Review $BackupDir to restore any file if needed." -ForegroundColor Green
Write-Host "Next: run the git commands to stage the moves, commit and push. I can prepare those for you if you want." -ForegroundColor Magenta

# Optionally move NIfTI data files out of inference (safer than permanent delete)
$moveNiftisResp = Read-Host "Do you also want to remove .nii / .nii.gz files from the inference folder? This will MOVE them to a backup subfolder (y/N)"
if ($moveNiftisResp -and $moveNiftisResp.ToLower().StartsWith('y')) {
    $DataBackupDir = Join-Path $BackupDir "data_backup"
    if (-not (Test-Path -Path $DataBackupDir)) {
        New-Item -ItemType Directory -Path $DataBackupDir -Force | Out-Null
        Write-Host "Created data backup directory: $DataBackupDir" -ForegroundColor Yellow
    }

    $niftiFiles = Get-ChildItem -Path $InferenceDir -Filter *.nii* -File -ErrorAction SilentlyContinue
    if ($niftiFiles -and $niftiFiles.Count -gt 0) {
        foreach ($nf in $niftiFiles) {
            try {
                Move-Item -Path $nf.FullName -Destination (Join-Path $DataBackupDir $nf.Name) -Force
                Write-Host ("Moved data file: " + $nf.Name + " -> " + $DataBackupDir) -ForegroundColor Cyan
            } catch {
                Write-Host ("Failed to move data file " + $nf.Name + ": " + $_) -ForegroundColor Red
            }
        }
        Write-Host "All matching .nii/.nii.gz files were moved to $DataBackupDir" -ForegroundColor Green
    } else {
        Write-Host "No .nii/.nii.gz files found in $InferenceDir" -ForegroundColor Yellow
    }
} else {
    Write-Host "Skipping moving NIfTI data files. If you want to remove them later, re-run this script." -ForegroundColor Cyan
}
