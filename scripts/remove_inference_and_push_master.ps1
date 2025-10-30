# Safe remove specified inference files: backup outside repo, git rm, commit to master and push
# Usage: run from repo root in PowerShell: .\scripts\remove_inference_and_push_master.ps1

$FilesToRemove = @( 
    "nnunetv2\inference\predict_with_mc_dropout.py",
    "nnunetv2\inference\predict_with_mc_dropout_ver4_from_amarel.py",
    "nnunetv2\inference\visualize_segmentation.py"
)

$ts = Get-Date -Format yyyyMMdd_HHmmss
$BackupDir = "C:\Users\lpnhu\Downloads\nnUNet\inference_backup_$ts"

Write-Host "Backup dir will be: $BackupDir" -ForegroundColor Cyan
New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null

foreach ($f in $FilesToRemove) {
    $src = Join-Path (Get-Location) $f
    if (Test-Path -Path $src) {
        $dst = Join-Path $BackupDir (Split-Path -Path $f -Leaf)
        try {
            Move-Item -Path $src -Destination $dst -Force
            Write-Host "Moved $f -> $dst" -ForegroundColor Green
        } catch {
            Write-Host ("Failed to move " + $f + ": " + $_) -ForegroundColor Red
        }
    } else {
        Write-Host ("Not found (skipping): " + $f) -ForegroundColor Yellow
    }
}

# Now remove from git (on master) and commit
try {
    git checkout master
} catch {
    Write-Host "Failed to checkout master: $_" -ForegroundColor Red
    exit 1
}

try {
    git rm -f @FilesToRemove
} catch {
    Write-Host "git rm returned non-zero or files not tracked: $_" -ForegroundColor Yellow
}

$msg = "chore(inference): remove obsolete local inference files; backups moved to inference_backup_$ts"
try {
    git commit -m $msg
    Write-Host "Committed removal: $msg" -ForegroundColor Green
} catch {
    Write-Host "No changes to commit or commit failed: $_" -ForegroundColor Yellow
}

try {
    git push origin master
    Write-Host "Pushed commit to origin/master" -ForegroundColor Cyan
} catch {
    Write-Host "git push failed: $_" -ForegroundColor Red
}

# Print last commit
try {
    $sha = git rev-parse --short HEAD
    Write-Host "HEAD is now: $sha" -ForegroundColor Cyan
} catch {
    Write-Host "Could not get HEAD: $_" -ForegroundColor Yellow
}

Write-Host "Done. Backups are in: $BackupDir" -ForegroundColor Magenta
