Param(
    [Parameter(Position=0)]
    [ValidateSet("create","visualize")]
    [string]$Script = "create",
    [string]$Original = "",
    [string]$Uncertainty = "",
    [string]$Output = ".\\out"
)

if ($Script -eq "create") {
    if (-not $Original -or -not $Uncertainty) {
        Write-Host "Usage: .\\run_visualization.ps1 -Script create -Original <orig.nii.gz> -Uncertainty <unc.nii.gz> -Output <outdir>"
        exit 1
    }
    Write-Host "Running create_proper_uncertainty_overlay.py with provided files..."
    python .\\create_proper_uncertainty_overlay.py --original "$Original" --uncertainty "$Uncertainty" -o "$Output"
} else {
    if (-not $Original -or -not $Uncertainty) {
        Write-Host "Usage: .\\run_visualization.ps1 -Script visualize -Original <orig.nii.gz> -Uncertainty <unc.nii.gz> -Output <outdir>"
        exit 1
    }
    Write-Host "Running visualize_seg_and_uncertainty.py with provided files..."
    python .\\visualize_seg_and_uncertainty.py --original "$Original" --uncertainty "$Uncertainty" -o "$Output"
}

Write-Host "Done"
