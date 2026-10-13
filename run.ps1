<#
    Purple Loop - one-command runner for Windows PowerShell.

    .\run.ps1 setup      create .venv and install requirements
    .\run.ps1 test       run the test suite
    .\run.ps1 run  [20]  run the loop for N rounds
    .\run.ps1 report     rebuild figures and print the summary
    .\run.ps1 registry   print the rule registry audit trail
    .\run.ps1 ablate     compare feature groups
    .\run.ps1 reset      delete generated artifacts
#>
param(
    [Parameter(Position = 0)][string]$Task = "help",
    [Parameter(Position = 1)][int]$Rounds = 20
)

$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

switch ($Task) {
    "setup" {
        python -m venv (Join-Path $PSScriptRoot ".venv")
        & $py -m pip install --quiet --upgrade pip
        & $py -m pip install -r (Join-Path $PSScriptRoot "requirements.txt")
    }
    "test"     { & $py -m pytest }
    "run"      { & $py -m purpleloop run --rounds $Rounds }
    "report"   { & $py -m purpleloop report }
    "registry" { & $py -m purpleloop registry }
    "ablate"   { & $py -m purpleloop ablate --rounds 12 }
    "reset"    { & $py -m purpleloop reset --yes }
    default {
        Write-Host "usage: .\run.ps1 {setup|test|run [rounds]|report|registry|ablate|reset}"
    }
}
