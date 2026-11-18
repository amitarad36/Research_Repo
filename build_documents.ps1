param(
    [ValidateSet('all','proposal','backbone','auto')][string]$Document = 'all',
    [string]$Source = ''
)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
# XeLaTeX supports the proposal's Hebrew title; the backbone keeps pdfLaTeX.
if ($Document -eq 'auto') {
    if ([IO.Path]::GetFileNameWithoutExtension($Source) -eq 'research_backbone') { $Document = 'backbone' }
    else { $Document = 'proposal' }
}
$targets = @(
    @{ Name='proposal'; Input='research_proposal_main.tex'; Job='research_proposal_main'; Output='research_proposal.pdf' },
    @{ Name='backbone'; Input='backbone\research_backbone.tex'; Job='research_backbone'; Output='backbone\research_backbone.pdf' }
)
Push-Location -LiteralPath $projectRoot
try {
    foreach ($target in $targets) {
        if ($Document -ne 'all' -and $Document -ne $target.Name) { continue }
        $buildDir = Join-Path $projectRoot 'build'
        New-Item -ItemType Directory -Path $buildDir -Force | Out-Null
        $sourcePath = Join-Path $projectRoot $target.Input
        $engine = if ($target.Name -eq 'proposal') { 'xelatex' } else { 'pdflatex' }
        $compiler = (Get-Command $engine -ErrorAction Stop).Source
        for ($pass = 1; $pass -le 2; $pass++) {
            $consolePath = Join-Path $buildDir ("$($target.Name)-pass-$pass.txt")
            & $compiler '-interaction=nonstopmode' '-halt-on-error' '-file-line-error' '-no-shell-escape' '-synctex=1' "-jobname=$($target.Job)" "-output-directory=$buildDir" $sourcePath *> $consolePath
            if ($LASTEXITCODE -ne 0) {
                Get-Content -LiteralPath $consolePath -Tail 50 | Write-Host
                throw "Compilation failed for $($target.Name). See $consolePath"
            }
        }
        Copy-Item -LiteralPath (Join-Path $buildDir ($target.Job + '.pdf')) -Destination (Join-Path $projectRoot $target.Output) -Force
        Write-Output ("Built " + $target.Output)
    }
}
finally { Pop-Location }
