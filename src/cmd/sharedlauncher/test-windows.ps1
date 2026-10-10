$ErrorActionPreference = 'Stop'

$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$env:GOROOT = $root
$go = Join-Path $root 'bin\go.exe'
$python = (Get-Command python.exe -ErrorAction Stop).Source
$gcc = (Get-Command gcc.exe -ErrorAction Stop).Source

function Invoke-Go([string[]] $Arguments) {
    & $go @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "go $($Arguments -join ' ') failed with exit code $LASTEXITCODE"
    }
}

Invoke-Go @('version')
Invoke-Go @('env')
Invoke-Go @('tool', 'compile', '-V')
Invoke-Go @('tool', 'asm', '-V')
Invoke-Go @('tool', 'link', '-V')
Invoke-Go @('build', 'std')

$pythonVersion = & $python (Join-Path $root 'bin\go.py') version
if ($LASTEXITCODE -ne 0 -or $pythonVersion -notmatch '^go version ') { throw "Python go launcher failed: $pythonVersion" }
$batchVersion = & (Join-Path $root 'bin\go.bat') version
if ($LASTEXITCODE -ne 0 -or $batchVersion -notmatch '^go version ') { throw "Batch go launcher failed: $batchVersion" }
foreach ($tool in @('asm', 'cgo', 'compile', 'link', 'preprofile', 'cover', 'export', 'fix', 'vet')) {
    $toolPython = Join-Path $root "pkg\tool\windows_amd64\$tool.py"
    & $python $toolPython '-V' | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Python FFI launcher for $tool failed with exit code $LASTEXITCODE" }
    $toolBatch = Join-Path $root "pkg\tool\windows_amd64\$tool.bat"
    & $toolBatch '-V' | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Batch FFI launcher for $tool failed with exit code $LASTEXITCODE" }
}

$work = Join-Path $env:RUNNER_TEMP 'shared-toolchain-smoke'
Remove-Item $work -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path (Join-Path $work 'mathx') -Force | Out-Null
@'
module example.com/smoke

go 1.26
'@ | Set-Content -Encoding ascii (Join-Path $work 'go.mod')
@'
package mathx

func Add(a, b int) int { return a + b }
'@ | Set-Content -Encoding ascii (Join-Path $work 'mathx\mathx.go')
@'
package mathx

import "testing"

func TestAdd(t *testing.T) {
    if Add(20, 22) != 42 { t.Fatal("bad sum") }
}
'@ | Set-Content -Encoding ascii (Join-Path $work 'mathx\mathx_test.go')
@'
package main

import (
    "fmt"
    "example.com/smoke/mathx"
)

func main() { fmt.Println(mathx.Add(20, 22)) }
'@ | Set-Content -Encoding ascii (Join-Path $work 'main.go')

Push-Location $work
try {
    $runResult = & $go run .
    if ($LASTEXITCODE -ne 0 -or $runResult.Trim() -ne '42') { throw "go run returned '$runResult'" }
    Invoke-Go @('test', './...')
    Invoke-Go @('build', '-o', 'hello.exe', '.')
    $result = & (Join-Path $work 'hello.exe')
    if ($LASTEXITCODE -ne 0 -or $result.Trim() -ne '42') { throw "built program returned '$result'" }

    $env:GOOS = 'linux'
    $env:GOARCH = 'amd64'
    Invoke-Go @('build', '-o', 'hello-linux', '.')
    Remove-Item Env:GOOS
    Remove-Item Env:GOARCH
} finally {
    Pop-Location
}

$env:GOROOT = $root
$ffi = Join-Path $work 'ffi_smoke.exe'
& $gcc (Join-Path $PSScriptRoot 'ffi_smoke.c') '-O2' '-o' $ffi
if ($LASTEXITCODE -ne 0) { throw 'could not build the direct FFI smoke test' }
$ffiCases = @(
    @('bin\go.dll', 'dll_go_main', 'go', 'version'),
    @('bin\gofmt.dll', 'dll_gofmt_main', 'gofmt', (Join-Path $work 'main.go')),
    @('pkg\tool\windows_amd64\asm.dll', 'dll_asm_main', 'asm', '-V'),
    @('pkg\tool\windows_amd64\cgo.dll', 'dll_cgo_main', 'cgo', '-V'),
    @('pkg\tool\windows_amd64\compile.dll', 'dll_compile_main', 'compile', '-V'),
    @('pkg\tool\windows_amd64\link.dll', 'dll_link_main', 'link', '-V'),
    @('pkg\tool\windows_amd64\preprofile.dll', 'dll_preprofile_main', 'preprofile', '-V'),
    @('pkg\tool\windows_amd64\cover.dll', 'dll_cover_main', 'cover', '-V'),
    @('pkg\tool\windows_amd64\export.dll', 'dll_export_main', 'export', '-V'),
    @('pkg\tool\windows_amd64\fix.dll', 'dll_fix_main', 'fix', '-V'),
    @('pkg\tool\windows_amd64\vet.dll', 'dll_vet_main', 'vet', '-V')
)
foreach ($case in $ffiCases) {
    $library = Join-Path $root $case[0]
    $toolArguments = $case[2..($case.Length - 1)]
    & $ffi $library $case[1] @toolArguments
    if ($LASTEXITCODE -ne 0) { throw "direct FFI invocation of $($case[1]) failed with exit code $LASTEXITCODE" }
}

# Exercise the native launcher itself with a tiny DLL so argument forwarding
# and exact exit-status propagation are independent of the Go commands above.
$probeDir = Join-Path $work 'launcher-probe'
New-Item -ItemType Directory -Path $probeDir -Force | Out-Null
$probeSource = Join-Path $probeDir 'probe.c'
@'
#include <stdio.h>
#include <string.h>

__declspec(dllexport) int __cdecl dll_probe_main(int argc, char **argv) {
    if (argc != 3 || argv[1] == NULL || argv[2] == NULL) return 91;
    if (strcmp(argv[1], "argument with spaces") != 0 || strcmp(argv[2], "tail") != 0) return 92;
    puts("launcher-probe-ok");
    return 23;
}
'@ | Set-Content -Encoding ascii $probeSource
$probeLauncher = Join-Path $probeDir 'probe.exe'
$probeLibrary = Join-Path $probeDir 'probe.dll'
$probePython = Join-Path $probeDir 'probe.py'
$probeBatch = Join-Path $probeDir 'probe.bat'
& $gcc '-municode' '-Os' '-static' '-o' $probeLauncher (Join-Path $PSScriptRoot 'launcher.c')
if ($LASTEXITCODE -ne 0) { throw 'could not build the native launcher probe' }
& $gcc '-shared' '-o' $probeLibrary $probeSource
if ($LASTEXITCODE -ne 0) { throw 'could not build the launcher probe DLL' }
$probeOutput = & $probeLauncher 'argument with spaces' 'tail'
if ($LASTEXITCODE -ne 23 -or $probeOutput.Trim() -ne 'launcher-probe-ok') {
    throw "launcher did not preserve arguments/output/exit status (exit $LASTEXITCODE, output '$probeOutput')"
}
# GitHub's pwsh wrapper returns the last native program's status after the
# script completes, so clear the deliberately nonzero probe status on success.
$global:LASTEXITCODE = 0

Copy-Item (Join-Path $PSScriptRoot 'python_launcher.py') $probePython
$pythonOutput = & $python $probePython 'argument with spaces' 'tail'
if ($LASTEXITCODE -ne 23 -or $pythonOutput.Trim() -ne 'launcher-probe-ok') {
    throw "Python launcher did not preserve arguments/output/exit status (exit $LASTEXITCODE, output '$pythonOutput')"
}
$global:LASTEXITCODE = 0

Copy-Item (Join-Path $PSScriptRoot 'windows_launcher.bat') $probeBatch
$batchOutput = & $probeBatch 'argument with spaces' 'tail'
if ($LASTEXITCODE -ne 23 -or $batchOutput.Trim() -ne 'launcher-probe-ok') {
    throw "Batch launcher did not preserve arguments/output/exit status (exit $LASTEXITCODE, output '$batchOutput')"
}
$global:LASTEXITCODE = 0

Write-Host 'Windows amd64 shared-toolchain smoke tests passed.'
