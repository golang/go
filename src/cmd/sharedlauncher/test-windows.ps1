$ErrorActionPreference = 'Stop'

$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$go = Join-Path $root 'bin\go.exe'
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

Write-Host 'Windows amd64 shared-toolchain smoke tests passed.'
