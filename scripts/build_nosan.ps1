# The sanitizer-OFF configuration in its own build dir (build_nosan):
# NANOSM_SANITIZER=OFF compiles src/sanitizer.hpp's Sanitizer to no-ops.
# Its ctest runs the ordinary targets against the OFF library; the unit tests
# that expect a sanitizer panic are skipped (GTEST_SKIP). The default
# build (scripts/build.ps1) already builds an OFF twin and proves it
# byte-identical; this script exists to exercise the option itself.

# -Fast: run only the fast tests (ctest -LE slow) for iteration inside a
# task. The default runs every test: it is the verification gate.
param([switch]$Fast)

$ErrorActionPreference = 'Stop'

$compiler = 'C:\msys64\mingw64\bin\g++.exe'
if (-not (Test-Path -LiteralPath $compiler)) {
    $found = Get-Command g++ -ErrorAction SilentlyContinue
    if (-not $found) { throw 'g++ was not found' }
    $compiler = $found.Source
}

$cmake = Get-Command cmake -ErrorAction SilentlyContinue
if ($cmake) {
    $cmake = $cmake.Source
} else {
    $cmake = 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
    if (-not (Test-Path -LiteralPath $cmake)) { throw 'cmake was not found' }
}

$ninja = Get-Command ninja -ErrorAction SilentlyContinue
if ($ninja) {
    $ninja = $ninja.Source
} else {
    $ninja = 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe'
    if (-not (Test-Path -LiteralPath $ninja)) { throw 'ninja was not found' }
}

$env:PATH = "$(Split-Path -Parent $compiler);$(Split-Path -Parent $ninja);$env:PATH"
New-Item -ItemType Directory -Force -Path build_nosan | Out-Null
$env:TEMP = (Resolve-Path -LiteralPath build_nosan).Path
$env:TMP = $env:TEMP
$compilerCmake = $compiler -replace '\\', '/'
$ninjaCmake = $ninja -replace '\\', '/'

& $cmake -S . -B build_nosan -G Ninja "-DNANOSM_SANITIZER=OFF" "-DCMAKE_MAKE_PROGRAM=$ninjaCmake" "-DCMAKE_CXX_COMPILER=$compilerCmake"
if ($LASTEXITCODE -ne 0) { throw 'cmake configure failed' }

& $cmake --build build_nosan
if ($LASTEXITCODE -ne 0) { throw 'cmake build failed' }

# Every unit test is its own ctest entry, so the run spreads over the cores.
$ctest = Join-Path (Split-Path -Parent $cmake) 'ctest.exe'
$ctestArgs = @('--test-dir', 'build_nosan', '-j', [Environment]::ProcessorCount, '--output-on-failure')
if ($Fast) { $ctestArgs += @('-LE', 'slow') }
& $ctest @ctestArgs
if ($LASTEXITCODE -ne 0) { throw 'tests failed' }
