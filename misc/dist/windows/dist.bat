:: Copyright 2012 The Go Authors. All rights reserved.
:: Use of this source code is governed by a BSD-style
:: license that can be found in the LICENSE file.
@echo off

setlocal

:: Requires Windows Installer XML (WiX), 7zip, and Mercurial (hg)

echo # Setting some variables
for /f %%i in ('hg.exe root') do set ROOT=%%i
for /f %%i in ('hg.exe id -n') do set ID=%%i
for /f "tokens=3" %%i in ('%ROOT%\bin\go.exe version') do set VER=%%i
if errorlevel 1 goto end

echo # Getting GOARCH
%ROOT%\bin\go tool dist env -w>env.bat
call env.bat
del /F /Q /S env.bat>NUL
if errorlevel 1 goto end


rmdir /S /Q go>NUL
mkdir go

echo # Cloning the Go tree
hg clone -r %ID% %ROOT% go
if errorlevel 1 goto end

rmdir /S /Q  go\.hg>NUL
del /F /Q /S go\.hgignore go\.hgtags>NUL

echo # Copying pkg, bin, and src/pkg/runtime/z*
xcopy %ROOT%\pkg                   go\pkg /V /E /Y /I
xcopy %ROOT%\bin                   go\bin /V /E /Y /I
xcopy %ROOT%\src\pkg\runtime\z*.c  go\src\pkg\runtime  /V /E /Y
xcopy %ROOT%\src\pkg\runtime\z*.go go\src\pkg\runtime  /V /E /Y
xcopy %ROOT%\src\pkg\runtime\z*.h  go\src\pkg\runtime  /V /E /T

echo # Starting zip packaging
7za a -tzip -mx=9 go.%VER%.windows-%GOARCH%.zip "go/"
if errorlevel 1 goto end


echo # Starting Go directory file harvesting
heat dir go -nologo -cg AppFiles -gg -g1 -srd -sfrag -template fragment -dr INSTALLDIR -var var.SourceDir -out AppFiles.wxs
if errorlevel 1 goto end

echo # Starting installer packaging
candle -nologo -dVersion=%VER% -dArch=%GOARCH% -dSourceDir=go installer.wxs AppFiles.wxs
light -nologo -ext WixUIExtension -ext WixUtilExtension installer.wixobj AppFiles.wixobj -o go.%VER%.windows-%GOARCH%.msi
if errorlevel 1 goto end

del /F /Q /S *.wixobj AppFiles.wxs *.wixpdb>NUL

:end
endlocal
