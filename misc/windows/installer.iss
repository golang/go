;; Copyright 2010 The Go Authors.  All rights reserved.
;; Use of this source code is governed by a BSD-style
;; license that can be found in the LICENSE file.

[Setup]
;; (To generate a new GUID, click Tools | Generate GUID inside the IDE.)
AppId={{1AE268D9-FAE4-4EF8-AAE9-3B1B27D604F0}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=golang-nuts@googlegroups.com
AppPublisherURL=http://www.golang.org
DefaultDirName={sd}\Go
DisableDirPage=yes
DefaultGroupName={#AppName}
AllowNoIcons=yes
OutputBaseFilename={#AppNameLower}win{#AppVersion}_installer
Compression=lzma2/max
SolidCompression=yes
ChangesEnvironment=true
OutputDir=.

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "basque"; MessagesFile: "compiler:Languages\Basque.isl"
Name: "brazilianportuguese"; MessagesFile: "compiler:Languages\BrazilianPortuguese.isl"
Name: "catalan"; MessagesFile: "compiler:Languages\Catalan.isl"
Name: "czech"; MessagesFile: "compiler:Languages\Czech.isl"
Name: "danish"; MessagesFile: "compiler:Languages\Danish.isl"
Name: "dutch"; MessagesFile: "compiler:Languages\Dutch.isl"
Name: "finnish"; MessagesFile: "compiler:Languages\Finnish.isl"
Name: "french"; MessagesFile: "compiler:Languages\French.isl"
Name: "german"; MessagesFile: "compiler:Languages\German.isl"
Name: "hebrew"; MessagesFile: "compiler:Languages\Hebrew.isl"
Name: "hungarian"; MessagesFile: "compiler:Languages\Hungarian.isl"
Name: "italian"; MessagesFile: "compiler:Languages\Italian.isl"
Name: "japanese"; MessagesFile: "compiler:Languages\Japanese.isl"
Name: "norwegian"; MessagesFile: "compiler:Languages\Norwegian.isl"
Name: "polish"; MessagesFile: "compiler:Languages\Polish.isl"
Name: "portuguese"; MessagesFile: "compiler:Languages\Portuguese.isl"
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "slovak"; MessagesFile: "compiler:Languages\Slovak.isl"
Name: "slovenian"; MessagesFile: "compiler:Languages\Slovenian.isl"
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"

[Files]
Source: ".\go\*"; DestDir: "{sd}\Go"; Flags: ignoreversion recursesubdirs createallsubdirs

[Registry]
;Root: HKLM; Subkey: "SYSTEM\CurrentControlSet\Control\Session Manager\Environment"; ValueType: string; ValueName: "GOARCH"; ValueData: "386"; Flags: uninsdeletevalue
;Root: HKLM; Subkey: "SYSTEM\CurrentControlSet\Control\Session Manager\Environment"; ValueType: string; ValueName: "GOOS"; ValueData: "windows"; Flags: uninsdeletevalue
Root: HKLM; Subkey: "SYSTEM\CurrentControlSet\Control\Session Manager\Environment"; ValueType: string; ValueName: "GOBIN"; ValueData: "{sd}/Go/bin"; Flags: uninsdeletevalue
Root: HKLM; Subkey: "SYSTEM\CurrentControlSet\Control\Session Manager\Environment"; ValueType: string; ValueName: "GOROOT"; ValueData: "{sd}/Go"; Flags: uninsdeletevalue
Root: HKLM; Subkey: "SYSTEM\CurrentControlSet\Control\Session Manager\Environment"; ValueType: expandsz; ValueName: "Path";  ValueData: "{olddata};{sd}/Go/bin"; Check: PathCheck('{sd}/Go/bin')

;[Tasks]
;Name: AddToPath; Description: "&Adding Go's bin directory to your environment's search path. This allows the tools to be run from a shell without having to include the installation path as part of the command.";

[Icons]
;Name: "{group}\{cm:UninstallProgram,Go}"; Filename: {uninstallexe}
Name: "{group}\Uninstall Go"; Filename: "{uninstallexe}"

[Code]
function PathCheck(Param: string): Boolean;
var
    OrigPath: String;
    Index: Integer;
begin
    // check for an empty path
    if not RegQueryStringValue(HKEY_LOCAL_MACHINE, 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment', 'Path', OrigPath)
    then begin
        Result := True;
        exit;
    end;

    // Pos returns 0 if not found
    Index := Pos(';' + Param + ';', ';' + OrigPath + ';');

    if (IsUninstaller() = True) AND (Index > 0) then begin
        Delete(OrigPath, Index, Length(Param));

        // remove orphaned semicolon if necessary
        if (Length(OrigPath) >= Index) AND (Copy(OrigPath, Index, 1) = ';') then begin
            Delete(OrigPath, Index, 1);
        end;

        RegWriteStringValue(HKEY_LOCAL_MACHINE, 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment', 'Path', OrigPath);
    end;

    // during installation, the check in the Registry
    // section wants a Boolean value
    Result := Index = 0;
end;

function InitializeUninstall(): Boolean;
begin
    PathCheck(ExpandConstant('{sd}/Go/bin'));
    Result := True;
end;
