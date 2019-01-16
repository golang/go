# gopls testing extension

An extension for debugging the Go Language Server provided by 
https://golang.org/x/tools/cmd/gopls. The code for this extension comes from
a combination of 
https://github.com/Microsoft/vscode-extension-samples/blob/master/lsp-sample
and https://github.com/Microsoft/vscode-go. 

## Features

* Diagnostics (on file change)
* Completion (Ctrl + Space)
* Jump to definition (F12 or right-click -> Go to Definition)
* Signature help (Ctrl + Shift + Space)

## Installation

To package the extension, run `vsce package` from this directory. To install
the extension, navigate to the "Extensions" panel in VSCode, and select
"Install from VSIX..." from the menu in the top right corner. Choose the 
`gopls-1.0.0.vsix file` and reload VSCode.
