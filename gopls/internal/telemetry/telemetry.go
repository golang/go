// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package telemetry

import (
	"os"

	"golang.org/x/telemetry/counter"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
)

// Start starts telemetry instrumentation.
func Start() {
	if os.Getenv("GOPLS_TELEMETRY_EXP") != "" {
		counter.Open()
		// TODO: add upload logic.
	}
}

// RecordClientInfo records gopls client info.
func RecordClientInfo(params *protocol.ParamInitialize) {
	client := "gopls/client:other"
	if params != nil && params.ClientInfo != nil {
		switch params.ClientInfo.Name {
		case "Visual Studio Code":
			client = "gopls/client:vscode"
		case "VSCodium":
			client = "gopls/client:vscodium"
		case "code-server":
			// https://github.com/coder/code-server/blob/3cb92edc76ecc2cfa5809205897d93d4379b16a6/ci/build/build-vscode.sh#L19
			client = "gopls/client:code-server"
		case "Eglot":
			// https://lists.gnu.org/archive/html/bug-gnu-emacs/2023-03/msg00954.html
			client = "gopls/client:eglot"
		case "govim":
			// https://github.com/govim/govim/pull/1189
			client = "gopls/client:govim"
		case "Neovim":
			// https://github.com/neovim/neovim/blob/42333ea98dfcd2994ee128a3467dfe68205154cd/runtime/lua/vim/lsp.lua#L1361
			client = "gopls/client:neovim"
		case "coc.nvim":
			// https://github.com/neoclide/coc.nvim/blob/3dc6153a85ed0f185abec1deb972a66af3fbbfb4/src/language-client/client.ts#L994
			client = "gopls/client:coc.nvim"
		case "Sublime Text LSP":
			// https://github.com/sublimelsp/LSP/blob/e608f878e7e9dd34aabe4ff0462540fadcd88fcc/plugin/core/sessions.py#L493
			client = "gopls/client:sublimetext"
		}
	}
	counter.Inc(client)
}
