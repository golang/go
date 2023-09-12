// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.19
// +build go1.19

package telemetry

import (
	"fmt"

	"golang.org/x/telemetry"
	"golang.org/x/telemetry/counter"
	"golang.org/x/telemetry/upload"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
)

// Mode calls x/telemetry.Mode.
func Mode() string {
	return telemetry.Mode()
}

// SetMode calls x/telemetry.SetMode.
func SetMode(mode string) error {
	return telemetry.SetMode(mode)
}

// Start starts telemetry instrumentation.
func Start() {
	counter.Open()
	// upload only once at startup, hoping that users restart gopls often.
	go upload.Run(nil)
}

// RecordClientInfo records gopls client info.
func RecordClientInfo(params *protocol.ParamInitialize) {
	client := "gopls/client:other"
	if params != nil && params.ClientInfo != nil {
		switch params.ClientInfo.Name {
		case "Visual Studio Code":
			client = "gopls/client:vscode"
		case "Visual Studio Code - Insiders":
			client = "gopls/client:vscode-insiders"
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
		default:
			// at least accumulate the client name locally
			counter.New(fmt.Sprintf("gopls/client-other:%s", params.ClientInfo.Name)).Inc()
			// but also record client:other
		}
	}
	counter.Inc(client)
}

// RecordViewGoVersion records the Go minor version number (1.x) used for a view.
func RecordViewGoVersion(x int) {
	if x < 0 {
		return
	}
	name := fmt.Sprintf("gopls/goversion:1.%d", x)
	counter.Inc(name)
}

// AddForwardedCounters adds the given counters on behalf of clients.
// Names and values must have the same length.
func AddForwardedCounters(names []string, values []int64) {
	for i, n := range names {
		v := values[i]
		if n == "" || v < 0 {
			continue // Should we report an error? Who is the audience?
		}
		counter.Add("fwd/"+n, v)
	}
}
