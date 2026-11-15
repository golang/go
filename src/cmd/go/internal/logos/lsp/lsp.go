// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/logos/hints"
)

var CmdLSP = &base.Command{
	UsageLine: "logos lsp",
	Short:     "serve the language server protocol over stdio",
	Long: `
Lsp runs a Language Server Protocol server on standard input and output
for loGo/s and Go files. It starts gopls (which must be on PATH) and sits
between it and the client: loGo/s sources reach gopls as lowered Go
documents kept in memory, positions and URIs are mapped both ways, and the
compiler diagnostics are merged into the diagnostics gopls publishes.
Nothing is written into the workspace.

A stock LSP client needs no editor-specific code: point it at 'logos lsp'
for .lgo files and Go files alike.

See also: logos lint, logos build.
	`,
}

func Register(root *base.Command) {
	CmdLSP.Run = runLSP
	root.Commands = append(root.Commands, CmdLSP)
}

func runLSP(ctx context.Context, cmd *base.Command, args []string) {
	if len(args) != 0 {
		base.Fatalf("logos lsp: unexpected argument %q", args[0])
	}
	gopls, err := goplsCommand()
	if err != nil {
		base.Fatalf("%v", err)
	}
	exe, err := os.Executable()
	if err != nil {
		base.Fatalf("logos lsp: %v", err)
	}
	tc := newToolchain(exe)
	err = Serve(ctx, os.Stdin, os.Stdout, Options{
		Gopls:       gopls,
		Env:         goplsEnv(os.Environ(), filepath.Join(cfg.GOROOT, "bin")),
		Compiler:    tc,
		HintsJSON:   tc.Hints,
		Hints:       hints.Serve,
		CodeActions: codeActionsAdapter,
	})
	if err != nil {
		base.Fatalf("logos lsp: %v", err)
	}
}

func goplsCommand() ([]string, error) {
	path, err := exec.LookPath("gopls")
	if err != nil {
		return nil, errors.New("logos lsp: gopls not found on PATH")
	}
	return []string{path, "-mode=stdio"}, nil
}

func goplsEnv(env []string, bin string) []string {
	out := make([]string, 0, len(env)+1)
	path := bin
	for _, kv := range env {
		if rest, ok := strings.CutPrefix(kv, "PATH="); ok {
			path = bin + string(os.PathListSeparator) + rest
			continue
		}
		out = append(out, kv)
	}
	return append(out, "PATH="+path)
}
