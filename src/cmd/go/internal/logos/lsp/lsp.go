// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package lsp implements logos lsp: a Language Server Protocol proxy that
// gives stock LSP clients loGo/s and Go support through gopls.
//
// # Model
//
// The proxy speaks JSON-RPC over stdio to the client and starts
// "gopls -mode=stdio" as its server. gopls knows Go only, so every loGo/s
// file reaches it as the Go text the compiler lowers it to
// (compile mode -logos.eject, the text of logos eject). The proxy keeps
// that lowered text in gopls as an open document named like the source
// with its extension replaced by .go (a.lgo becomes a.go in the same
// directory), so nothing is ever written into the workspace: the lowered
// documents live in gopls's memory, and the scratch files of a compile
// live under the system temporary directory. Go packages stay on disk for
// gopls.
//
// The compiler also writes an edit map beside each lowered file, the list
// of byte spans of the source that the lowering replaced. The proxy builds
// a position map from it. Text the lowering copied unchanged inside a
// replaced span, such as the operand of a ? statement, still maps
// exactly; generated text maps to the source that produced it, so a
// position inside the three lines a ? statement expands to maps to the
// statement. Positions use UTF-16 code units on both sides; the proxy
// removes general.positionEncodings from the client's capabilities, so
// gopls answers in UTF-16 too.
//
// A request on a loGo/s document is forwarded with its URI and every
// position and range mapped to the lowered document, and the answer is
// mapped back. Locations that name other lowered documents map to their
// sources, so definition, references, and rename cross files. A request on
// a loGo/s document whose lowering is not current, because the package
// does not compile at the client's text, gets a null result, except for
// textDocument/codeAction, which still answers the compiler autofix
// candidates of the source itself. Edits of the
// client update the proxy's text of the document at once; the proxy
// recompiles before it forwards the next request, after the client stops
// typing for the debounce delay, and on open and save.
//
// What gopls reports about text the lowering wrote, and the user never
// did, is removed: inlay hints inside generated text (the type of the error
// variable of a ? statement), and completions and document symbols named
// with the reserved _logos_ prefix.
//
// # Limits
//
// Test files are not lowered: the compiler lowers the files of each
// package's build, not its _test.lgo files, so a request on an open
// _test.lgo file gets a null result, except for textDocument/codeAction,
// which answers the compiler autofix candidates all the same, while the _test.go files of Go
// packages work as any Go file does. Diagnostics reach only documents the
// client has open.
//
// # Mixed workspaces
//
// The overlay holds lowered loGo/s files only. A Go document is forwarded
// to gopls unchanged, but the proxy first lowers the loGo/s packages it
// imports and opens them in gopls, so a definition request on a loGo/s
// symbol imported by a Go file resolves through the lowered document and
// answers with the position in the .lgo file. A definition request on a
// Go symbol imported by a loGo/s file resolves through gopls on the real
// .go files and answers with the position in the .go file.
//
// # Diagnostics
//
// The proxy publishes the diagnostics of gopls, mapped to the sources,
// together with the compiler diagnostics of the package (logos lint
// --json-diagnostics, SPEC Sec 16), without duplicates. It removes
// diagnosticProvider from the server capabilities, so clients take
// diagnostics through publishDiagnostics, where the merge happens. Only
// open loGo/s documents get diagnostics.
//
// # Hints hook (M21-S2)
//
// Inlay hints and hover previews come from compiler hints, not from gopls.
// The proxy exposes them through three optional Options fields:
//
//   - HintsJSON runs the compiler in hints mode for one package and returns
//     the hints JSON. runLSP sets it to the toolchain runner.
//   - Hints is a HintsFunc. For each textDocument/inlayHint and
//     textDocument/hover request the proxy calls it with the method, the
//     params as the client sent them, and the hints JSON of the document's
//     package, and merges the answer into gopls's: inlay hints are
//     appended; hover text is appended to gopls's hover. M21-S2's package
//     cmd/go/internal/logos/hints provides it as hints.Serve. The wiring
//     is the one line "Hints: hints.Serve" in the Options literal of
//     runLSP.
//   - CodeActions is a CodeActionsFunc that turns the compiler autofix
//     candidates into code actions; runLSP sets it to codeActionsAdapter,
//     the adapter around hints.CodeActions, in the same literal.
//
// With both hooks nil the proxy serves what gopls and the compiler
// diagnostics report.
//
// # Lifetime
//
// Serve returns when the client sends exit, closes its output (after the
// requests in flight are answered), or gopls dies. It stops gopls and all
// of its own background work first, and closes the client's input when that
// input is an io.Closer.
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

// CmdLSP implements the logos lsp command.
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

// Register adds the lsp subcommand to the logos root command.
func Register(root *base.Command) {
	CmdLSP.Run = runLSP
	root.Commands = append(root.Commands, CmdLSP)
}

// runLSP serves the client on standard input and output until it is done.
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
	tc, err := newToolchain(exe)
	if err != nil {
		base.Fatalf("logos lsp: %v", err)
	}
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

// goplsCommand returns the command that runs gopls over stdio, or the
// error that tells the user gopls is missing.
func goplsCommand() ([]string, error) {
	path, err := exec.LookPath("gopls")
	if err != nil {
		return nil, errors.New("logos lsp: gopls not found on PATH")
	}
	return []string{path, "-mode=stdio"}, nil
}

// goplsEnv returns env with bin first on PATH, so the go command gopls
// runs is the one of this toolchain: it resolves the logos/rt import of a
// lowered file from the same GOROOT that lowered it.
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
