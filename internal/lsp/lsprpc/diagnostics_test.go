// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsprpc

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"golang.org/x/tools/internal/jsonrpc2/servertest"
	"golang.org/x/tools/internal/lsp/cache"
	"golang.org/x/tools/internal/lsp/fake"
	"golang.org/x/tools/internal/lsp/protocol"
)

const exampleProgram = `
-- go.mod --
module mod

go 1.12
-- main.go --
package main

import "fmt"

func main() {
	fmt.Println("Hello World.")
}`

type testEnvironment struct {
	editor *fake.Editor
	ws     *fake.Workspace
	ts     *servertest.Server

	dw diagnosticsWatcher
}

func setupEnv(t *testing.T, txt string) (context.Context, testEnvironment, func()) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)

	ws, err := fake.NewWorkspace("lsprpc", []byte(txt))
	if err != nil {
		t.Fatal(err)
	}
	ss := NewStreamServer(cache.New(nil), false)
	ts := servertest.NewServer(ctx, ss)
	cc := ts.Connect(ctx)

	editor, err := fake.NewConnectedEditor(ctx, ws, cc)
	if err != nil {
		t.Fatal(err)
	}
	dw := newDiagWatcher(ws)
	editor.Client().OnDiagnostics(dw.onDiagnostics)
	cleanup := func() {
		cancel()
		ts.Close()
		ws.Close()
	}
	return ctx, testEnvironment{
		editor: editor,
		ws:     ws,
		ts:     ts,
		dw:     dw,
	}, cleanup
}

type diagnosticsWatcher struct {
	diagnostics chan *protocol.PublishDiagnosticsParams
	ws          *fake.Workspace
}

func newDiagWatcher(ws *fake.Workspace) diagnosticsWatcher {
	return diagnosticsWatcher{
		// Allow an arbitrarily large buffer, as we should never want onDiagnostics
		// to block.
		diagnostics: make(chan *protocol.PublishDiagnosticsParams, 1000),
		ws:          ws,
	}
}

func (w diagnosticsWatcher) onDiagnostics(_ context.Context, p *protocol.PublishDiagnosticsParams) error {
	w.diagnostics <- p
	return nil
}

func (w diagnosticsWatcher) await(ctx context.Context, expected ...string) (map[string]*protocol.PublishDiagnosticsParams, error) {
	expectedSet := make(map[string]bool)
	for _, e := range expected {
		expectedSet[e] = true
	}
	got := make(map[string]*protocol.PublishDiagnosticsParams)
	for len(got) < len(expectedSet) {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case d := <-w.diagnostics:
			pth := w.ws.URIToPath(d.URI)
			if expectedSet[pth] {
				got[pth] = d
			}
		}
	}
	return got, nil
}

func checkDiagnosticLocation(params *protocol.PublishDiagnosticsParams, filename string, line, col int) error {
	if got, want := params.URI, filename; got != want {
		return fmt.Errorf("got diagnostics for URI %q, want %q", got, want)
	}
	if len(params.Diagnostics) == 0 {
		return errors.New("empty diagnostics")
	}
	diag := params.Diagnostics[0]
	if diag.Range.Start.Line != float64(line) || diag.Range.Start.Character != float64(col) {
		return fmt.Errorf("Diagnostics[0].Range.Start = %v, want (5,5)", diag.Range.Start)
	}
	return nil
}

func TestDiagnosticErrorInEditedFile(t *testing.T) {
	t.Parallel()
	ctx, env, cleanup := setupEnv(t, exampleProgram)
	defer cleanup()

	// Deleting the 'n' at the end of Println should generate a single error
	// diagnostic.
	edits := []fake.Edit{
		{
			Start: fake.Pos{Line: 5, Column: 11},
			End:   fake.Pos{Line: 5, Column: 12},
			Text:  "",
		},
	}
	if err := env.editor.OpenFile(ctx, "main.go"); err != nil {
		t.Fatal(err)
	}
	if err := env.editor.EditBuffer(ctx, "main.go", edits); err != nil {
		t.Fatal(err)
	}
	diags, err := env.dw.await(ctx, "main.go")
	if err != nil {
		t.Fatal(err)
	}
	if err := checkDiagnosticLocation(diags["main.go"], env.ws.URI("main.go"), 5, 5); err != nil {
		t.Fatal(err)
	}
}

func TestSimultaneousEdits(t *testing.T) {
	t.Parallel()
	ctx, env, cleanup := setupEnv(t, exampleProgram)
	defer cleanup()

	// Set up a second editor session connected to the same server, using the
	// same workspace.
	conn2 := env.ts.Connect(ctx)
	editor2, err := fake.NewConnectedEditor(ctx, env.ws, conn2)
	if err != nil {
		t.Fatal(err)
	}
	dw2 := newDiagWatcher(env.ws)
	editor2.Client().OnDiagnostics(dw2.onDiagnostics)
	// In editor #1, break fmt.Println as before.
	edits1 := []fake.Edit{{
		Start: fake.Pos{Line: 5, Column: 11},
		End:   fake.Pos{Line: 5, Column: 12},
		Text:  "",
	}}
	if err := env.editor.OpenFile(ctx, "main.go"); err != nil {
		t.Fatal(err)
	}
	if err := env.editor.EditBuffer(ctx, "main.go", edits1); err != nil {
		t.Fatal(err)
	}

	// In editor #2 remove the closing brace.
	edits2 := []fake.Edit{{
		Start: fake.Pos{Line: 6, Column: 0},
		End:   fake.Pos{Line: 6, Column: 1},
		Text:  "",
	}}
	if err := editor2.OpenFile(ctx, "main.go"); err != nil {
		t.Fatal(err)
	}
	if err := editor2.EditBuffer(ctx, "main.go", edits2); err != nil {
		t.Fatal(err)
	}
	diags1, err := env.dw.await(ctx, "main.go")
	if err != nil {
		t.Fatal(err)
	}
	diags2, err := dw2.await(ctx, "main.go")
	if err != nil {
		t.Fatal(err)
	}
	if err := checkDiagnosticLocation(diags1["main.go"], env.ws.URI("main.go"), 5, 5); err != nil {
		t.Fatal(err)
	}
	if err := checkDiagnosticLocation(diags2["main.go"], env.ws.URI("main.go"), 7, 0); err != nil {
		t.Fatal(err)
	}
}

const brokenFile = `package main

const Foo = "abc
`

func TestDiagnosticErrorInNewFile(t *testing.T) {
	t.Parallel()
	ctx, env, cleanup := setupEnv(t, exampleProgram)
	defer cleanup()

	if err := env.editor.CreateBuffer(ctx, "broken.go", brokenFile); err != nil {
		t.Fatal(err)
	}
	_, err := env.dw.await(ctx, "broken.go")
	if err != nil {
		t.Fatal(err)
	}
}

// badPackage contains a duplicate definition of the 'a' const.
const badPackage = `
-- go.mod --
module mod

go 1.12
-- a.go --
package consts

const a = 1
-- b.go --
package consts

const a = 2
`

func TestDiagnosticClearingOnEdit(t *testing.T) {
	t.Parallel()
	ctx, env, cleanup := setupEnv(t, badPackage)
	defer cleanup()

	if err := env.editor.OpenFile(ctx, "b.go"); err != nil {
		t.Fatal(err)
	}
	_, err := env.dw.await(ctx, "a.go", "b.go")
	if err != nil {
		t.Fatal(err)
	}

	// In editor #2 remove the closing brace.
	edits := []fake.Edit{{
		Start: fake.Pos{Line: 2, Column: 6},
		End:   fake.Pos{Line: 2, Column: 7},
		Text:  "b",
	}}
	if err := env.editor.EditBuffer(ctx, "b.go", edits); err != nil {
		t.Fatal(err)
	}
	diags, err := env.dw.await(ctx, "a.go", "b.go")
	if err != nil {
		t.Fatal(err)
	}
	for pth, d := range diags {
		if len(d.Diagnostics) != 0 {
			t.Errorf("non-empty diagnostics for %q", pth)
		}
	}
}

func TestDiagnosticClearingOnDelete(t *testing.T) {
	t.Skip("skipping due to golang.org/issues/37049")

	t.Parallel()
	ctx, env, cleanup := setupEnv(t, badPackage)
	defer cleanup()

	if err := env.editor.OpenFile(ctx, "a.go"); err != nil {
		t.Fatal(err)
	}
	_, err := env.dw.await(ctx, "a.go", "b.go")
	if err != nil {
		t.Fatal(err)
	}
	env.ws.RemoveFile(ctx, "b.go")

	// TODO(golang.org/issues/37049): here we only get diagnostics for a.go.
	diags, err := env.dw.await(ctx, "a.go", "b.go")
	if err != nil {
		t.Fatal(err)
	}
	for pth, d := range diags {
		if len(d.Diagnostics) != 0 {
			t.Errorf("non-empty diagnostics for %q", pth)
		}
	}
}
