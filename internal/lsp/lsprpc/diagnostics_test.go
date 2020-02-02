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

	diagnostics <-chan *protocol.PublishDiagnosticsParams
}

func setupEnv(t *testing.T) (context.Context, testEnvironment, func()) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)

	ws, err := fake.NewWorkspace("get-diagnostics", []byte(exampleProgram))
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
	diags := make(chan *protocol.PublishDiagnosticsParams, 10)
	editor.Client().OnDiagnostics(func(_ context.Context, params *protocol.PublishDiagnosticsParams) error {
		diags <- params
		return nil
	})
	cleanup := func() {
		cancel()
		ts.Close()
		ws.Close()
	}
	return ctx, testEnvironment{
		editor:      editor,
		ws:          ws,
		ts:          ts,
		diagnostics: diags,
	}, cleanup
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
	ctx, env, cleanup := setupEnv(t)
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
	params := awaitDiagnostics(ctx, t, env.diagnostics)
	if err := checkDiagnosticLocation(params, env.ws.URI("main.go"), 5, 5); err != nil {
		t.Fatal(err)
	}
}

func TestSimultaneousEdits(t *testing.T) {
	t.Parallel()
	ctx, env, cleanup := setupEnv(t)
	defer cleanup()

	// Set up a second editor session connected to the same server, using the
	// same workspace.
	conn2 := env.ts.Connect(ctx)
	editor2, err := fake.NewConnectedEditor(ctx, env.ws, conn2)
	if err != nil {
		t.Fatal(err)
	}
	diags2 := make(chan *protocol.PublishDiagnosticsParams, 10)
	editor2.Client().OnDiagnostics(func(_ context.Context, params *protocol.PublishDiagnosticsParams) error {
		diags2 <- params
		return nil
	})

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
	params1 := awaitDiagnostics(ctx, t, env.diagnostics)
	params2 := awaitDiagnostics(ctx, t, diags2)
	if err := checkDiagnosticLocation(params1, env.ws.URI("main.go"), 5, 5); err != nil {
		t.Fatal(err)
	}
	if err := checkDiagnosticLocation(params2, env.ws.URI("main.go"), 7, 0); err != nil {
		t.Fatal(err)
	}
}

func awaitDiagnostics(ctx context.Context, t *testing.T, diags <-chan *protocol.PublishDiagnosticsParams) *protocol.PublishDiagnosticsParams {
	t.Helper()
	select {
	case <-ctx.Done():
		panic(ctx.Err())
	case d := <-diags:
		return d
	}
}

const brokenFile = `package main

const Foo = "abc
`

func TestDiagnosticErrorInNewFile(t *testing.T) {
	t.Parallel()
	ctx, env, cleanup := setupEnv(t)
	defer cleanup()

	if err := env.editor.CreateBuffer(ctx, "broken.go", brokenFile); err != nil {
		t.Fatal(err)
	}
	params := awaitDiagnostics(ctx, t, env.diagnostics)
	if got, want := params.URI, env.ws.URI("broken.go"); got != want {
		t.Fatalf("got diagnostics for URI %q, want %q", got, want)
	}
}
