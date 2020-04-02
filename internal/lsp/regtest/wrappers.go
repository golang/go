// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regtest

import (
	"golang.org/x/tools/internal/lsp/fake"
	"golang.org/x/tools/internal/lsp/protocol"
)

// RemoveFileFromWorkspace deletes a file on disk but does nothing in the
// editor. It calls t.Fatal on any error.
func (e *Env) RemoveFileFromWorkspace(name string) {
	e.T.Helper()
	if err := e.W.RemoveFile(e.Ctx, name); err != nil {
		e.T.Fatal(err)
	}
}

// ReadWorkspaceFile reads a file from the workspace, calling t.Fatal on any
// error.
func (e *Env) ReadWorkspaceFile(name string) string {
	e.T.Helper()
	content, err := e.W.ReadFile(name)
	if err != nil {
		e.T.Fatal(err)
	}
	return content
}

// OpenFile opens a file in the editor, calling t.Fatal on any error.
func (e *Env) OpenFile(name string) {
	e.T.Helper()
	if err := e.E.OpenFile(e.Ctx, name); err != nil {
		e.T.Fatal(err)
	}
}

// CreateBuffer creates a buffer in the editor, calling t.Fatal on any error.
func (e *Env) CreateBuffer(name string, content string) {
	e.T.Helper()
	if err := e.E.CreateBuffer(e.Ctx, name, content); err != nil {
		e.T.Fatal(err)
	}
}

// CloseBuffer closes an editor buffer without saving, calling t.Fatal on any
// error.
func (e *Env) CloseBuffer(name string) {
	e.T.Helper()
	if err := e.E.CloseBuffer(e.Ctx, name); err != nil {
		e.T.Fatal(err)
	}
}

// EditBuffer applies edits to an editor buffer, calling t.Fatal on any error.
func (e *Env) EditBuffer(name string, edits ...fake.Edit) {
	e.T.Helper()
	if err := e.E.EditBuffer(e.Ctx, name, edits); err != nil {
		e.T.Fatal(err)
	}
}

// RegexpSearch returns the starting position of the first match for re in the
// buffer specified by name, calling t.Fatal on any error. It first searches
// for the position in open buffers, then in workspace files.
func (e *Env) RegexpSearch(name, re string) fake.Pos {
	e.T.Helper()
	pos, err := e.E.RegexpSearch(name, re)
	if err == fake.ErrUnknownBuffer {
		pos, err = e.W.RegexpSearch(name, re)
	}
	if err != nil {
		e.T.Fatalf("RegexpSearch: %v, %v", name, err)
	}
	return pos
}

// RegexpReplace replaces the first group in the first match of regexpStr with
// the replace text, calling t.Fatal on any error.
func (e *Env) RegexpReplace(name, regexpStr, replace string) {
	e.T.Helper()
	if err := e.E.RegexpReplace(e.Ctx, name, regexpStr, replace); err != nil {
		e.T.Fatalf("RegexpReplace: %v", err)
	}
}

// SaveBuffer saves an editor buffer, calling t.Fatal on any error.
func (e *Env) SaveBuffer(name string) {
	e.T.Helper()
	if err := e.E.SaveBuffer(e.Ctx, name); err != nil {
		e.T.Fatal(err)
	}
}

// GoToDefinition goes to definition in the editor, calling t.Fatal on any
// error.
func (e *Env) GoToDefinition(name string, pos fake.Pos) (string, fake.Pos) {
	e.T.Helper()
	n, p, err := e.E.GoToDefinition(e.Ctx, name, pos)
	if err != nil {
		e.T.Fatal(err)
	}
	return n, p
}

// FormatBuffer formats the editor buffer, calling t.Fatal on any error.
func (e *Env) FormatBuffer(name string) {
	e.T.Helper()
	if err := e.E.FormatBuffer(e.Ctx, name); err != nil {
		e.T.Fatal(err)
	}
}

// OrganizeImports processes the source.organizeImports codeAction, calling
// t.Fatal on any error.
func (e *Env) OrganizeImports(name string) {
	e.T.Helper()
	if err := e.E.OrganizeImports(e.Ctx, name); err != nil {
		e.T.Fatal(err)
	}
}

// ApplyQuickFixes processes the quickfix codeAction, calling t.Fatal on any error.
func (e *Env) ApplyQuickFixes(path string, diagnostics []protocol.Diagnostic) {
	e.T.Helper()
	if err := e.E.ApplyQuickFixes(e.Ctx, path, diagnostics); err != nil {
		e.T.Fatal(err)
	}
}

// CloseEditor shuts down the editor, calling t.Fatal on any error.
func (e *Env) CloseEditor() {
	e.T.Helper()
	if err := e.E.Shutdown(e.Ctx); err != nil {
		e.T.Fatal(err)
	}
	if err := e.E.Exit(e.Ctx); err != nil {
		e.T.Fatal(err)
	}
}
