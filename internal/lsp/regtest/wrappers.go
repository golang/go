// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regtest

import "golang.org/x/tools/internal/lsp/fake"

// RemoveFileFromWorkspace deletes a file on disk but does nothing in the
// editor. It calls t.Fatal on any error.
func (e *Env) RemoveFileFromWorkspace(name string) {
	e.t.Helper()
	if err := e.W.RemoveFile(e.ctx, name); err != nil {
		e.t.Fatal(err)
	}
}

// ReadWorkspaceFile reads a file from the workspace, calling t.Fatal on any
// error.
func (e *Env) ReadWorkspaceFile(name string) string {
	e.t.Helper()
	content, err := e.W.ReadFile(name)
	if err != nil {
		e.t.Fatal(err)
	}
	return content
}

// OpenFile opens a file in the editor, calling t.Fatal on any error.
func (e *Env) OpenFile(name string) {
	e.t.Helper()
	if err := e.E.OpenFile(e.ctx, name); err != nil {
		e.t.Fatal(err)
	}
}

// CreateBuffer creates a buffer in the editor, calling t.Fatal on any error.
func (e *Env) CreateBuffer(name string, content string) {
	e.t.Helper()
	if err := e.E.CreateBuffer(e.ctx, name, content); err != nil {
		e.t.Fatal(err)
	}
}

// CloseBuffer closes an editor buffer without saving, calling t.Fatal on any
// error.
func (e *Env) CloseBuffer(name string) {
	e.t.Helper()
	if err := e.E.CloseBuffer(e.ctx, name); err != nil {
		e.t.Fatal(err)
	}
}

// EditBuffer applies edits to an editor buffer, calling t.Fatal on any error.
func (e *Env) EditBuffer(name string, edits ...fake.Edit) {
	e.t.Helper()
	if err := e.E.EditBuffer(e.ctx, name, edits); err != nil {
		e.t.Fatal(err)
	}
}

// SaveBuffer saves an editor buffer, calling t.Fatal on any error.
func (e *Env) SaveBuffer(name string) {
	e.t.Helper()
	if err := e.E.SaveBuffer(e.ctx, name); err != nil {
		e.t.Fatal(err)
	}
}

// GoToDefinition goes to definition in the editor, calling t.Fatal on any
// error.
func (e *Env) GoToDefinition(name string, pos fake.Pos) (string, fake.Pos) {
	e.t.Helper()
	n, p, err := e.E.GoToDefinition(e.ctx, name, pos)
	if err != nil {
		e.t.Fatal(err)
	}
	return n, p
}

// FormatBuffer formats the editor buffer, calling t.Fatal on any error.
func (e *Env) FormatBuffer(name string) {
	e.t.Helper()
	if err := e.E.FormatBuffer(e.ctx, name); err != nil {
		e.t.Fatal(err)
	}
}

// OrganizeImports processes the source.organizeImports codeAction, calling
// t.Fatal on any error.
func (e *Env) OrganizeImports(name string) {
	e.t.Helper()
	if err := e.E.OrganizeImports(e.ctx, name); err != nil {
		e.t.Fatal(err)
	}
}

// CloseEditor shuts down the editor, calling t.Fatal on any error.
func (e *Env) CloseEditor() {
	e.t.Helper()
	if err := e.E.Shutdown(e.ctx); err != nil {
		e.t.Fatal(err)
	}
	if err := e.E.Exit(e.ctx); err != nil {
		e.t.Fatal(err)
	}
}
