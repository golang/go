// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regtest

import (
	"errors"
	"io"
	"testing"

	"golang.org/x/tools/internal/lsp/fake"
	"golang.org/x/tools/internal/lsp/protocol"
)

func (e *Env) ChangeFilesOnDisk(events []fake.FileEvent) {
	e.T.Helper()
	if err := e.Sandbox.Workdir.ChangeFilesOnDisk(e.Ctx, events); err != nil {
		e.T.Fatal(err)
	}
}

// RemoveWorkspaceFile deletes a file on disk but does nothing in the
// editor. It calls t.Fatal on any error.
func (e *Env) RemoveWorkspaceFile(name string) {
	e.T.Helper()
	if err := e.Sandbox.Workdir.RemoveFile(e.Ctx, name); err != nil {
		e.T.Fatal(err)
	}
}

// ReadWorkspaceFile reads a file from the workspace, calling t.Fatal on any
// error.
func (e *Env) ReadWorkspaceFile(name string) string {
	e.T.Helper()
	content, err := e.Sandbox.Workdir.ReadFile(name)
	if err != nil {
		e.T.Fatal(err)
	}
	return content
}

// WriteWorkspaceFile writes a file to disk but does nothing in the editor.
// It calls t.Fatal on any error.
func (e *Env) WriteWorkspaceFile(name, content string) {
	e.T.Helper()
	if err := e.Sandbox.Workdir.WriteFile(e.Ctx, name, content); err != nil {
		e.T.Fatal(err)
	}
}

// WriteWorkspaceFiles deletes a file on disk but does nothing in the
// editor. It calls t.Fatal on any error.
func (e *Env) WriteWorkspaceFiles(files map[string]string) {
	e.T.Helper()
	if err := e.Sandbox.Workdir.WriteFiles(e.Ctx, files); err != nil {
		e.T.Fatal(err)
	}
}

// OpenFile opens a file in the editor, calling t.Fatal on any error.
func (e *Env) OpenFile(name string) {
	e.T.Helper()
	if err := e.Editor.OpenFile(e.Ctx, name); err != nil {
		e.T.Fatal(err)
	}
}

func (e *Env) OpenFileWithContent(name, content string) {
	e.T.Helper()
	if err := e.Editor.OpenFileWithContent(e.Ctx, name, content); err != nil {
		e.T.Fatal(err)
	}
}

// CreateBuffer creates a buffer in the editor, calling t.Fatal on any error.
func (e *Env) CreateBuffer(name string, content string) {
	e.T.Helper()
	if err := e.Editor.CreateBuffer(e.Ctx, name, content); err != nil {
		e.T.Fatal(err)
	}
}

// CloseBuffer closes an editor buffer without saving, calling t.Fatal on any
// error.
func (e *Env) CloseBuffer(name string) {
	e.T.Helper()
	if err := e.Editor.CloseBuffer(e.Ctx, name); err != nil {
		e.T.Fatal(err)
	}
}

// EditBuffer applies edits to an editor buffer, calling t.Fatal on any error.
func (e *Env) EditBuffer(name string, edits ...fake.Edit) {
	e.T.Helper()
	if err := e.Editor.EditBuffer(e.Ctx, name, edits); err != nil {
		e.T.Fatal(err)
	}
}

// RegexpSearch returns the starting position of the first match for re in the
// buffer specified by name, calling t.Fatal on any error. It first searches
// for the position in open buffers, then in workspace files.
func (e *Env) RegexpSearch(name, re string) fake.Pos {
	e.T.Helper()
	pos, err := e.Editor.RegexpSearch(name, re)
	if err == fake.ErrUnknownBuffer {
		pos, err = e.Sandbox.Workdir.RegexpSearch(name, re)
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
	if err := e.Editor.RegexpReplace(e.Ctx, name, regexpStr, replace); err != nil {
		e.T.Fatalf("RegexpReplace: %v", err)
	}
}

// SaveBuffer saves an editor buffer, calling t.Fatal on any error.
func (e *Env) SaveBuffer(name string) {
	e.T.Helper()
	if err := e.Editor.SaveBuffer(e.Ctx, name); err != nil {
		e.T.Fatal(err)
	}
}

// GoToDefinition goes to definition in the editor, calling t.Fatal on any
// error.
func (e *Env) GoToDefinition(name string, pos fake.Pos) (string, fake.Pos) {
	e.T.Helper()
	n, p, err := e.Editor.GoToDefinition(e.Ctx, name, pos)
	if err != nil {
		e.T.Fatal(err)
	}
	return n, p
}

// Symbol returns symbols matching query
func (e *Env) Symbol(query string) []fake.SymbolInformation {
	e.T.Helper()
	r, err := e.Editor.Symbol(e.Ctx, query)
	if err != nil {
		e.T.Fatal(err)
	}
	return r
}

// FormatBuffer formats the editor buffer, calling t.Fatal on any error.
func (e *Env) FormatBuffer(name string) {
	e.T.Helper()
	if err := e.Editor.FormatBuffer(e.Ctx, name); err != nil {
		e.T.Fatal(err)
	}
}

// OrganizeImports processes the source.organizeImports codeAction, calling
// t.Fatal on any error.
func (e *Env) OrganizeImports(name string) {
	e.T.Helper()
	if err := e.Editor.OrganizeImports(e.Ctx, name); err != nil {
		e.T.Fatal(err)
	}
}

// ApplyQuickFixes processes the quickfix codeAction, calling t.Fatal on any error.
func (e *Env) ApplyQuickFixes(path string, diagnostics []protocol.Diagnostic) {
	e.T.Helper()
	if err := e.Editor.ApplyQuickFixes(e.Ctx, path, nil, diagnostics); err != nil {
		e.T.Fatal(err)
	}
}

// Hover in the editor, calling t.Fatal on any error.
func (e *Env) Hover(name string, pos fake.Pos) (*protocol.MarkupContent, fake.Pos) {
	e.T.Helper()
	c, p, err := e.Editor.Hover(e.Ctx, name, pos)
	if err != nil {
		e.T.Fatal(err)
	}
	return c, p
}

func (e *Env) DocumentLink(name string) []protocol.DocumentLink {
	e.T.Helper()
	links, err := e.Editor.DocumentLink(e.Ctx, name)
	if err != nil {
		e.T.Fatal(err)
	}
	return links
}

func checkIsFatal(t *testing.T, err error) {
	t.Helper()
	if err != nil && !errors.Is(err, io.EOF) && !errors.Is(err, io.ErrClosedPipe) {
		t.Fatal(err)
	}
}

// CloseEditor shuts down the editor, calling t.Fatal on any error.
func (e *Env) CloseEditor() {
	e.T.Helper()
	checkIsFatal(e.T, e.Editor.Close(e.Ctx))
}

// RunGenerate runs go:generate on the given dir, calling t.Fatal on any error.
// It waits for the generate command to complete and checks for file changes
// before returning.
func (e *Env) RunGenerate(dir string) {
	e.T.Helper()
	if err := e.Editor.RunGenerate(e.Ctx, dir); err != nil {
		e.T.Fatal(err)
	}
	e.Await(NoOutstandingWork())
	// Ideally the fake.Workspace would handle all synthetic file watching, but
	// we help it out here as we need to wait for the generate command to
	// complete before checking the filesystem.
	e.CheckForFileChanges()
}

// CheckForFileChanges triggers a manual poll of the workspace for any file
// changes since creation, or since last polling. It is a workaround for the
// lack of true file watching support in the fake workspace.
func (e *Env) CheckForFileChanges() {
	e.T.Helper()
	if err := e.Sandbox.Workdir.CheckForFileChanges(e.Ctx); err != nil {
		e.T.Fatal(err)
	}
}

// CodeLens calls textDocument/codeLens for the given path, calling t.Fatal on
// any error.
func (e *Env) CodeLens(path string) []protocol.CodeLens {
	e.T.Helper()
	lens, err := e.Editor.CodeLens(e.Ctx, path)
	if err != nil {
		e.T.Fatal(err)
	}
	return lens
}

// ReferencesAtRegexp calls textDocument/references for the given path at the
// position of the given regexp.
func (e *Env) References(path string, pos fake.Pos) []protocol.Location {
	e.T.Helper()
	locations, err := e.Editor.References(e.Ctx, path, pos)
	if err != nil {
		e.T.Fatal(err)
	}
	return locations
}

// CodeAction calls testDocument/codeAction for the given path, and calls
// t.Fatal if there are errors.
func (e *Env) CodeAction(path string) []protocol.CodeAction {
	e.T.Helper()
	actions, err := e.Editor.CodeAction(e.Ctx, path, nil)
	if err != nil {
		e.T.Fatal(err)
	}
	return actions
}

// ChangeEnv modifies the editor environment and reconfigures the LSP client.
// TODO: extend this to "ChangeConfiguration", once we refactor the way editor
// configuration is defined.
func (e *Env) ChangeEnv(overlay map[string]string) {
	e.T.Helper()
	// TODO: to be correct, this should probably be synchronized, but right now
	// configuration is only ever modified synchronously in a regtest, so this
	// correctness can wait for the previously mentioned refactoring.
	if e.Editor.Config.Env == nil {
		e.Editor.Config.Env = make(map[string]string)
	}
	for k, v := range overlay {
		e.Editor.Config.Env[k] = v
	}
	var params protocol.DidChangeConfigurationParams
	if err := e.Editor.Server.DidChangeConfiguration(e.Ctx, &params); err != nil {
		e.T.Fatal(err)
	}
}
