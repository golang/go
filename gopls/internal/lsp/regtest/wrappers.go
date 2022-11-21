// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regtest

import (
	"encoding/json"
	"path"

	"golang.org/x/tools/gopls/internal/lsp/command"
	"golang.org/x/tools/gopls/internal/lsp/fake"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
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

// ListFiles lists relative paths to files in the given directory.
// It calls t.Fatal on any error.
func (e *Env) ListFiles(dir string) []string {
	e.T.Helper()
	paths, err := e.Sandbox.Workdir.ListFiles(dir)
	if err != nil {
		e.T.Fatal(err)
	}
	return paths
}

// OpenFile opens a file in the editor, calling t.Fatal on any error.
func (e *Env) OpenFile(name string) {
	e.T.Helper()
	if err := e.Editor.OpenFile(e.Ctx, name); err != nil {
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

func (e *Env) SetBufferContent(name string, content string) {
	e.T.Helper()
	if err := e.Editor.SetBufferContent(e.Ctx, name, content); err != nil {
		e.T.Fatal(err)
	}
}

// RegexpRange returns the range of the first match for re in the buffer
// specified by name, calling t.Fatal on any error. It first searches for the
// position in open buffers, then in workspace files.
func (e *Env) RegexpRange(name, re string) (fake.Pos, fake.Pos) {
	e.T.Helper()
	start, end, err := e.Editor.RegexpRange(name, re)
	if err == fake.ErrUnknownBuffer {
		start, end, err = e.Sandbox.Workdir.RegexpRange(name, re)
	}
	if err != nil {
		e.T.Fatalf("RegexpRange: %v, %v", name, err)
	}
	return start, end
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

func (e *Env) SaveBufferWithoutActions(name string) {
	e.T.Helper()
	if err := e.Editor.SaveBufferWithoutActions(e.Ctx, name); err != nil {
		e.T.Fatal(err)
	}
}

// GoToDefinition goes to definition in the editor, calling t.Fatal on any
// error. It returns the path and position of the resulting jump.
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

// ApplyCodeAction applies the given code action.
func (e *Env) ApplyCodeAction(action protocol.CodeAction) {
	e.T.Helper()
	if err := e.Editor.ApplyCodeAction(e.Ctx, action); err != nil {
		e.T.Fatal(err)
	}
}

// GetQuickFixes returns the available quick fix code actions.
func (e *Env) GetQuickFixes(path string, diagnostics []protocol.Diagnostic) []protocol.CodeAction {
	e.T.Helper()
	actions, err := e.Editor.GetQuickFixes(e.Ctx, path, nil, diagnostics)
	if err != nil {
		e.T.Fatal(err)
	}
	return actions
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

func (e *Env) DocumentHighlight(name string, pos fake.Pos) []protocol.DocumentHighlight {
	e.T.Helper()
	highlights, err := e.Editor.DocumentHighlight(e.Ctx, name, pos)
	if err != nil {
		e.T.Fatal(err)
	}
	return highlights
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

// RunGoCommand runs the given command in the sandbox's default working
// directory.
func (e *Env) RunGoCommand(verb string, args ...string) {
	e.T.Helper()
	if err := e.Sandbox.RunGoCommand(e.Ctx, "", verb, args, true); err != nil {
		e.T.Fatal(err)
	}
}

// RunGoCommandInDir is like RunGoCommand, but executes in the given
// relative directory of the sandbox.
func (e *Env) RunGoCommandInDir(dir, verb string, args ...string) {
	e.T.Helper()
	if err := e.Sandbox.RunGoCommand(e.Ctx, dir, verb, args, true); err != nil {
		e.T.Fatal(err)
	}
}

// GoVersion checks the version of the go command.
// It returns the X in Go 1.X.
func (e *Env) GoVersion() int {
	e.T.Helper()
	v, err := e.Sandbox.GoVersion(e.Ctx)
	if err != nil {
		e.T.Fatal(err)
	}
	return v
}

// DumpGoSum prints the correct go.sum contents for dir in txtar format,
// for use in creating regtests.
func (e *Env) DumpGoSum(dir string) {
	e.T.Helper()

	if err := e.Sandbox.RunGoCommand(e.Ctx, dir, "list", []string{"-mod=mod", "..."}, true); err != nil {
		e.T.Fatal(err)
	}
	sumFile := path.Join(dir, "/go.sum")
	e.T.Log("\n\n-- " + sumFile + " --\n" + e.ReadWorkspaceFile(sumFile))
	e.T.Fatal("see contents above")
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

// ExecuteCodeLensCommand executes the command for the code lens matching the
// given command name.
func (e *Env) ExecuteCodeLensCommand(path string, cmd command.Command, result interface{}) {
	e.T.Helper()
	lenses := e.CodeLens(path)
	var lens protocol.CodeLens
	var found bool
	for _, l := range lenses {
		if l.Command.Command == cmd.ID() {
			lens = l
			found = true
		}
	}
	if !found {
		e.T.Fatalf("found no command with the ID %s", cmd.ID())
	}
	e.ExecuteCommand(&protocol.ExecuteCommandParams{
		Command:   lens.Command.Command,
		Arguments: lens.Command.Arguments,
	}, result)
}

func (e *Env) ExecuteCommand(params *protocol.ExecuteCommandParams, result interface{}) {
	e.T.Helper()
	response, err := e.Editor.ExecuteCommand(e.Ctx, params)
	if err != nil {
		e.T.Fatal(err)
	}
	if result == nil {
		return
	}
	// Hack: The result of an executeCommand request will be unmarshaled into
	// maps. Re-marshal and unmarshal into the type we expect.
	//
	// This could be improved by generating a jsonrpc2 command client from the
	// command.Interface, but that should only be done if we're consolidating
	// this part of the tsprotocol generation.
	data, err := json.Marshal(response)
	if err != nil {
		e.T.Fatal(err)
	}
	if err := json.Unmarshal(data, result); err != nil {
		e.T.Fatal(err)
	}
}

// InlayHints calls textDocument/inlayHints for the given path, calling t.Fatal on
// any error.
func (e *Env) InlayHints(path string) []protocol.InlayHint {
	e.T.Helper()
	hints, err := e.Editor.InlayHint(e.Ctx, path)
	if err != nil {
		e.T.Fatal(err)
	}
	return hints
}

// WorkspaceSymbol calls workspace/symbol
func (e *Env) WorkspaceSymbol(sym string) []protocol.SymbolInformation {
	e.T.Helper()
	ans, err := e.Editor.Symbols(e.Ctx, sym)
	if err != nil {
		e.T.Fatal(err)
	}
	return ans
}

// References wraps Editor.References, calling t.Fatal on any error.
func (e *Env) References(path string, pos fake.Pos) []protocol.Location {
	e.T.Helper()
	locations, err := e.Editor.References(e.Ctx, path, pos)
	if err != nil {
		e.T.Fatal(err)
	}
	return locations
}

// Rename wraps Editor.Rename, calling t.Fatal on any error.
func (e *Env) Rename(path string, pos fake.Pos, newName string) {
	e.T.Helper()
	if err := e.Editor.Rename(e.Ctx, path, pos, newName); err != nil {
		e.T.Fatal(err)
	}
}

// Implementations wraps Editor.Implementations, calling t.Fatal on any error.
func (e *Env) Implementations(path string, pos fake.Pos) []protocol.Location {
	e.T.Helper()
	locations, err := e.Editor.Implementations(e.Ctx, path, pos)
	if err != nil {
		e.T.Fatal(err)
	}
	return locations
}

// RenameFile wraps Editor.RenameFile, calling t.Fatal on any error.
func (e *Env) RenameFile(oldPath, newPath string) {
	e.T.Helper()
	if err := e.Editor.RenameFile(e.Ctx, oldPath, newPath); err != nil {
		e.T.Fatal(err)
	}
}

// Completion executes a completion request on the server.
func (e *Env) Completion(path string, pos fake.Pos) *protocol.CompletionList {
	e.T.Helper()
	completions, err := e.Editor.Completion(e.Ctx, path, pos)
	if err != nil {
		e.T.Fatal(err)
	}
	return completions
}

// AcceptCompletion accepts a completion for the given item at the given
// position.
func (e *Env) AcceptCompletion(path string, pos fake.Pos, item protocol.CompletionItem) {
	e.T.Helper()
	if err := e.Editor.AcceptCompletion(e.Ctx, path, pos, item); err != nil {
		e.T.Fatal(err)
	}
}

// CodeAction calls testDocument/codeAction for the given path, and calls
// t.Fatal if there are errors.
func (e *Env) CodeAction(path string, diagnostics []protocol.Diagnostic) []protocol.CodeAction {
	e.T.Helper()
	actions, err := e.Editor.CodeAction(e.Ctx, path, nil, diagnostics)
	if err != nil {
		e.T.Fatal(err)
	}
	return actions
}

// ChangeConfiguration updates the editor config, calling t.Fatal on any error.
func (e *Env) ChangeConfiguration(newConfig fake.EditorConfig) {
	e.T.Helper()
	if err := e.Editor.ChangeConfiguration(e.Ctx, newConfig); err != nil {
		e.T.Fatal(err)
	}
}

// ChangeWorkspaceFolders updates the editor workspace folders, calling t.Fatal
// on any error.
func (e *Env) ChangeWorkspaceFolders(newFolders ...string) {
	e.T.Helper()
	if err := e.Editor.ChangeWorkspaceFolders(e.Ctx, newFolders); err != nil {
		e.T.Fatal(err)
	}
}
