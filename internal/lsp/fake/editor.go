// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fake

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"sync"

	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/lsp/protocol"
)

// Editor is a fake editor client.  It keeps track of client state and can be
// used for writing LSP tests.
type Editor struct {
	Config EditorConfig

	// server, client, and sandbox are concurrency safe and written only
	// at construction time, so do not require synchronization.
	server  protocol.Server
	client  *Client
	sandbox *Sandbox

	// Since this editor is intended just for testing, we use very coarse
	// locking.
	mu sync.Mutex
	// Editor state.
	buffers map[string]buffer
	// Capabilities / Options
	serverCapabilities protocol.ServerCapabilities
}

type buffer struct {
	version int
	path    string
	content []string
}

func (b buffer) text() string {
	return strings.Join(b.content, "\n")
}

// EditorConfig configures the editor's LSP session. This is similar to
// source.UserOptions, but we use a separate type here so that we expose only
// that configuration which we support.
//
// The zero value for EditorConfig should correspond to its defaults.
type EditorConfig struct {
	Env []string

	// CodeLens is a map defining whether codelens are enabled, keyed by the
	// codeLens command. CodeLens which are not present in this map are left in
	// their default state.
	CodeLens map[string]bool

	// SymbolMatcher is the config associated with the "symbolMatcher" gopls
	// config option.
	SymbolMatcher *string
}

// NewEditor Creates a new Editor.
func NewEditor(ws *Sandbox, config EditorConfig) *Editor {
	return &Editor{
		buffers: make(map[string]buffer),
		sandbox: ws,
		Config:  config,
	}
}

// Connect configures the editor to communicate with an LSP server on conn. It
// is not concurrency safe, and should be called at most once, before using the
// editor.
//
// It returns the editor, so that it may be called as follows:
//   editor, err := NewEditor(s).Connect(ctx, conn)
func (e *Editor) Connect(ctx context.Context, conn *jsonrpc2.Conn, hooks ClientHooks) (*Editor, error) {
	e.server = protocol.ServerDispatcher(conn)
	e.client = &Client{editor: e, hooks: hooks}
	go conn.Run(ctx,
		protocol.Handlers(
			protocol.ClientHandler(e.client,
				jsonrpc2.MethodNotFound)))
	if err := e.initialize(ctx); err != nil {
		return nil, err
	}
	e.sandbox.Workdir.AddWatcher(e.onFileChanges)
	return e, nil
}

// Shutdown issues the 'shutdown' LSP notification.
func (e *Editor) Shutdown(ctx context.Context) error {
	if e.server != nil {
		if err := e.server.Shutdown(ctx); err != nil {
			return fmt.Errorf("Shutdown: %w", err)
		}
	}
	return nil
}

// Exit issues the 'exit' LSP notification.
func (e *Editor) Exit(ctx context.Context) error {
	if e.server != nil {
		// Not all LSP clients issue the exit RPC, but we do so here to ensure that
		// we gracefully handle it on multi-session servers.
		if err := e.server.Exit(ctx); err != nil {
			return fmt.Errorf("Exit: %w", err)
		}
	}
	return nil
}

// Client returns the LSP client for this editor.
func (e *Editor) Client() *Client {
	return e.client
}

func (e *Editor) configuration() map[string]interface{} {
	config := map[string]interface{}{
		"verboseWorkDoneProgress": true,
	}

	envvars := e.sandbox.GoEnv()
	envvars = append(envvars, e.Config.Env...)
	env := map[string]interface{}{}
	for _, value := range envvars {
		kv := strings.SplitN(value, "=", 2)
		env[kv[0]] = kv[1]
	}
	config["env"] = env

	if e.Config.CodeLens != nil {
		config["codelens"] = e.Config.CodeLens
	}

	if e.Config.SymbolMatcher != nil {
		config["symbolMatcher"] = *e.Config.SymbolMatcher
	}

	return config
}

func (e *Editor) initialize(ctx context.Context) error {
	params := &protocol.ParamInitialize{}
	params.ClientInfo.Name = "fakeclient"
	params.ClientInfo.Version = "v1.0.0"
	params.RootURI = e.sandbox.Workdir.RootURI()
	params.Capabilities.Workspace.Configuration = true
	params.Capabilities.Window.WorkDoneProgress = true
	// TODO: set client capabilities
	params.InitializationOptions = e.configuration()

	params.Trace = "messages"
	// TODO: support workspace folders.
	if e.server != nil {
		resp, err := e.server.Initialize(ctx, params)
		if err != nil {
			return fmt.Errorf("initialize: %w", err)
		}
		e.mu.Lock()
		e.serverCapabilities = resp.Capabilities
		e.mu.Unlock()

		if err := e.server.Initialized(ctx, &protocol.InitializedParams{}); err != nil {
			return fmt.Errorf("initialized: %w", err)
		}
	}
	// TODO: await initial configuration here, or expect gopls to manage that?
	return nil
}

func (e *Editor) onFileChanges(ctx context.Context, evts []FileEvent) {
	if e.server == nil {
		return
	}
	var lspevts []protocol.FileEvent
	for _, evt := range evts {
		lspevts = append(lspevts, evt.ProtocolEvent)
	}
	e.server.DidChangeWatchedFiles(ctx, &protocol.DidChangeWatchedFilesParams{
		Changes: lspevts,
	})
}

// OpenFile creates a buffer for the given workdir-relative file.
func (e *Editor) OpenFile(ctx context.Context, path string) error {
	content, err := e.sandbox.Workdir.ReadFile(path)
	if err != nil {
		return err
	}
	buf := newBuffer(path, content)
	e.mu.Lock()
	e.buffers[path] = buf
	item := textDocumentItem(e.sandbox.Workdir, buf)
	e.mu.Unlock()

	if e.server != nil {
		if err := e.server.DidOpen(ctx, &protocol.DidOpenTextDocumentParams{
			TextDocument: item,
		}); err != nil {
			return fmt.Errorf("DidOpen: %w", err)
		}
	}
	return nil
}

func newBuffer(path, content string) buffer {
	return buffer{
		version: 1,
		path:    path,
		content: strings.Split(content, "\n"),
	}
}

func textDocumentItem(wd *Workdir, buf buffer) protocol.TextDocumentItem {
	uri := wd.URI(buf.path)
	languageID := ""
	if strings.HasSuffix(buf.path, ".go") {
		// TODO: what about go.mod files? What is their language ID?
		languageID = "go"
	}
	return protocol.TextDocumentItem{
		URI:        uri,
		LanguageID: languageID,
		Version:    float64(buf.version),
		Text:       buf.text(),
	}
}

// CreateBuffer creates a new unsaved buffer corresponding to the workdir path,
// containing the given textual content.
func (e *Editor) CreateBuffer(ctx context.Context, path, content string) error {
	buf := newBuffer(path, content)
	e.mu.Lock()
	e.buffers[path] = buf
	item := textDocumentItem(e.sandbox.Workdir, buf)
	e.mu.Unlock()

	if e.server != nil {
		if err := e.server.DidOpen(ctx, &protocol.DidOpenTextDocumentParams{
			TextDocument: item,
		}); err != nil {
			return fmt.Errorf("DidOpen: %w", err)
		}
	}
	return nil
}

// CloseBuffer removes the current buffer (regardless of whether it is saved).
func (e *Editor) CloseBuffer(ctx context.Context, path string) error {
	e.mu.Lock()
	_, ok := e.buffers[path]
	if !ok {
		e.mu.Unlock()
		return ErrUnknownBuffer
	}
	delete(e.buffers, path)
	e.mu.Unlock()

	if e.server != nil {
		if err := e.server.DidClose(ctx, &protocol.DidCloseTextDocumentParams{
			TextDocument: e.textDocumentIdentifier(path),
		}); err != nil {
			return fmt.Errorf("DidClose: %w", err)
		}
	}
	return nil
}

func (e *Editor) textDocumentIdentifier(path string) protocol.TextDocumentIdentifier {
	return protocol.TextDocumentIdentifier{
		URI: e.sandbox.Workdir.URI(path),
	}
}

// SaveBuffer writes the content of the buffer specified by the given path to
// the filesystem.
func (e *Editor) SaveBuffer(ctx context.Context, path string) error {
	if err := e.OrganizeImports(ctx, path); err != nil {
		return fmt.Errorf("organizing imports before save: %w", err)
	}
	if err := e.FormatBuffer(ctx, path); err != nil {
		return fmt.Errorf("formatting before save: %w", err)
	}

	e.mu.Lock()
	buf, ok := e.buffers[path]
	if !ok {
		e.mu.Unlock()
		return fmt.Errorf(fmt.Sprintf("unknown buffer: %q", path))
	}
	content := buf.text()
	includeText := false
	syncOptions, ok := e.serverCapabilities.TextDocumentSync.(protocol.TextDocumentSyncOptions)
	if ok {
		includeText = syncOptions.Save.IncludeText
	}
	e.mu.Unlock()

	docID := e.textDocumentIdentifier(buf.path)
	if e.server != nil {
		if err := e.server.WillSave(ctx, &protocol.WillSaveTextDocumentParams{
			TextDocument: docID,
			Reason:       protocol.Manual,
		}); err != nil {
			return fmt.Errorf("WillSave: %w", err)
		}
	}
	if err := e.sandbox.Workdir.WriteFile(ctx, path, content); err != nil {
		return fmt.Errorf("writing %q: %w", path, err)
	}
	if e.server != nil {
		params := &protocol.DidSaveTextDocumentParams{
			TextDocument: protocol.VersionedTextDocumentIdentifier{
				Version:                float64(buf.version),
				TextDocumentIdentifier: docID,
			},
		}
		if includeText {
			params.Text = &content
		}
		if err := e.server.DidSave(ctx, params); err != nil {
			return fmt.Errorf("DidSave: %w", err)
		}
	}
	return nil
}

// contentPosition returns the (Line, Column) position corresponding to offset
// in the buffer referenced by path.
func contentPosition(content string, offset int) (Pos, error) {
	scanner := bufio.NewScanner(strings.NewReader(content))
	start := 0
	line := 0
	for scanner.Scan() {
		end := start + len([]rune(scanner.Text())) + 1
		if offset < end {
			return Pos{Line: line, Column: offset - start}, nil
		}
		start = end
		line++
	}
	if err := scanner.Err(); err != nil {
		return Pos{}, fmt.Errorf("scanning content: %w", err)
	}
	// Scan() will drop the last line if it is empty. Correct for this.
	if strings.HasSuffix(content, "\n") && offset == start {
		return Pos{Line: line, Column: 0}, nil
	}
	return Pos{}, fmt.Errorf("position %d out of bounds in %q (line = %d, start = %d)", offset, content, line, start)
}

// ErrNoMatch is returned if a regexp search fails.
var (
	ErrNoMatch       = errors.New("no match")
	ErrUnknownBuffer = errors.New("unknown buffer")
)

// regexpRange returns the start and end of the first occurrence of either re
// or its singular subgroup. It returns ErrNoMatch if the regexp doesn't match.
func regexpRange(content, re string) (Pos, Pos, error) {
	var start, end int
	rec, err := regexp.Compile(re)
	if err != nil {
		return Pos{}, Pos{}, err
	}
	indexes := rec.FindStringSubmatchIndex(content)
	if indexes == nil {
		return Pos{}, Pos{}, ErrNoMatch
	}
	switch len(indexes) {
	case 2:
		// no subgroups: return the range of the regexp expression
		start, end = indexes[0], indexes[1]
	case 4:
		// one subgroup: return its range
		start, end = indexes[2], indexes[3]
	default:
		return Pos{}, Pos{}, fmt.Errorf("invalid search regexp %q: expect either 0 or 1 subgroups, got %d", re, len(indexes)/2-1)
	}
	startPos, err := contentPosition(content, start)
	if err != nil {
		return Pos{}, Pos{}, err
	}
	endPos, err := contentPosition(content, end)
	if err != nil {
		return Pos{}, Pos{}, err
	}
	return startPos, endPos, nil
}

// RegexpSearch returns the position of the first match for re in the buffer
// bufName. For convenience, RegexpSearch supports the following two modes:
//  1. If re has no subgroups, return the position of the match for re itself.
//  2. If re has one subgroup, return the position of the first subgroup.
// It returns an error re is invalid, has more than one subgroup, or doesn't
// match the buffer.
func (e *Editor) RegexpSearch(bufName, re string) (Pos, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	buf, ok := e.buffers[bufName]
	if !ok {
		return Pos{}, ErrUnknownBuffer
	}
	start, _, err := regexpRange(buf.text(), re)
	return start, err
}

// RegexpReplace edits the buffer corresponding to path by replacing the first
// instance of re, or its first subgroup, with the replace text. See
// RegexpSearch for more explanation of these two modes.
// It returns an error if re is invalid, has more than one subgroup, or doesn't
// match the buffer.
func (e *Editor) RegexpReplace(ctx context.Context, path, re, replace string) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	buf, ok := e.buffers[path]
	if !ok {
		return ErrUnknownBuffer
	}
	content := buf.text()
	start, end, err := regexpRange(content, re)
	if err != nil {
		return err
	}
	return e.editBufferLocked(ctx, path, []Edit{{
		Start: start,
		End:   end,
		Text:  replace,
	}})
}

// EditBuffer applies the given test edits to the buffer identified by path.
func (e *Editor) EditBuffer(ctx context.Context, path string, edits []Edit) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.editBufferLocked(ctx, path, edits)
}

// BufferText returns the content of the buffer with the given name.
func (e *Editor) BufferText(name string) string {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.buffers[name].text()
}

// BufferVersion returns the current version of the buffer corresponding to
// name (or 0 if it is not being edited).
func (e *Editor) BufferVersion(name string) int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.buffers[name].version
}

func (e *Editor) editBufferLocked(ctx context.Context, path string, edits []Edit) error {
	buf, ok := e.buffers[path]
	if !ok {
		return fmt.Errorf("unknown buffer %q", path)
	}
	var (
		content = make([]string, len(buf.content))
		err     error
		evts    []protocol.TextDocumentContentChangeEvent
	)
	copy(content, buf.content)
	content, err = editContent(content, edits)
	if err != nil {
		return err
	}

	buf.content = content
	buf.version++
	e.buffers[path] = buf
	// A simple heuristic: if there is only one edit, send it incrementally.
	// Otherwise, send the entire content.
	if len(edits) == 1 {
		evts = append(evts, edits[0].toProtocolChangeEvent())
	} else {
		evts = append(evts, protocol.TextDocumentContentChangeEvent{
			Text: buf.text(),
		})
	}
	params := &protocol.DidChangeTextDocumentParams{
		TextDocument: protocol.VersionedTextDocumentIdentifier{
			Version:                float64(buf.version),
			TextDocumentIdentifier: e.textDocumentIdentifier(buf.path),
		},
		ContentChanges: evts,
	}
	if e.server != nil {
		if err := e.server.DidChange(ctx, params); err != nil {
			return fmt.Errorf("DidChange: %w", err)
		}
	}
	return nil
}

// GoToDefinition jumps to the definition of the symbol at the given position
// in an open buffer.
func (e *Editor) GoToDefinition(ctx context.Context, path string, pos Pos) (string, Pos, error) {
	if err := e.checkBufferPosition(path, pos); err != nil {
		return "", Pos{}, err
	}
	params := &protocol.DefinitionParams{}
	params.TextDocument.URI = e.sandbox.Workdir.URI(path)
	params.Position = pos.toProtocolPosition()

	resp, err := e.server.Definition(ctx, params)
	if err != nil {
		return "", Pos{}, fmt.Errorf("definition: %w", err)
	}
	if len(resp) == 0 {
		return "", Pos{}, nil
	}
	newPath := e.sandbox.Workdir.URIToPath(resp[0].URI)
	newPos := fromProtocolPosition(resp[0].Range.Start)
	if err := e.OpenFile(ctx, newPath); err != nil {
		return "", Pos{}, fmt.Errorf("OpenFile: %w", err)
	}
	return newPath, newPos, nil
}

// OrganizeImports requests and performs the source.organizeImports codeAction.
func (e *Editor) OrganizeImports(ctx context.Context, path string) error {
	return e.codeAction(ctx, path, nil, protocol.SourceOrganizeImports)
}

// ApplyQuickFixes requests and performs the quickfix codeAction.
func (e *Editor) ApplyQuickFixes(ctx context.Context, path string, diagnostics []protocol.Diagnostic) error {
	return e.codeAction(ctx, path, diagnostics, protocol.QuickFix, protocol.SourceFixAll)
}

func (e *Editor) codeAction(ctx context.Context, path string, diagnostics []protocol.Diagnostic, only ...protocol.CodeActionKind) error {
	if e.server == nil {
		return nil
	}
	params := &protocol.CodeActionParams{}
	params.TextDocument.URI = e.sandbox.Workdir.URI(path)
	params.Context.Only = only
	if diagnostics != nil {
		params.Context.Diagnostics = diagnostics
	}
	actions, err := e.server.CodeAction(ctx, params)
	if err != nil {
		return fmt.Errorf("textDocument/codeAction: %w", err)
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	for _, action := range actions {
		var match bool
		for _, o := range only {
			if action.Kind == o {
				match = true
				break
			}
		}
		if !match {
			continue
		}
		for _, change := range action.Edit.DocumentChanges {
			path := e.sandbox.Workdir.URIToPath(change.TextDocument.URI)
			if float64(e.buffers[path].version) != change.TextDocument.Version {
				// Skip edits for old versions.
				continue
			}
			edits := convertEdits(change.Edits)
			if err := e.editBufferLocked(ctx, path, edits); err != nil {
				return fmt.Errorf("editing buffer %q: %w", path, err)
			}
		}
	}
	return nil
}

func convertEdits(protocolEdits []protocol.TextEdit) []Edit {
	var edits []Edit
	for _, lspEdit := range protocolEdits {
		edits = append(edits, fromProtocolTextEdit(lspEdit))
	}
	return edits
}

// FormatBuffer gofmts a Go file.
func (e *Editor) FormatBuffer(ctx context.Context, path string) error {
	if e.server == nil {
		return nil
	}
	e.mu.Lock()
	version := e.buffers[path].version
	e.mu.Unlock()
	params := &protocol.DocumentFormattingParams{}
	params.TextDocument.URI = e.sandbox.Workdir.URI(path)
	resp, err := e.server.Formatting(ctx, params)
	if err != nil {
		return fmt.Errorf("textDocument/formatting: %w", err)
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if versionAfter := e.buffers[path].version; versionAfter != version {
		return fmt.Errorf("before receipt of formatting edits, buffer version changed from %d to %d", version, versionAfter)
	}
	edits := convertEdits(resp)
	return e.editBufferLocked(ctx, path, edits)
}

func (e *Editor) checkBufferPosition(path string, pos Pos) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	buf, ok := e.buffers[path]
	if !ok {
		return fmt.Errorf("buffer %q is not open", path)
	}
	if !inText(pos, buf.content) {
		return fmt.Errorf("position %v is invalid in buffer %q", pos, path)
	}
	return nil
}

// RunGenerate runs `go generate` non-recursively in the workdir-relative dir
// path. It does not report any resulting file changes as a watched file
// change, so must be followed by a call to Workdir.CheckForFileChanges once
// the generate command has completed.
func (e *Editor) RunGenerate(ctx context.Context, dir string) error {
	if e.server == nil {
		return nil
	}
	absDir := e.sandbox.Workdir.filePath(dir)
	params := &protocol.ExecuteCommandParams{
		Command:   "generate",
		Arguments: []interface{}{absDir, false},
	}
	if _, err := e.server.ExecuteCommand(ctx, params); err != nil {
		return fmt.Errorf("running generate: %v", err)
	}
	// Unfortunately we can't simply poll the workdir for file changes here,
	// because server-side command may not have completed. In regtests, we can
	// Await this state change, but here we must delegate that responsibility to
	// the caller.
	return nil
}

// CodeLens execute a codelens request on the server.
func (e *Editor) CodeLens(ctx context.Context, path string) ([]protocol.CodeLens, error) {
	if e.server == nil {
		return nil, nil
	}
	e.mu.Lock()
	_, ok := e.buffers[path]
	e.mu.Unlock()
	if !ok {
		return nil, fmt.Errorf("buffer %q is not open", path)
	}
	params := &protocol.CodeLensParams{
		TextDocument: e.textDocumentIdentifier(path),
	}
	lens, err := e.server.CodeLens(ctx, params)
	if err != nil {
		return nil, err
	}
	return lens, nil
}
