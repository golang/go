// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fake

import (
	"context"
	"fmt"
	"strings"
	"sync"

	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/lsp/protocol"
)

// Editor is a fake editor client.  It keeps track of client state and can be
// used for writing LSP tests.
type Editor struct {
	// server, client, and workspace are concurrency safe and written only at
	// construction, so do not require synchronization.
	server protocol.Server
	client *Client
	ws     *Workspace

	// Since this editor is intended just for testing, we use very coarse
	// locking.
	mu sync.Mutex
	// Editor state.
	buffers     map[string]buffer
	lastMessage *protocol.ShowMessageParams
	logs        []*protocol.LogMessageParams
	diagnostics *protocol.PublishDiagnosticsParams
	events      []interface{}
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

// NewConnectedEditor creates a new editor that dispatches the LSP across the
// provided jsonrpc2 connection.
//
// The returned editor is initialized and ready to use.
func NewConnectedEditor(ctx context.Context, ws *Workspace, conn *jsonrpc2.Conn) (*Editor, error) {
	e := NewEditor(ws)
	e.server = protocol.ServerDispatcher(conn)
	e.client = &Client{Editor: e}
	conn.AddHandler(protocol.ClientHandler(e.client))
	if err := e.initialize(ctx); err != nil {
		return nil, err
	}
	e.ws.AddWatcher(e.onFileChanges)
	return e, nil
}

// NewEditor Creates a new Editor.
func NewEditor(ws *Workspace) *Editor {
	return &Editor{
		buffers: make(map[string]buffer),
		ws:      ws,
	}
}

// Shutdown issues the 'shutdown' LSP notification.
func (e *Editor) Shutdown(ctx context.Context) error {
	if e.server != nil {
		if err := e.server.Shutdown(ctx); err != nil {
			return fmt.Errorf("Shutdown: %v", err)
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
			return fmt.Errorf("Exit: %v", err)
		}
	}
	return nil
}

// Client returns the LSP client for this editor.
func (e *Editor) Client() *Client {
	return e.client
}

func (e *Editor) configuration() map[string]interface{} {
	return map[string]interface{}{
		"env": map[string]interface{}{
			"GOPATH":      e.ws.GOPATH(),
			"GO111MODULE": "on",
		},
	}
}

func (e *Editor) initialize(ctx context.Context) error {
	params := &protocol.ParamInitialize{}
	params.ClientInfo.Name = "fakeclient"
	params.ClientInfo.Version = "v1.0.0"
	params.RootURI = e.ws.RootURI()

	// TODO: set client capabilities.
	params.Trace = "messages"
	// TODO: support workspace folders.

	if e.server != nil {
		resp, err := e.server.Initialize(ctx, params)
		if err != nil {
			return fmt.Errorf("initialize: %v", err)
		}
		e.mu.Lock()
		e.serverCapabilities = resp.Capabilities
		e.mu.Unlock()

		if err := e.server.Initialized(ctx, &protocol.InitializedParams{}); err != nil {
			return fmt.Errorf("initialized: %v", err)
		}
	}
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

// OpenFile creates a buffer for the given workspace-relative file.
func (e *Editor) OpenFile(ctx context.Context, path string) error {
	content, err := e.ws.ReadFile(path)
	if err != nil {
		return err
	}
	buf := newBuffer(path, content)
	e.mu.Lock()
	e.buffers[path] = buf
	item := textDocumentItem(e.ws, buf)
	e.mu.Unlock()

	if e.server != nil {
		if err := e.server.DidOpen(ctx, &protocol.DidOpenTextDocumentParams{
			TextDocument: item,
		}); err != nil {
			return fmt.Errorf("DidOpen: %v", err)
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

func textDocumentItem(ws *Workspace, buf buffer) protocol.TextDocumentItem {
	uri := ws.URI(buf.path)
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

// CreateBuffer creates a new unsaved buffer corresponding to the workspace
// path, containing the given textual content.
func (e *Editor) CreateBuffer(ctx context.Context, path, content string) error {
	buf := newBuffer(path, content)
	e.mu.Lock()
	e.buffers[path] = buf
	item := textDocumentItem(e.ws, buf)
	e.mu.Unlock()

	if e.server != nil {
		if err := e.server.DidOpen(ctx, &protocol.DidOpenTextDocumentParams{
			TextDocument: item,
		}); err != nil {
			return fmt.Errorf("DidOpen: %v", err)
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
		return fmt.Errorf("unknown path %q", path)
	}
	delete(e.buffers, path)
	e.mu.Unlock()

	if e.server != nil {
		if err := e.server.DidClose(ctx, &protocol.DidCloseTextDocumentParams{
			TextDocument: protocol.TextDocumentIdentifier{
				URI: e.ws.URI(path),
			},
		}); err != nil {
			return fmt.Errorf("DidClose: %v", err)
		}
	}
	return nil
}

// WriteBuffer writes the content of the buffer specified by the given path to
// the filesystem.
func (e *Editor) WriteBuffer(ctx context.Context, path string) error {
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

	docID := protocol.TextDocumentIdentifier{
		URI: e.ws.URI(buf.path),
	}
	if e.server != nil {
		if err := e.server.WillSave(ctx, &protocol.WillSaveTextDocumentParams{
			TextDocument: docID,
			Reason:       protocol.Manual,
		}); err != nil {
			return fmt.Errorf("WillSave: %v", err)
		}
	}
	if err := e.ws.WriteFile(ctx, path, content); err != nil {
		return fmt.Errorf("writing %q: %v", path, err)
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
			return fmt.Errorf("DidSave: %v", err)
		}
	}
	return nil
}

// EditBuffer applies the given test edits to the buffer identified by path.
func (e *Editor) EditBuffer(ctx context.Context, path string, edits []Edit) error {
	params, err := e.doEdits(ctx, path, edits)
	if err != nil {
		return err
	}
	if e.server != nil {
		if err := e.server.DidChange(ctx, params); err != nil {
			return fmt.Errorf("DidChange: %v", err)
		}
	}
	return nil
}

func (e *Editor) doEdits(ctx context.Context, path string, edits []Edit) (*protocol.DidChangeTextDocumentParams, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	buf, ok := e.buffers[path]
	if !ok {
		return nil, fmt.Errorf("unknown buffer %q", path)
	}
	var (
		content = make([]string, len(buf.content))
		err     error
		evts    []protocol.TextDocumentContentChangeEvent
	)
	copy(content, buf.content)
	for _, edit := range edits {
		content, err = editContent(content, edit)
		if err != nil {
			return nil, err
		}
		evts = append(evts, edit.toProtocolChangeEvent())
	}
	buf.content = content
	buf.version++
	e.buffers[path] = buf
	params := &protocol.DidChangeTextDocumentParams{
		TextDocument: protocol.VersionedTextDocumentIdentifier{
			Version: float64(buf.version),
			TextDocumentIdentifier: protocol.TextDocumentIdentifier{
				URI: e.ws.URI(buf.path),
			},
		},
		ContentChanges: evts,
	}
	return params, nil
}

// GoToDefinition jumps to the definition of the symbol at the given position
// in an open buffer.
func (e *Editor) GoToDefinition(ctx context.Context, path string, pos Pos) (string, Pos, error) {
	if err := e.checkBufferPosition(path, pos); err != nil {
		return "", Pos{}, err
	}
	params := &protocol.DefinitionParams{}
	params.TextDocument.URI = e.ws.URI(path)
	params.Position = pos.toProtocolPosition()

	resp, err := e.server.Definition(ctx, params)
	if err != nil {
		return "", Pos{}, fmt.Errorf("definition: %v", err)
	}
	if len(resp) == 0 {
		return "", Pos{}, nil
	}
	newPath := e.ws.URIToPath(resp[0].URI)
	newPos := fromProtocolPosition(resp[0].Range.Start)
	if err := e.OpenFile(ctx, newPath); err != nil {
		return "", Pos{}, fmt.Errorf("OpenFile: %v", err)
	}
	return newPath, newPos, nil
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

// TODO: expose more client functionality, for example Hover, CodeAction,
// Rename, Completion, etc.  setting the content of an entire buffer, etc.
