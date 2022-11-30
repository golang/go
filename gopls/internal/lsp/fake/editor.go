// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fake

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strings"
	"sync"

	"golang.org/x/tools/gopls/internal/lsp/command"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/jsonrpc2/servertest"
	"golang.org/x/tools/internal/xcontext"
)

// Editor is a fake editor client.  It keeps track of client state and can be
// used for writing LSP tests.
type Editor struct {

	// Server, client, and sandbox are concurrency safe and written only
	// at construction time, so do not require synchronization.
	Server     protocol.Server
	cancelConn func()
	serverConn jsonrpc2.Conn
	client     *Client
	sandbox    *Sandbox
	defaultEnv map[string]string

	mu                 sync.Mutex                  // guards config, buffers, serverCapabilities
	config             EditorConfig                // editor configuration
	buffers            map[string]buffer           // open buffers (relative path -> buffer content)
	serverCapabilities protocol.ServerCapabilities // capabilities / options

	// Call metrics for the purpose of expectations. This is done in an ad-hoc
	// manner for now. Perhaps in the future we should do something more
	// systematic. Guarded with a separate mutex as calls may need to be accessed
	// asynchronously via callbacks into the Editor.
	callsMu sync.Mutex
	calls   CallCounts
}

// CallCounts tracks the number of protocol notifications of different types.
type CallCounts struct {
	DidOpen, DidChange, DidSave, DidChangeWatchedFiles, DidClose uint64
}

// buffer holds information about an open buffer in the editor.
type buffer struct {
	windowsLineEndings bool     // use windows line endings when merging lines
	version            int      // monotonic version; incremented on edits
	path               string   // relative path in the workspace
	lines              []string // line content
	dirty              bool     // if true, content is unsaved (TODO(rfindley): rename this field)
}

func (b buffer) text() string {
	eol := "\n"
	if b.windowsLineEndings {
		eol = "\r\n"
	}
	return strings.Join(b.lines, eol)
}

// EditorConfig configures the editor's LSP session. This is similar to
// source.UserOptions, but we use a separate type here so that we expose only
// that configuration which we support.
//
// The zero value for EditorConfig should correspond to its defaults.
type EditorConfig struct {
	// Env holds environment variables to apply on top of the default editor
	// environment. When applying these variables, the special string
	// $SANDBOX_WORKDIR is replaced by the absolute path to the sandbox working
	// directory.
	Env map[string]string

	// WorkspaceFolders is the workspace folders to configure on the LSP server,
	// relative to the sandbox workdir.
	//
	// As a special case, if WorkspaceFolders is nil the editor defaults to
	// configuring a single workspace folder corresponding to the workdir root.
	// To explicitly send no workspace folders, use an empty (non-nil) slice.
	WorkspaceFolders []string

	// Whether to edit files with windows line endings.
	WindowsLineEndings bool

	// Map of language ID -> regexp to match, used to set the file type of new
	// buffers. Applied as an overlay on top of the following defaults:
	//  "go" -> ".*\.go"
	//  "go.mod" -> "go\.mod"
	//  "go.sum" -> "go\.sum"
	//  "gotmpl" -> ".*tmpl"
	FileAssociations map[string]string

	// Settings holds user-provided configuration for the LSP server.
	Settings map[string]interface{}
}

// NewEditor Creates a new Editor.
func NewEditor(sandbox *Sandbox, config EditorConfig) *Editor {
	return &Editor{
		buffers:    make(map[string]buffer),
		sandbox:    sandbox,
		defaultEnv: sandbox.GoEnv(),
		config:     config,
	}
}

// Connect configures the editor to communicate with an LSP server on conn. It
// is not concurrency safe, and should be called at most once, before using the
// editor.
//
// It returns the editor, so that it may be called as follows:
//
//	editor, err := NewEditor(s).Connect(ctx, conn, hooks)
func (e *Editor) Connect(ctx context.Context, connector servertest.Connector, hooks ClientHooks) (*Editor, error) {
	bgCtx, cancelConn := context.WithCancel(xcontext.Detach(ctx))
	conn := connector.Connect(bgCtx)
	e.cancelConn = cancelConn

	e.serverConn = conn
	e.Server = protocol.ServerDispatcher(conn)
	e.client = &Client{editor: e, hooks: hooks}
	conn.Go(bgCtx,
		protocol.Handlers(
			protocol.ClientHandler(e.client,
				jsonrpc2.MethodNotFound)))

	if err := e.initialize(ctx); err != nil {
		return nil, err
	}
	e.sandbox.Workdir.AddWatcher(e.onFileChanges)
	return e, nil
}

func (e *Editor) Stats() CallCounts {
	e.callsMu.Lock()
	defer e.callsMu.Unlock()
	return e.calls
}

// Shutdown issues the 'shutdown' LSP notification.
func (e *Editor) Shutdown(ctx context.Context) error {
	if e.Server != nil {
		if err := e.Server.Shutdown(ctx); err != nil {
			return fmt.Errorf("Shutdown: %w", err)
		}
	}
	return nil
}

// Exit issues the 'exit' LSP notification.
func (e *Editor) Exit(ctx context.Context) error {
	if e.Server != nil {
		// Not all LSP clients issue the exit RPC, but we do so here to ensure that
		// we gracefully handle it on multi-session servers.
		if err := e.Server.Exit(ctx); err != nil {
			return fmt.Errorf("Exit: %w", err)
		}
	}
	return nil
}

// Close issues the shutdown and exit sequence an editor should.
func (e *Editor) Close(ctx context.Context) error {
	if err := e.Shutdown(ctx); err != nil {
		return err
	}
	if err := e.Exit(ctx); err != nil {
		return err
	}
	defer func() {
		e.cancelConn()
	}()

	// called close on the editor should result in the connection closing
	select {
	case <-e.serverConn.Done():
		// connection closed itself
		return nil
	case <-ctx.Done():
		return fmt.Errorf("connection not closed: %w", ctx.Err())
	}
}

// Client returns the LSP client for this editor.
func (e *Editor) Client() *Client {
	return e.client
}

// settingsLocked builds the settings map for use in LSP settings RPCs.
//
// e.mu must be held while calling this function.
func (e *Editor) settingsLocked() map[string]interface{} {
	env := make(map[string]string)
	for k, v := range e.defaultEnv {
		env[k] = v
	}
	for k, v := range e.config.Env {
		env[k] = v
	}
	for k, v := range env {
		v = strings.ReplaceAll(v, "$SANDBOX_WORKDIR", e.sandbox.Workdir.RootURI().SpanURI().Filename())
		env[k] = v
	}

	settings := map[string]interface{}{
		"env": env,

		// Use verbose progress reporting so that regtests can assert on
		// asynchronous operations being completed (such as diagnosing a snapshot).
		"verboseWorkDoneProgress": true,

		// Set a generous completion budget, so that tests don't flake because
		// completions are too slow.
		"completionBudget": "10s",

		// Shorten the diagnostic delay to speed up test execution (else we'd add
		// the default delay to each assertion about diagnostics)
		"diagnosticsDelay": "10ms",
	}

	for k, v := range e.config.Settings {
		if k == "env" {
			panic("must not provide env via the EditorConfig.Settings field: use the EditorConfig.Env field instead")
		}
		settings[k] = v
	}

	return settings
}

func (e *Editor) initialize(ctx context.Context) error {
	params := &protocol.ParamInitialize{}
	params.ClientInfo.Name = "fakeclient"
	params.ClientInfo.Version = "v1.0.0"
	e.mu.Lock()
	params.WorkspaceFolders = e.makeWorkspaceFoldersLocked()
	params.InitializationOptions = e.settingsLocked()
	e.mu.Unlock()
	params.Capabilities.Workspace.Configuration = true
	params.Capabilities.Window.WorkDoneProgress = true

	// TODO: set client capabilities
	params.Capabilities.TextDocument.Completion.CompletionItem.TagSupport.ValueSet = []protocol.CompletionItemTag{protocol.ComplDeprecated}

	params.Capabilities.TextDocument.Completion.CompletionItem.SnippetSupport = true
	params.Capabilities.TextDocument.SemanticTokens.Requests.Full = true
	// copied from lsp/semantic.go to avoid import cycle in tests
	params.Capabilities.TextDocument.SemanticTokens.TokenTypes = []string{
		"namespace", "type", "class", "enum", "interface",
		"struct", "typeParameter", "parameter", "variable", "property", "enumMember",
		"event", "function", "method", "macro", "keyword", "modifier", "comment",
		"string", "number", "regexp", "operator",
	}
	params.Capabilities.TextDocument.SemanticTokens.TokenModifiers = []string{
		"declaration", "definition", "readonly", "static",
		"deprecated", "abstract", "async", "modification", "documentation", "defaultLibrary",
	}

	// This is a bit of a hack, since the fake editor doesn't actually support
	// watching changed files that match a specific glob pattern. However, the
	// editor does send didChangeWatchedFiles notifications, so set this to
	// true.
	params.Capabilities.Workspace.DidChangeWatchedFiles.DynamicRegistration = true
	params.Capabilities.Workspace.WorkspaceEdit = &protocol.WorkspaceEditClientCapabilities{
		ResourceOperations: []protocol.ResourceOperationKind{
			"rename",
		},
	}

	params.Trace = "messages"
	// TODO: support workspace folders.
	if e.Server != nil {
		resp, err := e.Server.Initialize(ctx, params)
		if err != nil {
			return fmt.Errorf("initialize: %w", err)
		}
		e.mu.Lock()
		e.serverCapabilities = resp.Capabilities
		e.mu.Unlock()

		if err := e.Server.Initialized(ctx, &protocol.InitializedParams{}); err != nil {
			return fmt.Errorf("initialized: %w", err)
		}
	}
	// TODO: await initial configuration here, or expect gopls to manage that?
	return nil
}

// makeWorkspaceFoldersLocked creates a slice of workspace folders to use for
// this editing session, based on the editor configuration.
//
// e.mu must be held while calling this function.
func (e *Editor) makeWorkspaceFoldersLocked() (folders []protocol.WorkspaceFolder) {
	paths := e.config.WorkspaceFolders
	if len(paths) == 0 {
		paths = append(paths, string(e.sandbox.Workdir.RelativeTo))
	}

	for _, path := range paths {
		uri := string(e.sandbox.Workdir.URI(path))
		folders = append(folders, protocol.WorkspaceFolder{
			URI:  uri,
			Name: filepath.Base(uri),
		})
	}

	return folders
}

// onFileChanges is registered to be called by the Workdir on any writes that
// go through the Workdir API. It is called synchronously by the Workdir.
func (e *Editor) onFileChanges(ctx context.Context, evts []FileEvent) {
	if e.Server == nil {
		return
	}

	// e may be locked when onFileChanges is called, but it is important that we
	// synchronously increment this counter so that we can subsequently assert on
	// the number of expected DidChangeWatchedFiles calls.
	e.callsMu.Lock()
	e.calls.DidChangeWatchedFiles++
	e.callsMu.Unlock()

	// Since e may be locked, we must run this mutation asynchronously.
	go func() {
		e.mu.Lock()
		defer e.mu.Unlock()
		var lspevts []protocol.FileEvent
		for _, evt := range evts {
			// Always send an on-disk change, even for events that seem useless
			// because they're shadowed by an open buffer.
			lspevts = append(lspevts, evt.ProtocolEvent)

			if buf, ok := e.buffers[evt.Path]; ok {
				// Following VS Code, don't honor deletions or changes to dirty buffers.
				if buf.dirty || evt.ProtocolEvent.Type == protocol.Deleted {
					continue
				}

				content, err := e.sandbox.Workdir.ReadFile(evt.Path)
				if err != nil {
					continue // A race with some other operation.
				}
				// No need to update if the buffer content hasn't changed.
				if content == buf.text() {
					continue
				}
				// During shutdown, this call will fail. Ignore the error.
				_ = e.setBufferContentLocked(ctx, evt.Path, false, lines(content), nil)
			}
		}
		e.Server.DidChangeWatchedFiles(ctx, &protocol.DidChangeWatchedFilesParams{
			Changes: lspevts,
		})
	}()
}

// OpenFile creates a buffer for the given workdir-relative file.
//
// If the file is already open, it is a no-op.
func (e *Editor) OpenFile(ctx context.Context, path string) error {
	if e.HasBuffer(path) {
		return nil
	}
	content, err := e.sandbox.Workdir.ReadFile(path)
	if err != nil {
		return err
	}
	return e.createBuffer(ctx, path, false, content)
}

// CreateBuffer creates a new unsaved buffer corresponding to the workdir path,
// containing the given textual content.
func (e *Editor) CreateBuffer(ctx context.Context, path, content string) error {
	return e.createBuffer(ctx, path, true, content)
}

func (e *Editor) createBuffer(ctx context.Context, path string, dirty bool, content string) error {
	e.mu.Lock()

	if _, ok := e.buffers[path]; ok {
		e.mu.Unlock()
		return fmt.Errorf("buffer %q already exists", path)
	}

	buf := buffer{
		windowsLineEndings: e.config.WindowsLineEndings,
		version:            1,
		path:               path,
		lines:              lines(content),
		dirty:              dirty,
	}
	e.buffers[path] = buf

	item := e.textDocumentItem(buf)
	e.mu.Unlock()

	return e.sendDidOpen(ctx, item)
}

// textDocumentItem builds a protocol.TextDocumentItem for the given buffer.
//
// Precondition: e.mu must be held.
func (e *Editor) textDocumentItem(buf buffer) protocol.TextDocumentItem {
	return protocol.TextDocumentItem{
		URI:        e.sandbox.Workdir.URI(buf.path),
		LanguageID: languageID(buf.path, e.config.FileAssociations),
		Version:    int32(buf.version),
		Text:       buf.text(),
	}
}

func (e *Editor) sendDidOpen(ctx context.Context, item protocol.TextDocumentItem) error {
	if e.Server != nil {
		if err := e.Server.DidOpen(ctx, &protocol.DidOpenTextDocumentParams{
			TextDocument: item,
		}); err != nil {
			return fmt.Errorf("DidOpen: %w", err)
		}
		e.callsMu.Lock()
		e.calls.DidOpen++
		e.callsMu.Unlock()
	}
	return nil
}

var defaultFileAssociations = map[string]*regexp.Regexp{
	"go":      regexp.MustCompile(`^.*\.go$`), // '$' is important: don't match .gotmpl!
	"go.mod":  regexp.MustCompile(`^go\.mod$`),
	"go.sum":  regexp.MustCompile(`^go(\.work)?\.sum$`),
	"go.work": regexp.MustCompile(`^go\.work$`),
	"gotmpl":  regexp.MustCompile(`^.*tmpl$`),
}

// languageID returns the language identifier for the path p given the user
// configured fileAssociations.
func languageID(p string, fileAssociations map[string]string) string {
	base := path.Base(p)
	for lang, re := range fileAssociations {
		re := regexp.MustCompile(re)
		if re.MatchString(base) {
			return lang
		}
	}
	for lang, re := range defaultFileAssociations {
		if re.MatchString(base) {
			return lang
		}
	}
	return ""
}

// lines returns line-ending agnostic line representation of content.
func lines(content string) []string {
	lines := strings.Split(content, "\n")
	for i, l := range lines {
		lines[i] = strings.TrimSuffix(l, "\r")
	}
	return lines
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

	return e.sendDidClose(ctx, e.TextDocumentIdentifier(path))
}

func (e *Editor) sendDidClose(ctx context.Context, doc protocol.TextDocumentIdentifier) error {
	if e.Server != nil {
		if err := e.Server.DidClose(ctx, &protocol.DidCloseTextDocumentParams{
			TextDocument: doc,
		}); err != nil {
			return fmt.Errorf("DidClose: %w", err)
		}
		e.callsMu.Lock()
		e.calls.DidClose++
		e.callsMu.Unlock()
	}
	return nil
}

func (e *Editor) TextDocumentIdentifier(path string) protocol.TextDocumentIdentifier {
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
	return e.SaveBufferWithoutActions(ctx, path)
}

func (e *Editor) SaveBufferWithoutActions(ctx context.Context, path string) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	buf, ok := e.buffers[path]
	if !ok {
		return fmt.Errorf(fmt.Sprintf("unknown buffer: %q", path))
	}
	content := buf.text()
	includeText := false
	syncOptions, ok := e.serverCapabilities.TextDocumentSync.(protocol.TextDocumentSyncOptions)
	if ok {
		includeText = syncOptions.Save.IncludeText
	}

	docID := e.TextDocumentIdentifier(buf.path)
	if e.Server != nil {
		if err := e.Server.WillSave(ctx, &protocol.WillSaveTextDocumentParams{
			TextDocument: docID,
			Reason:       protocol.Manual,
		}); err != nil {
			return fmt.Errorf("WillSave: %w", err)
		}
	}
	if err := e.sandbox.Workdir.WriteFile(ctx, path, content); err != nil {
		return fmt.Errorf("writing %q: %w", path, err)
	}

	buf.dirty = false
	e.buffers[path] = buf

	if e.Server != nil {
		params := &protocol.DidSaveTextDocumentParams{
			TextDocument: docID,
		}
		if includeText {
			params.Text = &content
		}
		if err := e.Server.DidSave(ctx, params); err != nil {
			return fmt.Errorf("DidSave: %w", err)
		}
		e.callsMu.Lock()
		e.calls.DidSave++
		e.callsMu.Unlock()
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
	if (strings.HasSuffix(content, "\n") || content == "") && offset == start {
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
	content = normalizeEOL(content)
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

func normalizeEOL(content string) string {
	return strings.Join(lines(content), "\n")
}

// RegexpRange returns the first range in the buffer bufName matching re. See
// RegexpSearch for more information on matching.
func (e *Editor) RegexpRange(bufName, re string) (Pos, Pos, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	buf, ok := e.buffers[bufName]
	if !ok {
		return Pos{}, Pos{}, ErrUnknownBuffer
	}
	return regexpRange(buf.text(), re)
}

// RegexpSearch returns the position of the first match for re in the buffer
// bufName. For convenience, RegexpSearch supports the following two modes:
//  1. If re has no subgroups, return the position of the match for re itself.
//  2. If re has one subgroup, return the position of the first subgroup.
//
// It returns an error re is invalid, has more than one subgroup, or doesn't
// match the buffer.
func (e *Editor) RegexpSearch(bufName, re string) (Pos, error) {
	start, _, err := e.RegexpRange(bufName, re)
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

func (e *Editor) SetBufferContent(ctx context.Context, path, content string) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	lines := lines(content)
	return e.setBufferContentLocked(ctx, path, true, lines, nil)
}

// HasBuffer reports whether the file name is open in the editor.
func (e *Editor) HasBuffer(name string) bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	_, ok := e.buffers[name]
	return ok
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
	content, err := applyEdits(buf.lines, edits)
	if err != nil {
		return fmt.Errorf("editing %q: %v; edits:\n%v", path, err, edits)
	}
	return e.setBufferContentLocked(ctx, path, true, content, edits)
}

func (e *Editor) setBufferContentLocked(ctx context.Context, path string, dirty bool, content []string, fromEdits []Edit) error {
	buf, ok := e.buffers[path]
	if !ok {
		return fmt.Errorf("unknown buffer %q", path)
	}
	buf.lines = content
	buf.version++
	buf.dirty = dirty
	e.buffers[path] = buf
	// A simple heuristic: if there is only one edit, send it incrementally.
	// Otherwise, send the entire content.
	var evts []protocol.TextDocumentContentChangeEvent
	if len(fromEdits) == 1 {
		evts = append(evts, fromEdits[0].toProtocolChangeEvent())
	} else {
		evts = append(evts, protocol.TextDocumentContentChangeEvent{
			Text: buf.text(),
		})
	}
	params := &protocol.DidChangeTextDocumentParams{
		TextDocument: protocol.VersionedTextDocumentIdentifier{
			Version:                int32(buf.version),
			TextDocumentIdentifier: e.TextDocumentIdentifier(buf.path),
		},
		ContentChanges: evts,
	}
	if e.Server != nil {
		if err := e.Server.DidChange(ctx, params); err != nil {
			return fmt.Errorf("DidChange: %w", err)
		}
		e.callsMu.Lock()
		e.calls.DidChange++
		e.callsMu.Unlock()
	}
	return nil
}

// GoToDefinition jumps to the definition of the symbol at the given position
// in an open buffer. It returns the path and position of the resulting jump.
func (e *Editor) GoToDefinition(ctx context.Context, path string, pos Pos) (string, Pos, error) {
	if err := e.checkBufferPosition(path, pos); err != nil {
		return "", Pos{}, err
	}
	params := &protocol.DefinitionParams{}
	params.TextDocument.URI = e.sandbox.Workdir.URI(path)
	params.Position = pos.ToProtocolPosition()

	resp, err := e.Server.Definition(ctx, params)
	if err != nil {
		return "", Pos{}, fmt.Errorf("definition: %w", err)
	}
	return e.extractFirstPathAndPos(ctx, resp)
}

// GoToTypeDefinition jumps to the type definition of the symbol at the given position
// in an open buffer.
func (e *Editor) GoToTypeDefinition(ctx context.Context, path string, pos Pos) (string, Pos, error) {
	if err := e.checkBufferPosition(path, pos); err != nil {
		return "", Pos{}, err
	}
	params := &protocol.TypeDefinitionParams{}
	params.TextDocument.URI = e.sandbox.Workdir.URI(path)
	params.Position = pos.ToProtocolPosition()

	resp, err := e.Server.TypeDefinition(ctx, params)
	if err != nil {
		return "", Pos{}, fmt.Errorf("type definition: %w", err)
	}
	return e.extractFirstPathAndPos(ctx, resp)
}

// extractFirstPathAndPos returns the path and the position of the first location.
// It opens the file if needed.
func (e *Editor) extractFirstPathAndPos(ctx context.Context, locs []protocol.Location) (string, Pos, error) {
	if len(locs) == 0 {
		return "", Pos{}, nil
	}

	newPath := e.sandbox.Workdir.URIToPath(locs[0].URI)
	newPos := fromProtocolPosition(locs[0].Range.Start)
	if !e.HasBuffer(newPath) {
		if err := e.OpenFile(ctx, newPath); err != nil {
			return "", Pos{}, fmt.Errorf("OpenFile: %w", err)
		}
	}
	return newPath, newPos, nil
}

// Symbol performs a workspace symbol search using query
func (e *Editor) Symbol(ctx context.Context, query string) ([]SymbolInformation, error) {
	params := &protocol.WorkspaceSymbolParams{}
	params.Query = query

	resp, err := e.Server.Symbol(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("symbol: %w", err)
	}
	var res []SymbolInformation
	for _, si := range resp {
		ploc := si.Location
		path := e.sandbox.Workdir.URIToPath(ploc.URI)
		start := fromProtocolPosition(ploc.Range.Start)
		end := fromProtocolPosition(ploc.Range.End)
		rnge := Range{
			Start: start,
			End:   end,
		}
		loc := Location{
			Path:  path,
			Range: rnge,
		}
		res = append(res, SymbolInformation{
			Name:     si.Name,
			Kind:     si.Kind,
			Location: loc,
		})
	}
	return res, nil
}

// OrganizeImports requests and performs the source.organizeImports codeAction.
func (e *Editor) OrganizeImports(ctx context.Context, path string) error {
	_, err := e.applyCodeActions(ctx, path, nil, nil, protocol.SourceOrganizeImports)
	return err
}

// RefactorRewrite requests and performs the source.refactorRewrite codeAction.
func (e *Editor) RefactorRewrite(ctx context.Context, path string, rng *protocol.Range) error {
	applied, err := e.applyCodeActions(ctx, path, rng, nil, protocol.RefactorRewrite)
	if applied == 0 {
		return fmt.Errorf("no refactorings were applied")
	}
	return err
}

// ApplyQuickFixes requests and performs the quickfix codeAction.
func (e *Editor) ApplyQuickFixes(ctx context.Context, path string, rng *protocol.Range, diagnostics []protocol.Diagnostic) error {
	applied, err := e.applyCodeActions(ctx, path, rng, diagnostics, protocol.SourceFixAll, protocol.QuickFix)
	if applied == 0 {
		return fmt.Errorf("no quick fixes were applied")
	}
	return err
}

// ApplyCodeAction applies the given code action.
func (e *Editor) ApplyCodeAction(ctx context.Context, action protocol.CodeAction) error {
	for _, change := range action.Edit.DocumentChanges {
		if change.TextDocumentEdit != nil {
			path := e.sandbox.Workdir.URIToPath(change.TextDocumentEdit.TextDocument.URI)
			if int32(e.buffers[path].version) != change.TextDocumentEdit.TextDocument.Version {
				// Skip edits for old versions.
				continue
			}
			edits := convertEdits(change.TextDocumentEdit.Edits)
			if err := e.EditBuffer(ctx, path, edits); err != nil {
				return fmt.Errorf("editing buffer %q: %w", path, err)
			}
		}
	}
	// Execute any commands. The specification says that commands are
	// executed after edits are applied.
	if action.Command != nil {
		if _, err := e.ExecuteCommand(ctx, &protocol.ExecuteCommandParams{
			Command:   action.Command.Command,
			Arguments: action.Command.Arguments,
		}); err != nil {
			return err
		}
	}
	// Some commands may edit files on disk.
	return e.sandbox.Workdir.CheckForFileChanges(ctx)
}

// GetQuickFixes returns the available quick fix code actions.
func (e *Editor) GetQuickFixes(ctx context.Context, path string, rng *protocol.Range, diagnostics []protocol.Diagnostic) ([]protocol.CodeAction, error) {
	return e.getCodeActions(ctx, path, rng, diagnostics, protocol.QuickFix, protocol.SourceFixAll)
}

func (e *Editor) applyCodeActions(ctx context.Context, path string, rng *protocol.Range, diagnostics []protocol.Diagnostic, only ...protocol.CodeActionKind) (int, error) {
	actions, err := e.getCodeActions(ctx, path, rng, diagnostics, only...)
	if err != nil {
		return 0, err
	}
	applied := 0
	for _, action := range actions {
		if action.Title == "" {
			return 0, fmt.Errorf("empty title for code action")
		}
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
		applied++
		if err := e.ApplyCodeAction(ctx, action); err != nil {
			return 0, err
		}
	}
	return applied, nil
}

func (e *Editor) getCodeActions(ctx context.Context, path string, rng *protocol.Range, diagnostics []protocol.Diagnostic, only ...protocol.CodeActionKind) ([]protocol.CodeAction, error) {
	if e.Server == nil {
		return nil, nil
	}
	params := &protocol.CodeActionParams{}
	params.TextDocument.URI = e.sandbox.Workdir.URI(path)
	params.Context.Only = only
	if diagnostics != nil {
		params.Context.Diagnostics = diagnostics
	}
	if rng != nil {
		params.Range = *rng
	}
	return e.Server.CodeAction(ctx, params)
}

func (e *Editor) ExecuteCommand(ctx context.Context, params *protocol.ExecuteCommandParams) (interface{}, error) {
	if e.Server == nil {
		return nil, nil
	}
	var match bool
	// Ensure that this command was actually listed as a supported command.
	for _, command := range e.serverCapabilities.ExecuteCommandProvider.Commands {
		if command == params.Command {
			match = true
			break
		}
	}
	if !match {
		return nil, fmt.Errorf("unsupported command %q", params.Command)
	}
	result, err := e.Server.ExecuteCommand(ctx, params)
	if err != nil {
		return nil, err
	}
	// Some commands use the go command, which writes directly to disk.
	// For convenience, check for those changes.
	if err := e.sandbox.Workdir.CheckForFileChanges(ctx); err != nil {
		return nil, err
	}
	return result, nil
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
	if e.Server == nil {
		return nil
	}
	e.mu.Lock()
	version := e.buffers[path].version
	e.mu.Unlock()
	params := &protocol.DocumentFormattingParams{}
	params.TextDocument.URI = e.sandbox.Workdir.URI(path)
	resp, err := e.Server.Formatting(ctx, params)
	if err != nil {
		return fmt.Errorf("textDocument/formatting: %w", err)
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if versionAfter := e.buffers[path].version; versionAfter != version {
		return fmt.Errorf("before receipt of formatting edits, buffer version changed from %d to %d", version, versionAfter)
	}
	edits := convertEdits(resp)
	if len(edits) == 0 {
		return nil
	}
	return e.editBufferLocked(ctx, path, edits)
}

func (e *Editor) checkBufferPosition(path string, pos Pos) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	buf, ok := e.buffers[path]
	if !ok {
		return fmt.Errorf("buffer %q is not open", path)
	}
	if !inText(pos, buf.lines) {
		return fmt.Errorf("position %v is invalid in buffer %q", pos, path)
	}
	return nil
}

// RunGenerate runs `go generate` non-recursively in the workdir-relative dir
// path. It does not report any resulting file changes as a watched file
// change, so must be followed by a call to Workdir.CheckForFileChanges once
// the generate command has completed.
// TODO(rFindley): this shouldn't be necessary anymore. Delete it.
func (e *Editor) RunGenerate(ctx context.Context, dir string) error {
	if e.Server == nil {
		return nil
	}
	absDir := e.sandbox.Workdir.AbsPath(dir)
	cmd, err := command.NewGenerateCommand("", command.GenerateArgs{
		Dir:       protocol.URIFromSpanURI(span.URIFromPath(absDir)),
		Recursive: false,
	})
	if err != nil {
		return err
	}
	params := &protocol.ExecuteCommandParams{
		Command:   cmd.Command,
		Arguments: cmd.Arguments,
	}
	if _, err := e.ExecuteCommand(ctx, params); err != nil {
		return fmt.Errorf("running generate: %v", err)
	}
	// Unfortunately we can't simply poll the workdir for file changes here,
	// because server-side command may not have completed. In regtests, we can
	// Await this state change, but here we must delegate that responsibility to
	// the caller.
	return nil
}

// CodeLens executes a codelens request on the server.
func (e *Editor) CodeLens(ctx context.Context, path string) ([]protocol.CodeLens, error) {
	if e.Server == nil {
		return nil, nil
	}
	e.mu.Lock()
	_, ok := e.buffers[path]
	e.mu.Unlock()
	if !ok {
		return nil, fmt.Errorf("buffer %q is not open", path)
	}
	params := &protocol.CodeLensParams{
		TextDocument: e.TextDocumentIdentifier(path),
	}
	lens, err := e.Server.CodeLens(ctx, params)
	if err != nil {
		return nil, err
	}
	return lens, nil
}

// Completion executes a completion request on the server.
func (e *Editor) Completion(ctx context.Context, path string, pos Pos) (*protocol.CompletionList, error) {
	if e.Server == nil {
		return nil, nil
	}
	e.mu.Lock()
	_, ok := e.buffers[path]
	e.mu.Unlock()
	if !ok {
		return nil, fmt.Errorf("buffer %q is not open", path)
	}
	params := &protocol.CompletionParams{
		TextDocumentPositionParams: protocol.TextDocumentPositionParams{
			TextDocument: e.TextDocumentIdentifier(path),
			Position:     pos.ToProtocolPosition(),
		},
	}
	completions, err := e.Server.Completion(ctx, params)
	if err != nil {
		return nil, err
	}
	return completions, nil
}

// AcceptCompletion accepts a completion for the given item at the given
// position.
func (e *Editor) AcceptCompletion(ctx context.Context, path string, pos Pos, item protocol.CompletionItem) error {
	if e.Server == nil {
		return nil
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	_, ok := e.buffers[path]
	if !ok {
		return fmt.Errorf("buffer %q is not open", path)
	}
	return e.editBufferLocked(ctx, path, convertEdits(append([]protocol.TextEdit{
		*item.TextEdit,
	}, item.AdditionalTextEdits...)))
}

// Symbols executes a workspace/symbols request on the server.
func (e *Editor) Symbols(ctx context.Context, sym string) ([]protocol.SymbolInformation, error) {
	if e.Server == nil {
		return nil, nil
	}
	params := &protocol.WorkspaceSymbolParams{Query: sym}
	ans, err := e.Server.Symbol(ctx, params)
	return ans, err
}

// CodeLens executes a codelens request on the server.
func (e *Editor) InlayHint(ctx context.Context, path string) ([]protocol.InlayHint, error) {
	if e.Server == nil {
		return nil, nil
	}
	e.mu.Lock()
	_, ok := e.buffers[path]
	e.mu.Unlock()
	if !ok {
		return nil, fmt.Errorf("buffer %q is not open", path)
	}
	params := &protocol.InlayHintParams{
		TextDocument: e.TextDocumentIdentifier(path),
	}
	hints, err := e.Server.InlayHint(ctx, params)
	if err != nil {
		return nil, err
	}
	return hints, nil
}

// References returns references to the object at (path, pos), as returned by
// the connected LSP server. If no server is connected, it returns (nil, nil).
func (e *Editor) References(ctx context.Context, path string, pos Pos) ([]protocol.Location, error) {
	if e.Server == nil {
		return nil, nil
	}
	e.mu.Lock()
	_, ok := e.buffers[path]
	e.mu.Unlock()
	if !ok {
		return nil, fmt.Errorf("buffer %q is not open", path)
	}
	params := &protocol.ReferenceParams{
		TextDocumentPositionParams: protocol.TextDocumentPositionParams{
			TextDocument: e.TextDocumentIdentifier(path),
			Position:     pos.ToProtocolPosition(),
		},
		Context: protocol.ReferenceContext{
			IncludeDeclaration: true,
		},
	}
	locations, err := e.Server.References(ctx, params)
	if err != nil {
		return nil, err
	}
	return locations, nil
}

// Rename performs a rename of the object at (path, pos) to newName, using the
// connected LSP server. If no server is connected, it returns nil.
func (e *Editor) Rename(ctx context.Context, path string, pos Pos, newName string) error {
	if e.Server == nil {
		return nil
	}

	// Verify that PrepareRename succeeds.
	prepareParams := &protocol.PrepareRenameParams{}
	prepareParams.TextDocument = e.TextDocumentIdentifier(path)
	prepareParams.Position = pos.ToProtocolPosition()
	if _, err := e.Server.PrepareRename(ctx, prepareParams); err != nil {
		return fmt.Errorf("preparing rename: %v", err)
	}

	params := &protocol.RenameParams{
		TextDocument: e.TextDocumentIdentifier(path),
		Position:     pos.ToProtocolPosition(),
		NewName:      newName,
	}
	wsEdits, err := e.Server.Rename(ctx, params)
	if err != nil {
		return err
	}
	for _, change := range wsEdits.DocumentChanges {
		if err := e.applyDocumentChange(ctx, change); err != nil {
			return err
		}
	}
	return nil
}

// Implementations returns implementations for the object at (path, pos), as
// returned by the connected LSP server. If no server is connected, it returns
// (nil, nil).
func (e *Editor) Implementations(ctx context.Context, path string, pos Pos) ([]protocol.Location, error) {
	if e.Server == nil {
		return nil, nil
	}
	e.mu.Lock()
	_, ok := e.buffers[path]
	e.mu.Unlock()
	if !ok {
		return nil, fmt.Errorf("buffer %q is not open", path)
	}
	params := &protocol.ImplementationParams{
		TextDocumentPositionParams: protocol.TextDocumentPositionParams{
			TextDocument: e.TextDocumentIdentifier(path),
			Position:     pos.ToProtocolPosition(),
		},
	}
	return e.Server.Implementation(ctx, params)
}

func (e *Editor) RenameFile(ctx context.Context, oldPath, newPath string) error {
	closed, opened, err := e.renameBuffers(ctx, oldPath, newPath)
	if err != nil {
		return err
	}

	for _, c := range closed {
		if err := e.sendDidClose(ctx, c); err != nil {
			return err
		}
	}
	for _, o := range opened {
		if err := e.sendDidOpen(ctx, o); err != nil {
			return err
		}
	}

	// Finally, perform the renaming on disk.
	if err := e.sandbox.Workdir.RenameFile(ctx, oldPath, newPath); err != nil {
		return fmt.Errorf("renaming sandbox file: %w", err)
	}
	return nil
}

// renameBuffers renames in-memory buffers affected by the renaming of
// oldPath->newPath, returning the resulting text documents that must be closed
// and opened over the LSP.
func (e *Editor) renameBuffers(ctx context.Context, oldPath, newPath string) (closed []protocol.TextDocumentIdentifier, opened []protocol.TextDocumentItem, _ error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	// In case either oldPath or newPath is absolute, convert to absolute paths
	// before checking for containment.
	oldAbs := e.sandbox.Workdir.AbsPath(oldPath)
	newAbs := e.sandbox.Workdir.AbsPath(newPath)

	// Collect buffers that are affected by the given file or directory renaming.
	buffersToRename := make(map[string]string) // old path -> new path

	for path := range e.buffers {
		abs := e.sandbox.Workdir.AbsPath(path)
		if oldAbs == abs || source.InDir(oldAbs, abs) {
			rel, err := filepath.Rel(oldAbs, abs)
			if err != nil {
				return nil, nil, fmt.Errorf("filepath.Rel(%q, %q): %v", oldAbs, abs, err)
			}
			nabs := filepath.Join(newAbs, rel)
			newPath := e.sandbox.Workdir.RelPath(nabs)
			buffersToRename[path] = newPath
		}
	}

	// Update buffers, and build protocol changes.
	for old, new := range buffersToRename {
		buf := e.buffers[old]
		delete(e.buffers, old)
		buf.version = 1
		buf.path = new
		e.buffers[new] = buf

		closed = append(closed, e.TextDocumentIdentifier(old))
		opened = append(opened, e.textDocumentItem(buf))
	}

	return closed, opened, nil
}

func (e *Editor) applyDocumentChange(ctx context.Context, change protocol.DocumentChanges) error {
	if change.RenameFile != nil {
		oldPath := e.sandbox.Workdir.URIToPath(change.RenameFile.OldURI)
		newPath := e.sandbox.Workdir.URIToPath(change.RenameFile.NewURI)

		return e.RenameFile(ctx, oldPath, newPath)
	}
	if change.TextDocumentEdit != nil {
		return e.applyTextDocumentEdit(ctx, *change.TextDocumentEdit)
	}
	panic("Internal error: one of RenameFile or TextDocumentEdit must be set")
}

func (e *Editor) applyTextDocumentEdit(ctx context.Context, change protocol.TextDocumentEdit) error {
	path := e.sandbox.Workdir.URIToPath(change.TextDocument.URI)
	if ver := int32(e.BufferVersion(path)); ver != change.TextDocument.Version {
		return fmt.Errorf("buffer versions for %q do not match: have %d, editing %d", path, ver, change.TextDocument.Version)
	}
	if !e.HasBuffer(path) {
		err := e.OpenFile(ctx, path)
		if os.IsNotExist(err) {
			// TODO: it's unclear if this is correct. Here we create the buffer (with
			// version 1), then apply edits. Perhaps we should apply the edits before
			// sending the didOpen notification.
			e.CreateBuffer(ctx, path, "")
			err = nil
		}
		if err != nil {
			return err
		}
	}
	fakeEdits := convertEdits(change.Edits)
	return e.EditBuffer(ctx, path, fakeEdits)
}

// Config returns the current editor configuration.
func (e *Editor) Config() EditorConfig {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.config
}

// ChangeConfiguration sets the new editor configuration, and if applicable
// sends a didChangeConfiguration notification.
//
// An error is returned if the change notification failed to send.
func (e *Editor) ChangeConfiguration(ctx context.Context, newConfig EditorConfig) error {
	e.mu.Lock()
	e.config = newConfig
	e.mu.Unlock() // don't hold e.mu during server calls
	if e.Server != nil {
		var params protocol.DidChangeConfigurationParams // empty: gopls ignores the Settings field
		if err := e.Server.DidChangeConfiguration(ctx, &params); err != nil {
			return err
		}
	}
	return nil
}

// ChangeWorkspaceFolders sets the new workspace folders, and sends a
// didChangeWorkspaceFolders notification to the server.
//
// The given folders must all be unique.
func (e *Editor) ChangeWorkspaceFolders(ctx context.Context, folders []string) error {
	// capture existing folders so that we can compute the change.
	e.mu.Lock()
	oldFolders := e.makeWorkspaceFoldersLocked()
	e.config.WorkspaceFolders = folders
	newFolders := e.makeWorkspaceFoldersLocked()
	e.mu.Unlock()

	if e.Server == nil {
		return nil
	}

	var params protocol.DidChangeWorkspaceFoldersParams

	// Keep track of old workspace folders that must be removed.
	toRemove := make(map[protocol.URI]protocol.WorkspaceFolder)
	for _, folder := range oldFolders {
		toRemove[folder.URI] = folder
	}

	// Sanity check: if we see a folder twice the algorithm below doesn't work,
	// so track seen folders to ensure that we panic in that case.
	seen := make(map[protocol.URI]protocol.WorkspaceFolder)
	for _, folder := range newFolders {
		if _, ok := seen[folder.URI]; ok {
			panic(fmt.Sprintf("folder %s seen twice", folder.URI))
		}

		// If this folder already exists, we don't want to remove it.
		// Otherwise, we need to add it.
		if _, ok := toRemove[folder.URI]; ok {
			delete(toRemove, folder.URI)
		} else {
			params.Event.Added = append(params.Event.Added, folder)
		}
	}

	for _, v := range toRemove {
		params.Event.Removed = append(params.Event.Removed, v)
	}

	return e.Server.DidChangeWorkspaceFolders(ctx, &params)
}

// CodeAction executes a codeAction request on the server.
func (e *Editor) CodeAction(ctx context.Context, path string, rng *protocol.Range, diagnostics []protocol.Diagnostic) ([]protocol.CodeAction, error) {
	if e.Server == nil {
		return nil, nil
	}
	e.mu.Lock()
	_, ok := e.buffers[path]
	e.mu.Unlock()
	if !ok {
		return nil, fmt.Errorf("buffer %q is not open", path)
	}
	params := &protocol.CodeActionParams{
		TextDocument: e.TextDocumentIdentifier(path),
		Context: protocol.CodeActionContext{
			Diagnostics: diagnostics,
		},
	}
	if rng != nil {
		params.Range = *rng
	}
	lens, err := e.Server.CodeAction(ctx, params)
	if err != nil {
		return nil, err
	}
	return lens, nil
}

// Hover triggers a hover at the given position in an open buffer.
func (e *Editor) Hover(ctx context.Context, path string, pos Pos) (*protocol.MarkupContent, Pos, error) {
	if err := e.checkBufferPosition(path, pos); err != nil {
		return nil, Pos{}, err
	}
	params := &protocol.HoverParams{}
	params.TextDocument.URI = e.sandbox.Workdir.URI(path)
	params.Position = pos.ToProtocolPosition()

	resp, err := e.Server.Hover(ctx, params)
	if err != nil {
		return nil, Pos{}, fmt.Errorf("hover: %w", err)
	}
	if resp == nil {
		return nil, Pos{}, nil
	}
	return &resp.Contents, fromProtocolPosition(resp.Range.Start), nil
}

func (e *Editor) DocumentLink(ctx context.Context, path string) ([]protocol.DocumentLink, error) {
	if e.Server == nil {
		return nil, nil
	}
	params := &protocol.DocumentLinkParams{}
	params.TextDocument.URI = e.sandbox.Workdir.URI(path)
	return e.Server.DocumentLink(ctx, params)
}

func (e *Editor) DocumentHighlight(ctx context.Context, path string, pos Pos) ([]protocol.DocumentHighlight, error) {
	if e.Server == nil {
		return nil, nil
	}
	if err := e.checkBufferPosition(path, pos); err != nil {
		return nil, err
	}
	params := &protocol.DocumentHighlightParams{}
	params.TextDocument.URI = e.sandbox.Workdir.URI(path)
	params.Position = pos.ToProtocolPosition()

	return e.Server.DocumentHighlight(ctx, params)
}
