// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fake

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"

	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
	errors "golang.org/x/xerrors"
)

// Editor is a fake editor client.  It keeps track of client state and can be
// used for writing LSP tests.
type Editor struct {
	Config EditorConfig

	// Server, client, and sandbox are concurrency safe and written only
	// at construction time, so do not require synchronization.
	Server     protocol.Server
	serverConn jsonrpc2.Conn
	client     *Client
	sandbox    *Sandbox
	defaultEnv map[string]string

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
	dirty   bool
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
	Env        map[string]string
	BuildFlags []string

	// CodeLenses is a map defining whether codelens are enabled, keyed by the
	// codeLens command. CodeLenses which are not present in this map are left in
	// their default state.
	CodeLenses map[string]bool

	// SymbolMatcher is the config associated with the "symbolMatcher" gopls
	// config option.
	SymbolMatcher, SymbolStyle *string

	// LimitWorkspaceScope is true if the user does not want to expand their
	// workspace scope to the entire module.
	LimitWorkspaceScope bool

	// WithoutWorkspaceFolders is used to simulate opening a single file in the
	// editor, without a workspace root. In that case, the client sends neither
	// workspace folders nor a root URI.
	WithoutWorkspaceFolders bool

	// WorkspaceRoot specifies the root path of the workspace folder used when
	// initializing gopls in the sandbox. If empty, the Workdir is used.
	WorkspaceRoot string

	// EnableStaticcheck enables staticcheck analyzers.
	EnableStaticcheck bool

	// AllExperiments sets the "allExperiments" configuration, which enables
	// all of gopls's opt-in settings.
	AllExperiments bool

	// Whether to send the current process ID, for testing data that is joined to
	// the PID. This can only be set by one test.
	SendPID bool

	VerboseOutput bool
}

// NewEditor Creates a new Editor.
func NewEditor(sandbox *Sandbox, config EditorConfig) *Editor {
	return &Editor{
		buffers:    make(map[string]buffer),
		sandbox:    sandbox,
		defaultEnv: sandbox.GoEnv(),
		Config:     config,
	}
}

// Connect configures the editor to communicate with an LSP server on conn. It
// is not concurrency safe, and should be called at most once, before using the
// editor.
//
// It returns the editor, so that it may be called as follows:
//   editor, err := NewEditor(s).Connect(ctx, conn)
func (e *Editor) Connect(ctx context.Context, conn jsonrpc2.Conn, hooks ClientHooks) (*Editor, error) {
	e.serverConn = conn
	e.Server = protocol.ServerDispatcher(conn)
	e.client = &Client{editor: e, hooks: hooks}
	conn.Go(ctx,
		protocol.Handlers(
			protocol.ClientHandler(e.client,
				jsonrpc2.MethodNotFound)))
	if err := e.initialize(ctx, e.Config.WithoutWorkspaceFolders, e.Config.WorkspaceRoot); err != nil {
		return nil, err
	}
	e.sandbox.Workdir.AddWatcher(e.onFileChanges)
	return e, nil
}

// Shutdown issues the 'shutdown' LSP notification.
func (e *Editor) Shutdown(ctx context.Context) error {
	if e.Server != nil {
		if err := e.Server.Shutdown(ctx); err != nil {
			return errors.Errorf("Shutdown: %w", err)
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
			return errors.Errorf("Exit: %w", err)
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
	// called close on the editor should result in the connection closing
	select {
	case <-e.serverConn.Done():
		// connection closed itself
		return nil
	case <-ctx.Done():
		return errors.Errorf("connection not closed: %w", ctx.Err())
	}
}

// Client returns the LSP client for this editor.
func (e *Editor) Client() *Client {
	return e.client
}

func (e *Editor) overlayEnv() map[string]string {
	env := make(map[string]string)
	for k, v := range e.defaultEnv {
		env[k] = v
	}
	for k, v := range e.Config.Env {
		env[k] = v
	}
	return env
}

func (e *Editor) configuration() map[string]interface{} {
	config := map[string]interface{}{
		"verboseWorkDoneProgress": true,
		"env":                     e.overlayEnv(),
		"expandWorkspaceToModule": !e.Config.LimitWorkspaceScope,
		"completionBudget":        "10s",
	}

	if e.Config.BuildFlags != nil {
		config["buildFlags"] = e.Config.BuildFlags
	}

	if e.Config.CodeLenses != nil {
		config["codelenses"] = e.Config.CodeLenses
	}
	if e.Config.SymbolMatcher != nil {
		config["symbolMatcher"] = *e.Config.SymbolMatcher
	}
	if e.Config.SymbolStyle != nil {
		config["symbolStyle"] = *e.Config.SymbolStyle
	}
	if e.Config.EnableStaticcheck {
		config["staticcheck"] = true
	}
	if e.Config.AllExperiments {
		config["allExperiments"] = true
	}

	if e.Config.VerboseOutput {
		config["verboseOutput"] = true
	}

	// TODO(rFindley): change to the new settings name once it is no longer
	// designated experimental.
	config["experimentalDiagnosticsDelay"] = "10ms"

	// ExperimentalWorkspaceModule is only set as a mode, not a configuration.
	return config
}

func (e *Editor) initialize(ctx context.Context, withoutWorkspaceFolders bool, editorRootPath string) error {
	params := &protocol.ParamInitialize{}
	params.ClientInfo.Name = "fakeclient"
	params.ClientInfo.Version = "v1.0.0"
	if !withoutWorkspaceFolders {
		rootURI := e.sandbox.Workdir.RootURI()
		if editorRootPath != "" {
			rootURI = toURI(e.sandbox.Workdir.AbsPath(editorRootPath))
		}
		params.WorkspaceFolders = []protocol.WorkspaceFolder{{
			URI:  string(rootURI),
			Name: filepath.Base(rootURI.SpanURI().Filename()),
		}}
	}
	params.Capabilities.Workspace.Configuration = true
	params.Capabilities.Window.WorkDoneProgress = true
	// TODO: set client capabilities
	params.InitializationOptions = e.configuration()
	if e.Config.SendPID {
		params.ProcessID = float64(os.Getpid())
	}

	// This is a bit of a hack, since the fake editor doesn't actually support
	// watching changed files that match a specific glob pattern. However, the
	// editor does send didChangeWatchedFiles notifications, so set this to
	// true.
	params.Capabilities.Workspace.DidChangeWatchedFiles.DynamicRegistration = true

	params.Trace = "messages"
	// TODO: support workspace folders.
	if e.Server != nil {
		resp, err := e.Server.Initialize(ctx, params)
		if err != nil {
			return errors.Errorf("initialize: %w", err)
		}
		e.mu.Lock()
		e.serverCapabilities = resp.Capabilities
		e.mu.Unlock()

		if err := e.Server.Initialized(ctx, &protocol.InitializedParams{}); err != nil {
			return errors.Errorf("initialized: %w", err)
		}
	}
	// TODO: await initial configuration here, or expect gopls to manage that?
	return nil
}

func (e *Editor) onFileChanges(ctx context.Context, evts []FileEvent) {
	if e.Server == nil {
		return
	}
	e.mu.Lock()
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
			// During shutdown, this call will fail. Ignore the error.
			_ = e.setBufferContentLocked(ctx, evt.Path, false, strings.Split(content, "\n"), nil)
		}
	}
	e.mu.Unlock()
	e.Server.DidChangeWatchedFiles(ctx, &protocol.DidChangeWatchedFilesParams{
		Changes: lspevts,
	})
}

// OpenFile creates a buffer for the given workdir-relative file.
func (e *Editor) OpenFile(ctx context.Context, path string) error {
	content, err := e.sandbox.Workdir.ReadFile(path)
	if err != nil {
		return err
	}
	return e.createBuffer(ctx, path, false, content)
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
	return e.createBuffer(ctx, path, true, content)
}

func (e *Editor) createBuffer(ctx context.Context, path string, dirty bool, content string) error {
	buf := buffer{
		version: 1,
		path:    path,
		content: strings.Split(content, "\n"),
		dirty:   dirty,
	}
	e.mu.Lock()
	e.buffers[path] = buf
	item := textDocumentItem(e.sandbox.Workdir, buf)
	e.mu.Unlock()

	if e.Server != nil {
		if err := e.Server.DidOpen(ctx, &protocol.DidOpenTextDocumentParams{
			TextDocument: item,
		}); err != nil {
			return errors.Errorf("DidOpen: %w", err)
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

	if e.Server != nil {
		if err := e.Server.DidClose(ctx, &protocol.DidCloseTextDocumentParams{
			TextDocument: e.textDocumentIdentifier(path),
		}); err != nil {
			return errors.Errorf("DidClose: %w", err)
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
		return errors.Errorf("organizing imports before save: %w", err)
	}
	if err := e.FormatBuffer(ctx, path); err != nil {
		return errors.Errorf("formatting before save: %w", err)
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

	docID := e.textDocumentIdentifier(buf.path)
	if e.Server != nil {
		if err := e.Server.WillSave(ctx, &protocol.WillSaveTextDocumentParams{
			TextDocument: docID,
			Reason:       protocol.Manual,
		}); err != nil {
			return errors.Errorf("WillSave: %w", err)
		}
	}
	if err := e.sandbox.Workdir.WriteFile(ctx, path, content); err != nil {
		return errors.Errorf("writing %q: %w", path, err)
	}

	buf.dirty = false
	e.buffers[path] = buf

	if e.Server != nil {
		params := &protocol.DidSaveTextDocumentParams{
			TextDocument: protocol.VersionedTextDocumentIdentifier{
				Version:                float64(buf.version),
				TextDocumentIdentifier: docID,
			},
		}
		if includeText {
			params.Text = &content
		}
		if err := e.Server.DidSave(ctx, params); err != nil {
			return errors.Errorf("DidSave: %w", err)
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
		return Pos{}, errors.Errorf("scanning content: %w", err)
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
	lines := strings.Split(content, "\n")
	return e.setBufferContentLocked(ctx, path, true, lines, nil)
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
	content := make([]string, len(buf.content))
	copy(content, buf.content)
	content, err := editContent(content, edits)
	if err != nil {
		return err
	}
	return e.setBufferContentLocked(ctx, path, true, content, edits)
}

func (e *Editor) setBufferContentLocked(ctx context.Context, path string, dirty bool, content []string, fromEdits []Edit) error {
	buf, ok := e.buffers[path]
	if !ok {
		return fmt.Errorf("unknown buffer %q", path)
	}
	buf.content = content
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
			Version:                float64(buf.version),
			TextDocumentIdentifier: e.textDocumentIdentifier(buf.path),
		},
		ContentChanges: evts,
	}
	if e.Server != nil {
		if err := e.Server.DidChange(ctx, params); err != nil {
			return errors.Errorf("DidChange: %w", err)
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
	params.Position = pos.ToProtocolPosition()

	resp, err := e.Server.Definition(ctx, params)
	if err != nil {
		return "", Pos{}, errors.Errorf("definition: %w", err)
	}
	if len(resp) == 0 {
		return "", Pos{}, nil
	}
	newPath := e.sandbox.Workdir.URIToPath(resp[0].URI)
	newPos := fromProtocolPosition(resp[0].Range.Start)
	if err := e.OpenFile(ctx, newPath); err != nil {
		return "", Pos{}, errors.Errorf("OpenFile: %w", err)
	}
	return newPath, newPos, nil
}

// Symbol performs a workspace symbol search using query
func (e *Editor) Symbol(ctx context.Context, query string) ([]SymbolInformation, error) {
	params := &protocol.WorkspaceSymbolParams{}
	params.Query = query

	resp, err := e.Server.Symbol(ctx, params)
	if err != nil {
		return nil, errors.Errorf("symbol: %w", err)
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
	return e.codeAction(ctx, path, nil, nil, protocol.SourceOrganizeImports)
}

// RefactorRewrite requests and performs the source.refactorRewrite codeAction.
func (e *Editor) RefactorRewrite(ctx context.Context, path string, rng *protocol.Range) error {
	return e.codeAction(ctx, path, rng, nil, protocol.RefactorRewrite)
}

// ApplyQuickFixes requests and performs the quickfix codeAction.
func (e *Editor) ApplyQuickFixes(ctx context.Context, path string, rng *protocol.Range, diagnostics []protocol.Diagnostic) error {
	return e.codeAction(ctx, path, rng, diagnostics, protocol.QuickFix, protocol.SourceFixAll)
}

func (e *Editor) codeAction(ctx context.Context, path string, rng *protocol.Range, diagnostics []protocol.Diagnostic, only ...protocol.CodeActionKind) error {
	if e.Server == nil {
		return nil
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
	actions, err := e.Server.CodeAction(ctx, params)
	if err != nil {
		return errors.Errorf("textDocument/codeAction: %w", err)
	}
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
			if err := e.EditBuffer(ctx, path, edits); err != nil {
				return errors.Errorf("editing buffer %q: %w", path, err)
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
	}
	return nil
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
		return errors.Errorf("textDocument/formatting: %w", err)
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
	if e.Server == nil {
		return nil
	}
	absDir := e.sandbox.Workdir.AbsPath(dir)
	jsonArgs, err := source.MarshalArgs(span.URIFromPath(absDir), false)
	if err != nil {
		return err
	}
	params := &protocol.ExecuteCommandParams{
		Command:   source.CommandGenerate.ID(),
		Arguments: jsonArgs,
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
		TextDocument: e.textDocumentIdentifier(path),
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
			TextDocument: e.textDocumentIdentifier(path),
			Position:     pos.ToProtocolPosition(),
		},
	}
	completions, err := e.Server.Completion(ctx, params)
	if err != nil {
		return nil, err
	}
	return completions, nil
}

// References executes a reference request on the server.
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
			TextDocument: e.textDocumentIdentifier(path),
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

// CodeAction executes a codeAction request on the server.
func (e *Editor) CodeAction(ctx context.Context, path string, rng *protocol.Range) ([]protocol.CodeAction, error) {
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
		TextDocument: e.textDocumentIdentifier(path),
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
		return nil, Pos{}, errors.Errorf("hover: %w", err)
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
