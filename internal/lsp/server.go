// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"bytes"
	"context"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"net"
	"os"
	"path"
	"strings"
	"sync"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/lsp/cache"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/xlog"
	"golang.org/x/tools/internal/span"
)

// NewClientServer
func NewClientServer(client protocol.Client) *Server {
	return &Server{
		client: client,
		log:    xlog.New(protocol.NewLogger(client)),
	}
}

// NewServer starts an LSP server on the supplied stream, and waits until the
// stream is closed.
func NewServer(stream jsonrpc2.Stream) *Server {
	s := &Server{}
	s.Conn, s.client, s.log = protocol.NewServer(stream, s)
	return s
}

// RunServerOnPort starts an LSP server on the given port and does not exit.
// This function exists for debugging purposes.
func RunServerOnPort(ctx context.Context, port int, h func(s *Server)) error {
	return RunServerOnAddress(ctx, fmt.Sprintf(":%v", port), h)
}

// RunServerOnPort starts an LSP server on the given port and does not exit.
// This function exists for debugging purposes.
func RunServerOnAddress(ctx context.Context, addr string, h func(s *Server)) error {
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return err
	}
	for {
		conn, err := ln.Accept()
		if err != nil {
			return err
		}
		stream := jsonrpc2.NewHeaderStream(conn, conn)
		s := NewServer(stream)
		h(s)
		go s.Run(ctx)
	}
}

type Server struct {
	Conn   *jsonrpc2.Conn
	client protocol.Client
	log    xlog.Logger

	initializedMu sync.Mutex
	initialized   bool // set once the server has received "initialize" request

	// Configurations.
	// TODO(rstambler): Separate these into their own struct?
	usePlaceholders               bool
	snippetsSupported             bool
	configurationSupported        bool
	dynamicConfigurationSupported bool

	textDocumentSyncKind protocol.TextDocumentSyncKind

	views []*cache.View

	// undelivered is a cache of any diagnostics that the server
	// failed to deliver for some reason.
	undeliveredMu sync.Mutex
	undelivered   map[span.URI][]source.Diagnostic
}

func (s *Server) Run(ctx context.Context) error {
	return s.Conn.Run(ctx)
}

func (s *Server) Initialize(ctx context.Context, params *protocol.InitializeParams) (*protocol.InitializeResult, error) {
	s.initializedMu.Lock()
	defer s.initializedMu.Unlock()
	if s.initialized {
		return nil, jsonrpc2.NewErrorf(jsonrpc2.CodeInvalidRequest, "server already initialized")
	}
	s.initialized = true // mark server as initialized now

	// TODO(rstambler): Change this default to protocol.Incremental (or add a
	// flag). Disabled for now to simplify debugging.
	s.textDocumentSyncKind = protocol.Full

	s.setClientCapabilities(params.Capabilities)

	// We need a "detached" context so it does not get timeout cancelled.
	// TODO(iancottrell): Do we need to copy any values across?
	viewContext := context.Background()
	folders := params.WorkspaceFolders
	if len(folders) == 0 {
		if params.RootURI != "" {
			folders = []protocol.WorkspaceFolder{{
				URI:  params.RootURI,
				Name: path.Base(params.RootURI),
			}}
		} else {
			// no folders and no root, single file mode
			//TODO(iancottrell): not sure how to do single file mode yet
			//issue: golang.org/issue/31168
			return nil, fmt.Errorf("single file mode not supported yet")
		}
	}
	for _, folder := range folders {
		uri := span.NewURI(folder.URI)
		folderPath, err := uri.Filename()
		if err != nil {
			return nil, err
		}
		s.views = append(s.views, cache.NewView(viewContext, s.log, folder.Name, uri, &packages.Config{
			Context: ctx,
			Dir:     folderPath,
			Env:     os.Environ(),
			Mode:    packages.LoadImports,
			Fset:    token.NewFileSet(),
			Overlay: make(map[string][]byte),
			ParseFile: func(fset *token.FileSet, filename string, src []byte) (*ast.File, error) {
				return parser.ParseFile(fset, filename, src, parser.AllErrors|parser.ParseComments)
			},
			Tests: true,
		}))
	}

	return &protocol.InitializeResult{
		Capabilities: protocol.ServerCapabilities{
			CodeActionProvider: true,
			CompletionProvider: &protocol.CompletionOptions{
				TriggerCharacters: []string{"."},
			},
			DefinitionProvider:              true,
			DocumentFormattingProvider:      true,
			DocumentRangeFormattingProvider: true,
			DocumentSymbolProvider:          true,
			HoverProvider:                   true,
			DocumentHighlightProvider:       true,
			SignatureHelpProvider: &protocol.SignatureHelpOptions{
				TriggerCharacters: []string{"(", ","},
			},
			TextDocumentSync: &protocol.TextDocumentSyncOptions{
				Change:    s.textDocumentSyncKind,
				OpenClose: true,
			},
			TypeDefinitionProvider: true,
		},
	}, nil
}

func (s *Server) setClientCapabilities(caps protocol.ClientCapabilities) {
	// Check if the client supports snippets in completion items.
	s.snippetsSupported = caps.TextDocument.Completion.CompletionItem.SnippetSupport
	// Check if the client supports configuration messages.
	s.configurationSupported = caps.Workspace.Configuration
	s.dynamicConfigurationSupported = caps.Workspace.DidChangeConfiguration.DynamicRegistration
}

func (s *Server) Initialized(ctx context.Context, params *protocol.InitializedParams) error {
	if s.configurationSupported {
		if s.dynamicConfigurationSupported {
			s.client.RegisterCapability(ctx, &protocol.RegistrationParams{
				Registrations: []protocol.Registration{{
					ID:     "workspace/didChangeConfiguration",
					Method: "workspace/didChangeConfiguration",
				}},
			})
		}
		for _, view := range s.views {
			config, err := s.client.Configuration(ctx, &protocol.ConfigurationParams{
				Items: []protocol.ConfigurationItem{{
					ScopeURI: protocol.NewURI(view.Folder),
					Section:  "gopls",
				}},
			})
			if err != nil {
				return err
			}
			if err := s.processConfig(view, config[0]); err != nil {
				return err
			}
		}
	}
	return nil
}

func (s *Server) Shutdown(context.Context) error {
	s.initializedMu.Lock()
	defer s.initializedMu.Unlock()
	if !s.initialized {
		return jsonrpc2.NewErrorf(jsonrpc2.CodeInvalidRequest, "server not initialized")
	}
	s.initialized = false
	return nil
}

func (s *Server) Exit(ctx context.Context) error {
	if s.initialized {
		os.Exit(1)
	}
	os.Exit(0)
	return nil
}

func (s *Server) DidChangeWorkspaceFolders(context.Context, *protocol.DidChangeWorkspaceFoldersParams) error {
	return notImplemented("DidChangeWorkspaceFolders")
}

func (s *Server) DidChangeConfiguration(context.Context, *protocol.DidChangeConfigurationParams) error {
	return notImplemented("DidChangeConfiguration")
}

func (s *Server) DidChangeWatchedFiles(context.Context, *protocol.DidChangeWatchedFilesParams) error {
	return notImplemented("DidChangeWatchedFiles")
}

func (s *Server) Symbols(context.Context, *protocol.WorkspaceSymbolParams) ([]protocol.SymbolInformation, error) {
	return nil, notImplemented("Symbols")
}

func (s *Server) ExecuteCommand(context.Context, *protocol.ExecuteCommandParams) (interface{}, error) {
	return nil, notImplemented("ExecuteCommand")
}

func (s *Server) DidOpen(ctx context.Context, params *protocol.DidOpenTextDocumentParams) error {
	return s.cacheAndDiagnose(ctx, span.NewURI(params.TextDocument.URI), params.TextDocument.Text)
}

func (s *Server) applyChanges(ctx context.Context, params *protocol.DidChangeTextDocumentParams) (string, error) {
	if len(params.ContentChanges) == 1 && params.ContentChanges[0].Range == nil {
		// If range is empty, we expect the full content of file, i.e. a single change with no range.
		change := params.ContentChanges[0]
		if change.RangeLength != 0 {
			return "", jsonrpc2.NewErrorf(jsonrpc2.CodeInternalError, "unexpected change range provided")
		}
		return change.Text, nil
	}

	uri := span.NewURI(params.TextDocument.URI)
	view := s.findView(ctx, uri)
	file, m, err := newColumnMap(ctx, view, uri)
	if err != nil {
		return "", jsonrpc2.NewErrorf(jsonrpc2.CodeInternalError, "file not found")
	}
	content := file.GetContent(ctx)
	for _, change := range params.ContentChanges {
		spn, err := m.RangeSpan(*change.Range)
		if err != nil {
			return "", err
		}
		if !spn.HasOffset() {
			return "", jsonrpc2.NewErrorf(jsonrpc2.CodeInternalError, "invalid range for content change")
		}
		start, end := spn.Start().Offset(), spn.End().Offset()
		if end <= start {
			return "", jsonrpc2.NewErrorf(jsonrpc2.CodeInternalError, "invalid range for content change")
		}
		var buf bytes.Buffer
		buf.Write(content[:start])
		buf.WriteString(change.Text)
		buf.Write(content[end:])
		content = buf.Bytes()
	}
	return string(content), nil
}

func (s *Server) DidChange(ctx context.Context, params *protocol.DidChangeTextDocumentParams) error {
	if len(params.ContentChanges) < 1 {
		return jsonrpc2.NewErrorf(jsonrpc2.CodeInternalError, "no content changes provided")
	}

	var text string
	switch s.textDocumentSyncKind {
	case protocol.Incremental:
		var err error
		text, err = s.applyChanges(ctx, params)
		if err != nil {
			return err
		}
	case protocol.Full:
		// We expect the full content of file, i.e. a single change with no range.
		change := params.ContentChanges[0]
		if change.RangeLength != 0 {
			return jsonrpc2.NewErrorf(jsonrpc2.CodeInternalError, "unexpected change range provided")
		}
		text = change.Text
	}
	return s.cacheAndDiagnose(ctx, span.NewURI(params.TextDocument.URI), text)
}

func (s *Server) WillSave(context.Context, *protocol.WillSaveTextDocumentParams) error {
	return notImplemented("WillSave")
}

func (s *Server) WillSaveWaitUntil(context.Context, *protocol.WillSaveTextDocumentParams) ([]protocol.TextEdit, error) {
	return nil, notImplemented("WillSaveWaitUntil")
}

func (s *Server) DidSave(context.Context, *protocol.DidSaveTextDocumentParams) error {
	return nil // ignore
}

func (s *Server) DidClose(ctx context.Context, params *protocol.DidCloseTextDocumentParams) error {
	uri := span.NewURI(params.TextDocument.URI)
	view := s.findView(ctx, uri)
	return view.SetContent(ctx, uri, nil)
}

func (s *Server) Completion(ctx context.Context, params *protocol.CompletionParams) (*protocol.CompletionList, error) {
	return s.completion(ctx, params)
}

func (s *Server) CompletionResolve(context.Context, *protocol.CompletionItem) (*protocol.CompletionItem, error) {
	return nil, notImplemented("CompletionResolve")
}

func (s *Server) Hover(ctx context.Context, params *protocol.TextDocumentPositionParams) (*protocol.Hover, error) {
	uri := span.NewURI(params.TextDocument.URI)
	view := s.findView(ctx, uri)
	f, m, err := newColumnMap(ctx, view, uri)
	if err != nil {
		return nil, err
	}
	spn, err := m.PointSpan(params.Position)
	if err != nil {
		return nil, err
	}
	identRange, err := spn.Range(m.Converter)
	if err != nil {
		return nil, err
	}
	ident, err := source.Identifier(ctx, view, f, identRange.Start)
	if err != nil {
		return nil, err
	}
	content, err := ident.Hover(ctx, nil)
	if err != nil {
		return nil, err
	}
	markdown := "```go\n" + content + "\n```"
	identSpan, err := ident.Range.Span()
	if err != nil {
		return nil, err
	}
	rng, err := m.Range(identSpan)
	if err != nil {
		return nil, err
	}
	return &protocol.Hover{
		Contents: protocol.MarkupContent{
			Kind:  protocol.Markdown,
			Value: markdown,
		},
		Range: &rng,
	}, nil
}

func (s *Server) SignatureHelp(ctx context.Context, params *protocol.TextDocumentPositionParams) (*protocol.SignatureHelp, error) {
	uri := span.NewURI(params.TextDocument.URI)
	view := s.findView(ctx, uri)
	f, m, err := newColumnMap(ctx, view, uri)
	if err != nil {
		return nil, err
	}
	spn, err := m.PointSpan(params.Position)
	if err != nil {
		return nil, err
	}
	rng, err := spn.Range(m.Converter)
	if err != nil {
		return nil, err
	}
	info, err := source.SignatureHelp(ctx, f, rng.Start)
	if err != nil {
		s.log.Infof(ctx, "no signature help for %s:%v:%v", uri, int(params.Position.Line), int(params.Position.Character))
	}
	return toProtocolSignatureHelp(info), nil
}

func (s *Server) Definition(ctx context.Context, params *protocol.TextDocumentPositionParams) ([]protocol.Location, error) {
	uri := span.NewURI(params.TextDocument.URI)
	view := s.findView(ctx, uri)
	f, m, err := newColumnMap(ctx, view, uri)
	if err != nil {
		return nil, err
	}
	spn, err := m.PointSpan(params.Position)
	if err != nil {
		return nil, err
	}
	rng, err := spn.Range(m.Converter)
	if err != nil {
		return nil, err
	}
	ident, err := source.Identifier(ctx, view, f, rng.Start)
	if err != nil {
		return nil, err
	}
	decSpan, err := ident.Declaration.Range.Span()
	if err != nil {
		return nil, err
	}
	_, decM, err := newColumnMap(ctx, view, decSpan.URI())
	if err != nil {
		return nil, err
	}
	loc, err := decM.Location(decSpan)
	if err != nil {
		return nil, err
	}
	return []protocol.Location{loc}, nil
}

func (s *Server) TypeDefinition(ctx context.Context, params *protocol.TextDocumentPositionParams) ([]protocol.Location, error) {
	uri := span.NewURI(params.TextDocument.URI)
	view := s.findView(ctx, uri)
	f, m, err := newColumnMap(ctx, view, uri)
	if err != nil {
		return nil, err
	}
	spn, err := m.PointSpan(params.Position)
	if err != nil {
		return nil, err
	}
	rng, err := spn.Range(m.Converter)
	if err != nil {
		return nil, err
	}
	ident, err := source.Identifier(ctx, view, f, rng.Start)
	if err != nil {
		return nil, err
	}
	identSpan, err := ident.Type.Range.Span()
	if err != nil {
		return nil, err
	}
	_, identM, err := newColumnMap(ctx, view, identSpan.URI())
	if err != nil {
		return nil, err
	}
	loc, err := identM.Location(identSpan)
	if err != nil {
		return nil, err
	}
	return []protocol.Location{loc}, nil
}

func (s *Server) Implementation(context.Context, *protocol.TextDocumentPositionParams) ([]protocol.Location, error) {
	return nil, notImplemented("Implementation")
}

func (s *Server) References(context.Context, *protocol.ReferenceParams) ([]protocol.Location, error) {
	return nil, notImplemented("References")
}

func (s *Server) DocumentHighlight(ctx context.Context, params *protocol.TextDocumentPositionParams) ([]protocol.DocumentHighlight, error) {
	uri := span.NewURI(params.TextDocument.URI)
	view := s.findView(ctx, uri)
	f, m, err := newColumnMap(ctx, view, uri)
	if err != nil {
		return nil, err
	}
	spn, err := m.PointSpan(params.Position)
	if err != nil {
		return nil, err
	}
	rng, err := spn.Range(m.Converter)
	if err != nil {
		return nil, err
	}
	spans := source.Highlight(ctx, f, rng.Start)
	return toProtocolHighlight(m, spans), nil
}

func (s *Server) DocumentSymbol(ctx context.Context, params *protocol.DocumentSymbolParams) ([]protocol.DocumentSymbol, error) {
	uri := span.NewURI(params.TextDocument.URI)
	view := s.findView(ctx, uri)
	f, m, err := newColumnMap(ctx, view, uri)
	if err != nil {
		return nil, err
	}
	symbols := source.DocumentSymbols(ctx, f)
	return toProtocolDocumentSymbols(m, symbols), nil
}

func (s *Server) CodeAction(ctx context.Context, params *protocol.CodeActionParams) ([]protocol.CodeAction, error) {
	return s.codeAction(ctx, params)
}

func (s *Server) CodeLens(context.Context, *protocol.CodeLensParams) ([]protocol.CodeLens, error) {
	return nil, nil // ignore
}

func (s *Server) CodeLensResolve(context.Context, *protocol.CodeLens) (*protocol.CodeLens, error) {
	return nil, notImplemented("CodeLensResolve")
}

func (s *Server) DocumentLink(context.Context, *protocol.DocumentLinkParams) ([]protocol.DocumentLink, error) {
	return nil, nil // ignore
}

func (s *Server) DocumentLinkResolve(context.Context, *protocol.DocumentLink) (*protocol.DocumentLink, error) {
	return nil, notImplemented("DocumentLinkResolve")
}

func (s *Server) DocumentColor(context.Context, *protocol.DocumentColorParams) ([]protocol.ColorInformation, error) {
	return nil, notImplemented("DocumentColor")
}

func (s *Server) ColorPresentation(context.Context, *protocol.ColorPresentationParams) ([]protocol.ColorPresentation, error) {
	return nil, notImplemented("ColorPresentation")
}

func (s *Server) Formatting(ctx context.Context, params *protocol.DocumentFormattingParams) ([]protocol.TextEdit, error) {
	uri := span.NewURI(params.TextDocument.URI)
	view := s.findView(ctx, uri)
	spn := span.New(uri, span.Point{}, span.Point{})
	return formatRange(ctx, view, spn)
}

func (s *Server) RangeFormatting(ctx context.Context, params *protocol.DocumentRangeFormattingParams) ([]protocol.TextEdit, error) {
	uri := span.NewURI(params.TextDocument.URI)
	view := s.findView(ctx, uri)
	_, m, err := newColumnMap(ctx, view, uri)
	if err != nil {
		return nil, err
	}
	spn, err := m.RangeSpan(params.Range)
	if err != nil {
		return nil, err
	}
	return formatRange(ctx, view, spn)
}

func (s *Server) OnTypeFormatting(context.Context, *protocol.DocumentOnTypeFormattingParams) ([]protocol.TextEdit, error) {
	return nil, notImplemented("OnTypeFormatting")
}

func (s *Server) Rename(context.Context, *protocol.RenameParams) ([]protocol.WorkspaceEdit, error) {
	return nil, notImplemented("Rename")
}

func (s *Server) FoldingRanges(context.Context, *protocol.FoldingRangeParams) ([]protocol.FoldingRange, error) {
	return nil, notImplemented("FoldingRanges")
}

func (s *Server) processConfig(view *cache.View, config interface{}) error {
	// TODO: We should probably store and process more of the config.
	if config == nil {
		return nil // ignore error if you don't have a config
	}
	c, ok := config.(map[string]interface{})
	if !ok {
		return fmt.Errorf("invalid config gopls type %T", config)
	}
	// Get the environment for the go/packages config.
	if env := c["env"]; env != nil {
		menv, ok := env.(map[string]interface{})
		if !ok {
			return fmt.Errorf("invalid config gopls.env type %T", env)
		}
		for k, v := range menv {
			view.Config.Env = applyEnv(view.Config.Env, k, v)
		}
	}
	// Check if placeholders are enabled.
	if usePlaceholders, ok := c["usePlaceholders"].(bool); ok {
		s.usePlaceholders = usePlaceholders
	}
	return nil
}

func applyEnv(env []string, k string, v interface{}) []string {
	prefix := k + "="
	value := prefix + fmt.Sprint(v)
	for i, s := range env {
		if strings.HasPrefix(s, prefix) {
			env[i] = value
			return env
		}
	}
	return append(env, value)
}

func notImplemented(method string) *jsonrpc2.Error {
	return jsonrpc2.NewErrorf(jsonrpc2.CodeMethodNotFound, "method %q not yet implemented", method)
}

func (s *Server) findView(ctx context.Context, uri span.URI) *cache.View {
	// first see if a view already has this file
	for _, view := range s.views {
		if view.FindFile(ctx, uri) != nil {
			return view
		}
	}
	var longest *cache.View
	for _, view := range s.views {
		if longest != nil && len(longest.Folder) > len(view.Folder) {
			continue
		}
		if strings.HasPrefix(string(uri), string(view.Folder)) {
			longest = view
		}
	}
	if longest != nil {
		return longest
	}
	//TODO: are there any more heuristics we can use?
	return s.views[0]
}
