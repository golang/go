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
	"golang.org/x/tools/internal/span"
)

// NewClientServer
func NewClientServer(client protocol.Client) *Server {
	return &Server{
		client: client,
	}
}

// NewServer starts an LSP server on the supplied stream, and waits until the
// stream is closed.
func NewServer(stream jsonrpc2.Stream) *Server {
	s := &Server{}
	s.Conn, s.client = protocol.NewServer(stream, s)
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

	initializedMu sync.Mutex
	initialized   bool // set once the server has received "initialize" request

	signatureHelpEnabled bool
	snippetsSupported    bool

	textDocumentSyncKind protocol.TextDocumentSyncKind

	view *cache.View
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

	// Check if the client supports snippets in completion items.
	if x, ok := params.Capabilities["textDocument"].(map[string]interface{}); ok {
		if x, ok := x["completion"].(map[string]interface{}); ok {
			if x, ok := x["completionItem"].(map[string]interface{}); ok {
				if x, ok := x["snippetSupport"].(bool); ok {
					s.snippetsSupported = x
				}
			}
		}
	}
	s.signatureHelpEnabled = true

	var rootURI span.URI
	if params.RootURI != "" {
		rootURI = span.NewURI(params.RootURI)
	}
	rootPath, err := rootURI.Filename()
	if err != nil {
		return nil, err
	}

	// TODO(rstambler): Change this default to protocol.Incremental (or add a
	// flag). Disabled for now to simplify debugging.
	s.textDocumentSyncKind = protocol.Full

	//TODO:use workspace folders
	s.view = cache.NewView(path.Base(string(rootURI)), rootURI, &packages.Config{
		Context: ctx,
		Dir:     rootPath,
		Env:     os.Environ(),
		Mode:    packages.LoadImports,
		Fset:    token.NewFileSet(),
		Overlay: make(map[string][]byte),
		ParseFile: func(fset *token.FileSet, filename string, src []byte) (*ast.File, error) {
			return parser.ParseFile(fset, filename, src, parser.AllErrors|parser.ParseComments)
		},
		Tests: true,
	})

	return &protocol.InitializeResult{
		Capabilities: protocol.ServerCapabilities{
			InnerServerCapabilities: protocol.InnerServerCapabilities{
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
			},
			TypeDefinitionServerCapabilities: protocol.TypeDefinitionServerCapabilities{
				TypeDefinitionProvider: true,
			},
		},
	}, nil
}

func (s *Server) Initialized(ctx context.Context, params *protocol.InitializedParams) error {
	s.client.RegisterCapability(ctx, &protocol.RegistrationParams{
		Registrations: []protocol.Registration{{
			ID:     "workspace/didChangeConfiguration",
			Method: "workspace/didChangeConfiguration",
		}},
	})
	config, err := s.client.Configuration(ctx, &protocol.ConfigurationParams{
		Items: []protocol.ConfigurationItem{{
			ScopeURI: protocol.NewURI(s.view.Folder),
			Section:  "gopls",
		}},
	})
	if err != nil {
		return err
	}
	if err := s.processConfig(config[0]); err != nil {
		return err
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

	file, m, err := newColumnMap(ctx, s.view, span.NewURI(params.TextDocument.URI))
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
	s.setContent(ctx, span.NewURI(params.TextDocument.URI), nil)
	return nil
}

func (s *Server) Completion(ctx context.Context, params *protocol.CompletionParams) (*protocol.CompletionList, error) {
	f, m, err := newColumnMap(ctx, s.view, span.NewURI(params.TextDocument.URI))
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
	items, prefix, err := source.Completion(ctx, f, rng.Start)
	if err != nil {
		return nil, err
	}
	return &protocol.CompletionList{
		IsIncomplete: false,
		Items:        toProtocolCompletionItems(items, prefix, params.Position, s.snippetsSupported, s.signatureHelpEnabled),
	}, nil
}

func (s *Server) CompletionResolve(context.Context, *protocol.CompletionItem) (*protocol.CompletionItem, error) {
	return nil, notImplemented("CompletionResolve")
}

func (s *Server) Hover(ctx context.Context, params *protocol.TextDocumentPositionParams) (*protocol.Hover, error) {
	f, m, err := newColumnMap(ctx, s.view, span.NewURI(params.TextDocument.URI))
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
	ident, err := source.Identifier(ctx, s.view, f, identRange.Start)
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
	f, m, err := newColumnMap(ctx, s.view, span.NewURI(params.TextDocument.URI))
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
		return nil, err
	}
	return toProtocolSignatureHelp(info), nil
}

func (s *Server) Definition(ctx context.Context, params *protocol.TextDocumentPositionParams) ([]protocol.Location, error) {
	f, m, err := newColumnMap(ctx, s.view, span.NewURI(params.TextDocument.URI))
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
	ident, err := source.Identifier(ctx, s.view, f, rng.Start)
	if err != nil {
		return nil, err
	}
	decSpan, err := ident.Declaration.Range.Span()
	if err != nil {
		return nil, err
	}
	_, decM, err := newColumnMap(ctx, s.view, decSpan.URI())
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
	f, m, err := newColumnMap(ctx, s.view, span.NewURI(params.TextDocument.URI))
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
	ident, err := source.Identifier(ctx, s.view, f, rng.Start)
	if err != nil {
		return nil, err
	}
	identSpan, err := ident.Type.Range.Span()
	if err != nil {
		return nil, err
	}
	_, identM, err := newColumnMap(ctx, s.view, identSpan.URI())
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
	f, m, err := newColumnMap(ctx, s.view, span.NewURI(params.TextDocument.URI))
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
	f, m, err := newColumnMap(ctx, s.view, span.NewURI(params.TextDocument.URI))
	if err != nil {
		return nil, err
	}
	symbols := source.DocumentSymbols(ctx, f)
	return toProtocolDocumentSymbols(m, symbols), nil
}

func (s *Server) CodeAction(ctx context.Context, params *protocol.CodeActionParams) ([]protocol.CodeAction, error) {
	_, m, err := newColumnMap(ctx, s.view, span.NewURI(params.TextDocument.URI))
	if err != nil {
		return nil, err
	}
	spn, err := m.RangeSpan(params.Range)
	if err != nil {
		return nil, err
	}
	edits, err := organizeImports(ctx, s.view, spn)
	if err != nil {
		return nil, err
	}
	return []protocol.CodeAction{
		{
			Title: "Organize Imports",
			Kind:  protocol.SourceOrganizeImports,
			Edit: &protocol.WorkspaceEdit{
				Changes: &map[string][]protocol.TextEdit{
					params.TextDocument.URI: edits,
				},
			},
		},
	}, nil
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
	spn := span.New(span.URI(params.TextDocument.URI), span.Point{}, span.Point{})
	return formatRange(ctx, s.view, spn)
}

func (s *Server) RangeFormatting(ctx context.Context, params *protocol.DocumentRangeFormattingParams) ([]protocol.TextEdit, error) {
	_, m, err := newColumnMap(ctx, s.view, span.NewURI(params.TextDocument.URI))
	if err != nil {
		return nil, err
	}
	spn, err := m.RangeSpan(params.Range)
	if err != nil {
		return nil, err
	}
	return formatRange(ctx, s.view, spn)
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

func (s *Server) Error(err error) {
	s.client.LogMessage(context.Background(), &protocol.LogMessageParams{
		Type:    protocol.Error,
		Message: fmt.Sprint(err),
	})
}

func (s *Server) processConfig(config interface{}) error {
	//TODO: we should probably store and process more of the config
	c, ok := config.(map[string]interface{})
	if !ok {
		return fmt.Errorf("Invalid config gopls type %T", config)
	}
	env := c["env"]
	if env == nil {
		return nil
	}
	menv, ok := env.(map[string]interface{})
	if !ok {
		return fmt.Errorf("Invalid config gopls.env type %T", env)
	}
	for k, v := range menv {
		s.view.Config.Env = applyEnv(s.view.Config.Env, k, v)
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
