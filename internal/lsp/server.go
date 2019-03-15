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
	"sync"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/lsp/cache"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
)

// RunServer starts an LSP server on the supplied stream, and waits until the
// stream is closed.
func RunServer(ctx context.Context, stream jsonrpc2.Stream, opts ...interface{}) error {
	s := &server{}
	conn, client := protocol.RunServer(ctx, stream, s, opts...)
	s.client = client
	return conn.Wait(ctx)
}

// RunServerOnPort starts an LSP server on the given port and does not exit.
// This function exists for debugging purposes.
func RunServerOnPort(ctx context.Context, port int, opts ...interface{}) error {
	return RunServerOnAddress(ctx, fmt.Sprintf(":%v", port))
}

// RunServerOnPort starts an LSP server on the given port and does not exit.
// This function exists for debugging purposes.
func RunServerOnAddress(ctx context.Context, addr string, opts ...interface{}) error {
	s := &server{}
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
		go func() {
			conn, client := protocol.RunServer(ctx, stream, s, opts...)
			s.client = client
			conn.Wait(ctx)
		}()
	}
}

type server struct {
	client protocol.Client

	initializedMu sync.Mutex
	initialized   bool // set once the server has received "initialize" request

	signatureHelpEnabled bool
	snippetsSupported    bool

	textDocumentSyncKind protocol.TextDocumentSyncKind

	viewMu sync.Mutex
	view   *cache.View
}

func (s *server) Initialize(ctx context.Context, params *protocol.InitializeParams) (*protocol.InitializeResult, error) {
	s.initializedMu.Lock()
	defer s.initializedMu.Unlock()
	if s.initialized {
		return nil, jsonrpc2.NewErrorf(jsonrpc2.CodeInvalidRequest, "server already initialized")
	}
	s.initialized = true // mark server as initialized now

	// Check if the client supports snippets in completion items.
	capText := params.Capabilities.InnerClientCapabilities.TextDocument
	if capText != nil && capText.Completion != nil && capText.Completion.CompletionItem != nil {
		s.snippetsSupported = capText.Completion.CompletionItem.SnippetSupport
	}
	s.signatureHelpEnabled = true

	var rootURI span.URI
	if params.RootURI != "" {
		rootURI = span.URI(params.RootURI)
	}
	rootPath, err := rootURI.Filename()
	if err != nil {
		return nil, err
	}

	// TODO(rstambler): Change this default to protocol.Incremental (or add a
	// flag). Disabled for now to simplify debugging.
	s.textDocumentSyncKind = protocol.Full

	s.view = cache.NewView(&packages.Config{
		Context: ctx,
		Dir:     rootPath,
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
				HoverProvider:                   true,
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

func (s *server) Initialized(context.Context, *protocol.InitializedParams) error {
	return nil // ignore
}

func (s *server) Shutdown(context.Context) error {
	s.initializedMu.Lock()
	defer s.initializedMu.Unlock()
	if !s.initialized {
		return jsonrpc2.NewErrorf(jsonrpc2.CodeInvalidRequest, "server not initialized")
	}
	s.initialized = false
	return nil
}

func (s *server) Exit(ctx context.Context) error {
	if s.initialized {
		os.Exit(1)
	}
	os.Exit(0)
	return nil
}

func (s *server) DidChangeWorkspaceFolders(context.Context, *protocol.DidChangeWorkspaceFoldersParams) error {
	return notImplemented("DidChangeWorkspaceFolders")
}

func (s *server) DidChangeConfiguration(context.Context, *protocol.DidChangeConfigurationParams) error {
	return notImplemented("DidChangeConfiguration")
}

func (s *server) DidChangeWatchedFiles(context.Context, *protocol.DidChangeWatchedFilesParams) error {
	return notImplemented("DidChangeWatchedFiles")
}

func (s *server) Symbols(context.Context, *protocol.WorkspaceSymbolParams) ([]protocol.SymbolInformation, error) {
	return nil, notImplemented("Symbols")
}

func (s *server) ExecuteCommand(context.Context, *protocol.ExecuteCommandParams) (interface{}, error) {
	return nil, notImplemented("ExecuteCommand")
}

func (s *server) DidOpen(ctx context.Context, params *protocol.DidOpenTextDocumentParams) error {
	return s.cacheAndDiagnose(ctx, span.URI(params.TextDocument.URI), params.TextDocument.Text)
}

func (s *server) applyChanges(ctx context.Context, params *protocol.DidChangeTextDocumentParams) (string, error) {
	if len(params.ContentChanges) == 1 && params.ContentChanges[0].Range == nil {
		// If range is empty, we expect the full content of file, i.e. a single change with no range.
		change := params.ContentChanges[0]
		if change.RangeLength != 0 {
			return "", jsonrpc2.NewErrorf(jsonrpc2.CodeInternalError, "unexpected change range provided")
		}
		return change.Text, nil
	}

	file, m, err := newColumnMap(ctx, s.view, span.URI(params.TextDocument.URI))
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

func (s *server) DidChange(ctx context.Context, params *protocol.DidChangeTextDocumentParams) error {
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
	return s.cacheAndDiagnose(ctx, span.URI(params.TextDocument.URI), text)
}

func (s *server) WillSave(context.Context, *protocol.WillSaveTextDocumentParams) error {
	return notImplemented("WillSave")
}

func (s *server) WillSaveWaitUntil(context.Context, *protocol.WillSaveTextDocumentParams) ([]protocol.TextEdit, error) {
	return nil, notImplemented("WillSaveWaitUntil")
}

func (s *server) DidSave(context.Context, *protocol.DidSaveTextDocumentParams) error {
	return nil // ignore
}

func (s *server) DidClose(ctx context.Context, params *protocol.DidCloseTextDocumentParams) error {
	s.setContent(ctx, span.URI(params.TextDocument.URI), nil)
	return nil
}

func (s *server) Completion(ctx context.Context, params *protocol.CompletionParams) (*protocol.CompletionList, error) {
	f, m, err := newColumnMap(ctx, s.view, span.URI(params.TextDocument.URI))
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

func (s *server) CompletionResolve(context.Context, *protocol.CompletionItem) (*protocol.CompletionItem, error) {
	return nil, notImplemented("CompletionResolve")
}

func (s *server) Hover(ctx context.Context, params *protocol.TextDocumentPositionParams) (*protocol.Hover, error) {
	f, m, err := newColumnMap(ctx, s.view, span.URI(params.TextDocument.URI))
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

func (s *server) SignatureHelp(ctx context.Context, params *protocol.TextDocumentPositionParams) (*protocol.SignatureHelp, error) {
	f, m, err := newColumnMap(ctx, s.view, span.URI(params.TextDocument.URI))
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

func (s *server) Definition(ctx context.Context, params *protocol.TextDocumentPositionParams) ([]protocol.Location, error) {
	f, m, err := newColumnMap(ctx, s.view, span.URI(params.TextDocument.URI))
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

func (s *server) TypeDefinition(ctx context.Context, params *protocol.TextDocumentPositionParams) ([]protocol.Location, error) {
	f, m, err := newColumnMap(ctx, s.view, span.URI(params.TextDocument.URI))
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

func (s *server) Implementation(context.Context, *protocol.TextDocumentPositionParams) ([]protocol.Location, error) {
	return nil, notImplemented("Implementation")
}

func (s *server) References(context.Context, *protocol.ReferenceParams) ([]protocol.Location, error) {
	return nil, notImplemented("References")
}

func (s *server) DocumentHighlight(context.Context, *protocol.TextDocumentPositionParams) ([]protocol.DocumentHighlight, error) {
	return nil, notImplemented("DocumentHighlight")
}

func (s *server) DocumentSymbol(context.Context, *protocol.DocumentSymbolParams) ([]protocol.DocumentSymbol, error) {
	return nil, notImplemented("DocumentSymbol")
}

func (s *server) CodeAction(ctx context.Context, params *protocol.CodeActionParams) ([]protocol.CodeAction, error) {
	_, m, err := newColumnMap(ctx, s.view, span.URI(params.TextDocument.URI))
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

func (s *server) CodeLens(context.Context, *protocol.CodeLensParams) ([]protocol.CodeLens, error) {
	return nil, nil // ignore
}

func (s *server) CodeLensResolve(context.Context, *protocol.CodeLens) (*protocol.CodeLens, error) {
	return nil, notImplemented("CodeLensResolve")
}

func (s *server) DocumentLink(context.Context, *protocol.DocumentLinkParams) ([]protocol.DocumentLink, error) {
	return nil, nil // ignore
}

func (s *server) DocumentLinkResolve(context.Context, *protocol.DocumentLink) (*protocol.DocumentLink, error) {
	return nil, notImplemented("DocumentLinkResolve")
}

func (s *server) DocumentColor(context.Context, *protocol.DocumentColorParams) ([]protocol.ColorInformation, error) {
	return nil, notImplemented("DocumentColor")
}

func (s *server) ColorPresentation(context.Context, *protocol.ColorPresentationParams) ([]protocol.ColorPresentation, error) {
	return nil, notImplemented("ColorPresentation")
}

func (s *server) Formatting(ctx context.Context, params *protocol.DocumentFormattingParams) ([]protocol.TextEdit, error) {
	spn := span.New(span.URI(params.TextDocument.URI), span.Point{}, span.Point{})
	return formatRange(ctx, s.view, spn)
}

func (s *server) RangeFormatting(ctx context.Context, params *protocol.DocumentRangeFormattingParams) ([]protocol.TextEdit, error) {
	_, m, err := newColumnMap(ctx, s.view, span.URI(params.TextDocument.URI))
	if err != nil {
		return nil, err
	}
	spn, err := m.RangeSpan(params.Range)
	if err != nil {
		return nil, err
	}
	return formatRange(ctx, s.view, spn)
}

func (s *server) OnTypeFormatting(context.Context, *protocol.DocumentOnTypeFormattingParams) ([]protocol.TextEdit, error) {
	return nil, notImplemented("OnTypeFormatting")
}

func (s *server) Rename(context.Context, *protocol.RenameParams) ([]protocol.WorkspaceEdit, error) {
	return nil, notImplemented("Rename")
}

func (s *server) FoldingRanges(context.Context, *protocol.FoldingRangeParams) ([]protocol.FoldingRange, error) {
	return nil, notImplemented("FoldingRanges")
}

func notImplemented(method string) *jsonrpc2.Error {
	return jsonrpc2.NewErrorf(jsonrpc2.CodeMethodNotFound, "method %q not yet implemented", method)
}
