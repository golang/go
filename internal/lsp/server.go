// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"fmt"
	"go/token"
	"net"
	"os"
	"sync"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/lsp/cache"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
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
	s := &server{}
	ln, err := net.Listen("tcp", fmt.Sprintf(":%v", port))
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

	viewMu sync.Mutex
	view   source.View
}

func (s *server) Initialize(ctx context.Context, params *protocol.InitializeParams) (*protocol.InitializeResult, error) {
	s.initializedMu.Lock()
	defer s.initializedMu.Unlock()
	if s.initialized {
		return nil, jsonrpc2.NewErrorf(jsonrpc2.CodeInvalidRequest, "server already initialized")
	}
	s.initialized = true // mark server as initialized now

	// Check if the client supports snippets in completion items.
	s.snippetsSupported = params.Capabilities.TextDocument.Completion.CompletionItem.SnippetSupport
	s.signatureHelpEnabled = true

	rootPath, err := fromProtocolURI(*params.RootURI).Filename()
	if err != nil {
		return nil, err
	}
	s.view = cache.NewView(&packages.Config{
		Dir:     rootPath,
		Mode:    packages.LoadSyntax,
		Fset:    token.NewFileSet(),
		Tests:   true,
		Overlay: make(map[string][]byte),
	})

	return &protocol.InitializeResult{
		Capabilities: protocol.ServerCapabilities{
			CodeActionProvider: true,
			CompletionProvider: protocol.CompletionOptions{
				TriggerCharacters: []string{"."},
			},
			DefinitionProvider:              true,
			DocumentFormattingProvider:      true,
			DocumentRangeFormattingProvider: true,
			HoverProvider:                   true,
			SignatureHelpProvider: protocol.SignatureHelpOptions{
				TriggerCharacters: []string{"(", ","},
			},
			TextDocumentSync: protocol.TextDocumentSyncOptions{
				Change:    float64(protocol.Full), // full contents of file sent on each update
				OpenClose: true,
			},
			TypeDefinitionProvider: true,
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
	s.cacheAndDiagnose(ctx, params.TextDocument.URI, params.TextDocument.Text)
	return nil
}

func (s *server) DidChange(ctx context.Context, params *protocol.DidChangeTextDocumentParams) error {
	if len(params.ContentChanges) < 1 {
		return jsonrpc2.NewErrorf(jsonrpc2.CodeInternalError, "no content changes provided")
	}
	// We expect the full content of file, i.e. a single change with no range.
	if change := params.ContentChanges[0]; change.RangeLength == 0 {
		s.cacheAndDiagnose(ctx, params.TextDocument.URI, change.Text)
	}
	return nil
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
	s.setContent(ctx, fromProtocolURI(params.TextDocument.URI), nil)
	return nil
}

func (s *server) Completion(ctx context.Context, params *protocol.CompletionParams) (*protocol.CompletionList, error) {
	f, err := s.view.GetFile(ctx, fromProtocolURI(params.TextDocument.URI))
	if err != nil {
		return nil, err
	}
	tok, err := f.GetToken()
	if err != nil {
		return nil, err
	}
	pos := fromProtocolPosition(tok, params.Position)
	items, prefix, err := source.Completion(ctx, f, pos)
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
	f, err := s.view.GetFile(ctx, fromProtocolURI(params.TextDocument.URI))
	if err != nil {
		return nil, err
	}
	tok, err := f.GetToken()
	if err != nil {
		return nil, err
	}
	pos := fromProtocolPosition(tok, params.Position)
	contents, rng, err := source.Hover(ctx, f, pos)
	if err != nil {
		return nil, err
	}
	return &protocol.Hover{
		Contents: protocol.MarkupContent{
			Kind:  protocol.Markdown,
			Value: contents,
		},
		Range: toProtocolRange(tok, rng),
	}, nil
}

func (s *server) SignatureHelp(ctx context.Context, params *protocol.TextDocumentPositionParams) (*protocol.SignatureHelp, error) {
	f, err := s.view.GetFile(ctx, fromProtocolURI(params.TextDocument.URI))
	if err != nil {
		return nil, err
	}
	tok, err := f.GetToken()
	if err != nil {
		return nil, err
	}
	pos := fromProtocolPosition(tok, params.Position)
	info, err := source.SignatureHelp(ctx, f, pos)
	if err != nil {
		return nil, err
	}
	return toProtocolSignatureHelp(info), nil
}

func (s *server) Definition(ctx context.Context, params *protocol.TextDocumentPositionParams) ([]protocol.Location, error) {
	f, err := s.view.GetFile(ctx, fromProtocolURI(params.TextDocument.URI))
	if err != nil {
		return nil, err
	}
	tok, err := f.GetToken()
	if err != nil {
		return nil, err
	}
	pos := fromProtocolPosition(tok, params.Position)
	r, err := source.Definition(ctx, s.view, f, pos)
	if err != nil {
		return nil, err
	}
	return []protocol.Location{toProtocolLocation(s.view.FileSet(), r)}, nil
}

func (s *server) TypeDefinition(ctx context.Context, params *protocol.TextDocumentPositionParams) ([]protocol.Location, error) {
	f, err := s.view.GetFile(ctx, fromProtocolURI(params.TextDocument.URI))
	if err != nil {
		return nil, err
	}
	tok, err := f.GetToken()
	if err != nil {
		return nil, err
	}
	pos := fromProtocolPosition(tok, params.Position)
	r, err := source.TypeDefinition(ctx, s.view, f, pos)
	if err != nil {
		return nil, err
	}
	return []protocol.Location{toProtocolLocation(s.view.FileSet(), r)}, nil
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
	edits, err := organizeImports(ctx, s.view, params.TextDocument.URI)
	if err != nil {
		return nil, err
	}
	return []protocol.CodeAction{
		{
			Title: "Organize Imports",
			Kind:  protocol.SourceOrganizeImports,
			Edit: protocol.WorkspaceEdit{
				Changes: map[protocol.DocumentURI][]protocol.TextEdit{
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
	return formatRange(ctx, s.view, params.TextDocument.URI, nil)
}

func (s *server) RangeFormatting(ctx context.Context, params *protocol.DocumentRangeFormattingParams) ([]protocol.TextEdit, error) {
	return formatRange(ctx, s.view, params.TextDocument.URI, &params.Range)
}

func (s *server) OnTypeFormatting(context.Context, *protocol.DocumentOnTypeFormattingParams) ([]protocol.TextEdit, error) {
	return nil, notImplemented("OnTypeFormatting")
}

func (s *server) Rename(context.Context, *protocol.RenameParams) ([]protocol.WorkspaceEdit, error) {
	return nil, notImplemented("Rename")
}

func (s *server) FoldingRanges(context.Context, *protocol.FoldingRangeRequestParam) ([]protocol.FoldingRange, error) {
	return nil, notImplemented("FoldingRanges")
}

func notImplemented(method string) *jsonrpc2.Error {
	return jsonrpc2.NewErrorf(jsonrpc2.CodeMethodNotFound, "method %q not yet implemented", method)
}
