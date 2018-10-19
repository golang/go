// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"os"
	"sync"

	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/lsp/protocol"
)

// RunServer starts an LSP server on the supplied stream, and waits until the
// stream is closed.
func RunServer(ctx context.Context, stream jsonrpc2.Stream, opts ...interface{}) error {
	s := &server{
		view: newView(),
	}
	conn, client := protocol.RunServer(ctx, stream, s, opts...)
	s.client = client
	return conn.Wait(ctx)
}

type server struct {
	client protocol.Client

	initializedMu sync.Mutex
	initialized   bool // set once the server has received "initialize" request

	*view
}

func (s *server) Initialize(ctx context.Context, params *protocol.InitializeParams) (*protocol.InitializeResult, error) {
	s.initializedMu.Lock()
	defer s.initializedMu.Unlock()
	if s.initialized {
		return nil, jsonrpc2.NewErrorf(jsonrpc2.CodeInvalidRequest, "server already initialized")
	}
	s.initialized = true
	return &protocol.InitializeResult{
		Capabilities: protocol.ServerCapabilities{
			TextDocumentSync: protocol.TextDocumentSyncOptions{
				Change:    float64(protocol.Full), // full contents of file sent on each update
				OpenClose: true,
			},
			DocumentFormattingProvider:      true,
			DocumentRangeFormattingProvider: true,
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
	s.cacheAndDiagnoseFile(ctx, params.TextDocument.URI, params.TextDocument.Text)
	return nil
}

func (s *server) DidChange(ctx context.Context, params *protocol.DidChangeTextDocumentParams) error {
	if len(params.ContentChanges) < 1 {
		return jsonrpc2.NewErrorf(jsonrpc2.CodeInternalError, "no content changes provided")
	}
	// We expect the full content of file, i.e. a single change with no range.
	if change := params.ContentChanges[0]; change.RangeLength == 0 {
		s.cacheAndDiagnoseFile(ctx, params.TextDocument.URI, change.Text)
	}
	return nil
}

func (s *server) cacheAndDiagnoseFile(ctx context.Context, uri protocol.DocumentURI, text string) {
	s.view.activeFilesMu.Lock()
	s.view.activeFiles[uri] = []byte(text)
	s.view.activeFilesMu.Unlock()
	go func() {
		reports, err := s.diagnostics(uri)
		if err == nil {
			for filename, diagnostics := range reports {
				s.client.PublishDiagnostics(ctx, &protocol.PublishDiagnosticsParams{
					URI:         filenameToURI(filename),
					Diagnostics: diagnostics,
				})
			}
		}
	}()
}

func (s *server) WillSave(context.Context, *protocol.WillSaveTextDocumentParams) error {
	return notImplemented("WillSave")
}

func (s *server) WillSaveWaitUntil(context.Context, *protocol.WillSaveTextDocumentParams) ([]protocol.TextEdit, error) {
	return nil, notImplemented("WillSaveWaitUntil")
}

func (s *server) DidSave(context.Context, *protocol.DidSaveTextDocumentParams) error {
	// TODO(rstambler): Should we clear the cache here?
	return nil // ignore
}

func (s *server) DidClose(ctx context.Context, params *protocol.DidCloseTextDocumentParams) error {
	s.clearActiveFile(params.TextDocument.URI)
	return nil
}

func (s *server) Completion(context.Context, *protocol.CompletionParams) (*protocol.CompletionList, error) {
	return nil, notImplemented("Completion")
}

func (s *server) CompletionResolve(context.Context, *protocol.CompletionItem) (*protocol.CompletionItem, error) {
	return nil, notImplemented("CompletionResolve")
}

func (s *server) Hover(context.Context, *protocol.TextDocumentPositionParams) (*protocol.Hover, error) {
	return nil, notImplemented("Hover")
}

func (s *server) SignatureHelp(context.Context, *protocol.TextDocumentPositionParams) (*protocol.SignatureHelp, error) {
	return nil, notImplemented("SignatureHelp")
}

func (s *server) Definition(context.Context, *protocol.TextDocumentPositionParams) ([]protocol.Location, error) {
	return nil, notImplemented("Definition")
}

func (s *server) TypeDefinition(context.Context, *protocol.TextDocumentPositionParams) ([]protocol.Location, error) {
	return nil, notImplemented("TypeDefinition")
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

func (s *server) CodeAction(context.Context, *protocol.CodeActionParams) ([]protocol.CodeAction, error) {
	return nil, notImplemented("CodeAction")
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
	return s.format(params.TextDocument.URI, nil)
}

func (s *server) RangeFormatting(ctx context.Context, params *protocol.DocumentRangeFormattingParams) ([]protocol.TextEdit, error) {
	return s.format(params.TextDocument.URI, &params.Range)
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
