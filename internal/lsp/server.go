// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"fmt"
	"net"
	"sync"

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

func (s *Server) Run(ctx context.Context) error {
	return s.Conn.Run(ctx)
}

type Server struct {
	Conn   *jsonrpc2.Conn
	client protocol.Client
	log    xlog.Logger

	initializedMu sync.Mutex
	isInitialized bool // set once the server has received "initialize" request

	// Configurations.
	// TODO(rstambler): Separate these into their own struct?
	usePlaceholders               bool
	enhancedHover                 bool
	insertTextFormat              protocol.InsertTextFormat
	configurationSupported        bool
	dynamicConfigurationSupported bool
	preferredContentFormat        protocol.MarkupKind

	textDocumentSyncKind protocol.TextDocumentSyncKind

	views []*cache.View

	// undelivered is a cache of any diagnostics that the server
	// failed to deliver for some reason.
	undeliveredMu sync.Mutex
	undelivered   map[span.URI][]source.Diagnostic
}

// General

func (s *Server) Initialize(ctx context.Context, params *protocol.InitializeParams) (*protocol.InitializeResult, error) {
	return s.initialize(ctx, params)
}

func (s *Server) Initialized(ctx context.Context, params *protocol.InitializedParams) error {
	return s.initialized(ctx, params)
}

func (s *Server) Shutdown(ctx context.Context) error {
	return s.shutdown(ctx)
}

func (s *Server) Exit(ctx context.Context) error {
	return s.exit(ctx)
}

// Workspace

func (s *Server) DidChangeWorkspaceFolders(context.Context, *protocol.DidChangeWorkspaceFoldersParams) error {
	return notImplemented("DidChangeWorkspaceFolders")
}

func (s *Server) DidChangeConfiguration(context.Context, *protocol.DidChangeConfigurationParams) error {
	return notImplemented("DidChangeConfiguration")
}

func (s *Server) DidChangeWatchedFiles(context.Context, *protocol.DidChangeWatchedFilesParams) error {
	return notImplemented("DidChangeWatchedFiles")
}

func (s *Server) Symbol(context.Context, *protocol.WorkspaceSymbolParams) ([]protocol.SymbolInformation, error) {
	return nil, notImplemented("Symbol")
}

func (s *Server) ExecuteCommand(context.Context, *protocol.ExecuteCommandParams) (interface{}, error) {
	return nil, notImplemented("ExecuteCommand")
}

// Text Synchronization

func (s *Server) DidOpen(ctx context.Context, params *protocol.DidOpenTextDocumentParams) error {
	return s.cacheAndDiagnose(ctx, span.NewURI(params.TextDocument.URI), params.TextDocument.Text)
}

func (s *Server) DidChange(ctx context.Context, params *protocol.DidChangeTextDocumentParams) error {
	return s.didChange(ctx, params)
}

func (s *Server) WillSave(context.Context, *protocol.WillSaveTextDocumentParams) error {
	return notImplemented("WillSave")
}

func (s *Server) WillSaveWaitUntil(context.Context, *protocol.WillSaveTextDocumentParams) ([]protocol.TextEdit, error) {
	return nil, notImplemented("WillSaveWaitUntil")
}

func (s *Server) DidSave(ctx context.Context, params *protocol.DidSaveTextDocumentParams) error {
	return s.didSave(ctx, params)
}

func (s *Server) DidClose(ctx context.Context, params *protocol.DidCloseTextDocumentParams) error {
	return s.didClose(ctx, params)
}

// Language Features

func (s *Server) Completion(ctx context.Context, params *protocol.CompletionParams) (*protocol.CompletionList, error) {
	return s.completion(ctx, params)
}

func (s *Server) CompletionResolve(context.Context, *protocol.CompletionItem) (*protocol.CompletionItem, error) {
	return nil, notImplemented("CompletionResolve")
}

func (s *Server) Hover(ctx context.Context, params *protocol.TextDocumentPositionParams) (*protocol.Hover, error) {
	return s.hover(ctx, params)
}

func (s *Server) SignatureHelp(ctx context.Context, params *protocol.TextDocumentPositionParams) (*protocol.SignatureHelp, error) {
	return s.signatureHelp(ctx, params)
}

func (s *Server) Definition(ctx context.Context, params *protocol.TextDocumentPositionParams) ([]protocol.Location, error) {
	return s.definition(ctx, params)
}

func (s *Server) TypeDefinition(ctx context.Context, params *protocol.TextDocumentPositionParams) ([]protocol.Location, error) {
	return s.typeDefinition(ctx, params)
}

func (s *Server) Implementation(context.Context, *protocol.TextDocumentPositionParams) ([]protocol.Location, error) {
	return nil, notImplemented("Implementation")
}

func (s *Server) References(context.Context, *protocol.ReferenceParams) ([]protocol.Location, error) {
	return nil, notImplemented("References")
}

func (s *Server) DocumentHighlight(ctx context.Context, params *protocol.TextDocumentPositionParams) ([]protocol.DocumentHighlight, error) {
	return s.documentHighlight(ctx, params)
}

func (s *Server) DocumentSymbol(ctx context.Context, params *protocol.DocumentSymbolParams) ([]protocol.DocumentSymbol, error) {
	return s.documentSymbol(ctx, params)
}

func (s *Server) CodeAction(ctx context.Context, params *protocol.CodeActionParams) ([]protocol.CodeAction, error) {
	return s.codeAction(ctx, params)
}

func (s *Server) CodeLens(context.Context, *protocol.CodeLensParams) ([]protocol.CodeLens, error) {
	return nil, nil // ignore
}

func (s *Server) ResolveCodeLens(context.Context, *protocol.CodeLens) (*protocol.CodeLens, error) {
	return nil, notImplemented("ResolveCodeLens")
}

func (s *Server) DocumentLink(ctx context.Context, params *protocol.DocumentLinkParams) ([]protocol.DocumentLink, error) {
	return s.documentLink(ctx, params)
}

func (s *Server) ResolveDocumentLink(context.Context, *protocol.DocumentLink) (*protocol.DocumentLink, error) {
	return nil, notImplemented("ResolveDocumentLink")
}

func (s *Server) DocumentColor(context.Context, *protocol.DocumentColorParams) ([]protocol.ColorInformation, error) {
	return nil, notImplemented("DocumentColor")
}

func (s *Server) ColorPresentation(context.Context, *protocol.ColorPresentationParams) ([]protocol.ColorPresentation, error) {
	return nil, notImplemented("ColorPresentation")
}

func (s *Server) Formatting(ctx context.Context, params *protocol.DocumentFormattingParams) ([]protocol.TextEdit, error) {
	return s.formatting(ctx, params)
}

func (s *Server) RangeFormatting(ctx context.Context, params *protocol.DocumentRangeFormattingParams) ([]protocol.TextEdit, error) {
	return nil, notImplemented("RangeFormatting")
}

func (s *Server) OnTypeFormatting(context.Context, *protocol.DocumentOnTypeFormattingParams) ([]protocol.TextEdit, error) {
	return nil, notImplemented("OnTypeFormatting")
}

func (s *Server) Rename(context.Context, *protocol.RenameParams) (*protocol.WorkspaceEdit, error) {
	return nil, notImplemented("Rename")
}

func (s *Server) Declaration(context.Context, *protocol.TextDocumentPositionParams) ([]protocol.DeclarationLink, error) {
	return nil, notImplemented("Declaration")
}

func (s *Server) FoldingRange(context.Context, *protocol.FoldingRangeParams) ([]protocol.FoldingRange, error) {
	return nil, notImplemented("FoldingRange")
}

func (s *Server) LogTraceNotification(context.Context, *protocol.LogTraceParams) error {
	return notImplemented("LogtraceNotification")
}

func (s *Server) PrepareRename(context.Context, *protocol.TextDocumentPositionParams) (*protocol.Range, error) {
	return nil, notImplemented("PrepareRename")
}

func (s *Server) Resolve(context.Context, *protocol.CompletionItem) (*protocol.CompletionItem, error) {
	return nil, notImplemented("Resolve")
}

func (s *Server) SelectionRange(context.Context, *protocol.SelectionRangeParams) ([][]protocol.SelectionRange, error) {
	return nil, notImplemented("SelectionRange")
}

func (s *Server) SetTraceNotification(context.Context, *protocol.SetTraceParams) error {
	return notImplemented("SetTraceNotification")
}
func notImplemented(method string) *jsonrpc2.Error {
	return jsonrpc2.NewErrorf(jsonrpc2.CodeMethodNotFound, "method %q not yet implemented", method)
}
