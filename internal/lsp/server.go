// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"

	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/lsp/protocol"
)

// RunServer starts an LSP server on the supplied stream, and waits until the
// stream is closed.
func RunServer(ctx context.Context, stream jsonrpc2.Stream, opts ...interface{}) error {
	s := &server{}
	conn, client := protocol.RunServer(ctx, stream, s, opts...)
	s.client = client
	return conn.Wait(ctx)
}

type server struct {
	client protocol.Client
}

func notImplemented(method string) *jsonrpc2.Error {
	return jsonrpc2.NewErrorf(jsonrpc2.CodeMethodNotFound, "method %q not yet implemented", method)
}

func (s *server) Initialize(context.Context, *protocol.InitializeParams) (*protocol.InitializeResult, error) {
	return nil, notImplemented("Initialize")
}

func (s *server) Initialized(context.Context, *protocol.InitializedParams) error {
	return notImplemented("Initialized")
}

func (s *server) Shutdown(context.Context) error {
	return notImplemented("Shutdown")
}

func (s *server) Exit(context.Context) error {
	return notImplemented("Exit")
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

func (s *server) DidOpen(context.Context, *protocol.DidOpenTextDocumentParams) error {
	return notImplemented("DidOpen")
}

func (s *server) DidChange(context.Context, *protocol.DidChangeTextDocumentParams) error {
	return nil
}

func (s *server) WillSave(context.Context, *protocol.WillSaveTextDocumentParams) error {
	return notImplemented("WillSave")
}

func (s *server) WillSaveWaitUntil(context.Context, *protocol.WillSaveTextDocumentParams) ([]protocol.TextEdit, error) {
	return nil, notImplemented("WillSaveWaitUntil")
}

func (s *server) DidSave(context.Context, *protocol.DidSaveTextDocumentParams) error {
	return notImplemented("DidSave")
}

func (s *server) DidClose(context.Context, *protocol.DidCloseTextDocumentParams) error {
	return notImplemented("DidClose")
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
	return nil, notImplemented("CodeLens")
}

func (s *server) CodeLensResolve(context.Context, *protocol.CodeLens) (*protocol.CodeLens, error) {
	return nil, notImplemented("CodeLensResolve")
}

func (s *server) DocumentLink(context.Context, *protocol.DocumentLinkParams) ([]protocol.DocumentLink, error) {
	return nil, notImplemented("DocumentLink")
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

func (s *server) Formatting(context.Context, *protocol.DocumentFormattingParams) ([]protocol.TextEdit, error) {
	return nil, notImplemented("Formatting")
}

func (s *server) RangeFormatting(context.Context, *protocol.DocumentRangeFormattingParams) ([]protocol.TextEdit, error) {
	return nil, notImplemented("RangeFormatting")
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
