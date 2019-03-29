// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protocol

import (
	"context"
	"encoding/json"

	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/lsp/xlog"
)

type Server interface {
	Initialize(context.Context, *InitializeParams) (*InitializeResult, error)
	Initialized(context.Context, *InitializedParams) error
	Shutdown(context.Context) error
	Exit(context.Context) error
	DidChangeWorkspaceFolders(context.Context, *DidChangeWorkspaceFoldersParams) error
	DidChangeConfiguration(context.Context, *DidChangeConfigurationParams) error
	DidChangeWatchedFiles(context.Context, *DidChangeWatchedFilesParams) error
	Symbols(context.Context, *WorkspaceSymbolParams) ([]SymbolInformation, error)
	ExecuteCommand(context.Context, *ExecuteCommandParams) (interface{}, error)
	DidOpen(context.Context, *DidOpenTextDocumentParams) error
	DidChange(context.Context, *DidChangeTextDocumentParams) error
	WillSave(context.Context, *WillSaveTextDocumentParams) error
	WillSaveWaitUntil(context.Context, *WillSaveTextDocumentParams) ([]TextEdit, error)
	DidSave(context.Context, *DidSaveTextDocumentParams) error
	DidClose(context.Context, *DidCloseTextDocumentParams) error
	Completion(context.Context, *CompletionParams) (*CompletionList, error)
	CompletionResolve(context.Context, *CompletionItem) (*CompletionItem, error)
	Hover(context.Context, *TextDocumentPositionParams) (*Hover, error)
	SignatureHelp(context.Context, *TextDocumentPositionParams) (*SignatureHelp, error)
	Definition(context.Context, *TextDocumentPositionParams) ([]Location, error)
	TypeDefinition(context.Context, *TextDocumentPositionParams) ([]Location, error)
	Implementation(context.Context, *TextDocumentPositionParams) ([]Location, error)
	References(context.Context, *ReferenceParams) ([]Location, error)
	DocumentHighlight(context.Context, *TextDocumentPositionParams) ([]DocumentHighlight, error)
	DocumentSymbol(context.Context, *DocumentSymbolParams) ([]DocumentSymbol, error)
	CodeAction(context.Context, *CodeActionParams) ([]CodeAction, error)
	CodeLens(context.Context, *CodeLensParams) ([]CodeLens, error)
	CodeLensResolve(context.Context, *CodeLens) (*CodeLens, error)
	DocumentLink(context.Context, *DocumentLinkParams) ([]DocumentLink, error)
	DocumentLinkResolve(context.Context, *DocumentLink) (*DocumentLink, error)
	DocumentColor(context.Context, *DocumentColorParams) ([]ColorInformation, error)
	ColorPresentation(context.Context, *ColorPresentationParams) ([]ColorPresentation, error)
	Formatting(context.Context, *DocumentFormattingParams) ([]TextEdit, error)
	RangeFormatting(context.Context, *DocumentRangeFormattingParams) ([]TextEdit, error)
	OnTypeFormatting(context.Context, *DocumentOnTypeFormattingParams) ([]TextEdit, error)
	Rename(context.Context, *RenameParams) ([]WorkspaceEdit, error)
	FoldingRanges(context.Context, *FoldingRangeParams) ([]FoldingRange, error)
}

func serverHandler(log xlog.Logger, server Server) jsonrpc2.Handler {
	return func(ctx context.Context, conn *jsonrpc2.Conn, r *jsonrpc2.Request) {
		switch r.Method {
		case "initialize":
			var params InitializeParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				sendParseError(ctx, log, conn, r, err)
				return
			}
			resp, err := server.Initialize(ctx, &params)
			if err := conn.Reply(ctx, r, resp, err); err != nil {
				log.Errorf(ctx, "%v", err)
			}

		case "initialized":
			var params InitializedParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				sendParseError(ctx, log, conn, r, err)
				return
			}
			if err := server.Initialized(ctx, &params); err != nil {
				log.Errorf(ctx, "%v", err)
			}

		case "shutdown":
			if r.Params != nil {
				conn.Reply(ctx, r, nil, jsonrpc2.NewErrorf(jsonrpc2.CodeInvalidParams, "Expected no params"))
				return
			}
			if err := server.Shutdown(ctx); err != nil {
				log.Errorf(ctx, "%v", err)
			}

		case "exit":
			if r.Params != nil {
				conn.Reply(ctx, r, nil, jsonrpc2.NewErrorf(jsonrpc2.CodeInvalidParams, "Expected no params"))
				return
			}
			if err := server.Exit(ctx); err != nil {
				log.Errorf(ctx, "%v", err)
			}

		case "$/cancelRequest":
			var params CancelParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				sendParseError(ctx, log, conn, r, err)
				return
			}
			conn.Cancel(params.ID)

		case "workspace/didChangeWorkspaceFolders":
			var params DidChangeWorkspaceFoldersParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				sendParseError(ctx, log, conn, r, err)
				return
			}
			if err := server.DidChangeWorkspaceFolders(ctx, &params); err != nil {
				log.Errorf(ctx, "%v", err)
			}

		case "workspace/didChangeConfiguration":
			var params DidChangeConfigurationParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				sendParseError(ctx, log, conn, r, err)
				return
			}
			if err := server.DidChangeConfiguration(ctx, &params); err != nil {
				log.Errorf(ctx, "%v", err)
			}

		case "workspace/didChangeWatchedFiles":
			var params DidChangeWatchedFilesParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				sendParseError(ctx, log, conn, r, err)
				return
			}
			if err := server.DidChangeWatchedFiles(ctx, &params); err != nil {
				log.Errorf(ctx, "%v", err)
			}

		case "workspace/symbol":
			var params WorkspaceSymbolParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				sendParseError(ctx, log, conn, r, err)
				return
			}
			resp, err := server.Symbols(ctx, &params)
			if err := conn.Reply(ctx, r, resp, err); err != nil {
				log.Errorf(ctx, "%v", err)
			}

		case "workspace/executeCommand":
			var params ExecuteCommandParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				sendParseError(ctx, log, conn, r, err)
				return
			}
			resp, err := server.ExecuteCommand(ctx, &params)
			if err := conn.Reply(ctx, r, resp, err); err != nil {
				log.Errorf(ctx, "%v", err)
			}

		case "textDocument/didOpen":
			var params DidOpenTextDocumentParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				sendParseError(ctx, log, conn, r, err)
				return
			}
			if err := server.DidOpen(ctx, &params); err != nil {
				log.Errorf(ctx, "%v", err)
			}

		case "textDocument/didChange":
			var params DidChangeTextDocumentParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				sendParseError(ctx, log, conn, r, err)
				return
			}
			if err := server.DidChange(ctx, &params); err != nil {
				log.Errorf(ctx, "%v", err)
			}

		case "textDocument/willSave":
			var params WillSaveTextDocumentParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				sendParseError(ctx, log, conn, r, err)
				return
			}
			if err := server.WillSave(ctx, &params); err != nil {
				log.Errorf(ctx, "%v", err)
			}

		case "textDocument/willSaveWaitUntil":
			var params WillSaveTextDocumentParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				sendParseError(ctx, log, conn, r, err)
				return
			}
			resp, err := server.WillSaveWaitUntil(ctx, &params)
			if err := conn.Reply(ctx, r, resp, err); err != nil {
				log.Errorf(ctx, "%v", err)
			}

		case "textDocument/didSave":
			var params DidSaveTextDocumentParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				sendParseError(ctx, log, conn, r, err)
				return
			}
			if err := server.DidSave(ctx, &params); err != nil {
				log.Errorf(ctx, "%v", err)
			}

		case "textDocument/didClose":
			var params DidCloseTextDocumentParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				sendParseError(ctx, log, conn, r, err)
				return
			}
			if err := server.DidClose(ctx, &params); err != nil {
				log.Errorf(ctx, "%v", err)
			}

		case "textDocument/completion":
			var params CompletionParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				sendParseError(ctx, log, conn, r, err)
				return
			}
			resp, err := server.Completion(ctx, &params)
			if err := conn.Reply(ctx, r, resp, err); err != nil {
				log.Errorf(ctx, "%v", err)
			}

		case "completionItem/resolve":
			var params CompletionItem
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				sendParseError(ctx, log, conn, r, err)
				return
			}
			resp, err := server.CompletionResolve(ctx, &params)
			if err := conn.Reply(ctx, r, resp, err); err != nil {
				log.Errorf(ctx, "%v", err)
			}

		case "textDocument/hover":
			var params TextDocumentPositionParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				sendParseError(ctx, log, conn, r, err)
				return
			}
			resp, err := server.Hover(ctx, &params)
			if err := conn.Reply(ctx, r, resp, err); err != nil {
				log.Errorf(ctx, "%v", err)
			}

		case "textDocument/signatureHelp":
			var params TextDocumentPositionParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				sendParseError(ctx, log, conn, r, err)
				return
			}
			resp, err := server.SignatureHelp(ctx, &params)
			if err := conn.Reply(ctx, r, resp, err); err != nil {
				log.Errorf(ctx, "%v", err)
			}

		case "textDocument/definition":
			var params TextDocumentPositionParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				sendParseError(ctx, log, conn, r, err)
				return
			}
			resp, err := server.Definition(ctx, &params)
			if err := conn.Reply(ctx, r, resp, err); err != nil {
				log.Errorf(ctx, "%v", err)
			}

		case "textDocument/typeDefinition":
			var params TextDocumentPositionParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				sendParseError(ctx, log, conn, r, err)
				return
			}
			resp, err := server.TypeDefinition(ctx, &params)
			if err := conn.Reply(ctx, r, resp, err); err != nil {
				log.Errorf(ctx, "%v", err)
			}

		case "textDocument/implementation":
			var params TextDocumentPositionParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				sendParseError(ctx, log, conn, r, err)
				return
			}
			resp, err := server.Implementation(ctx, &params)
			if err := conn.Reply(ctx, r, resp, err); err != nil {
				log.Errorf(ctx, "%v", err)
			}

		case "textDocument/references":
			var params ReferenceParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				sendParseError(ctx, log, conn, r, err)
				return
			}
			resp, err := server.References(ctx, &params)
			if err := conn.Reply(ctx, r, resp, err); err != nil {
				log.Errorf(ctx, "%v", err)
			}

		case "textDocument/documentHighlight":
			var params TextDocumentPositionParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				sendParseError(ctx, log, conn, r, err)
				return
			}
			resp, err := server.DocumentHighlight(ctx, &params)
			if err := conn.Reply(ctx, r, resp, err); err != nil {
				log.Errorf(ctx, "%v", err)
			}

		case "textDocument/documentSymbol":
			var params DocumentSymbolParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				sendParseError(ctx, log, conn, r, err)
				return
			}
			resp, err := server.DocumentSymbol(ctx, &params)
			if err := conn.Reply(ctx, r, resp, err); err != nil {
				log.Errorf(ctx, "%v", err)
			}

		case "textDocument/codeAction":
			var params CodeActionParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				sendParseError(ctx, log, conn, r, err)
				return
			}
			resp, err := server.CodeAction(ctx, &params)
			if err := conn.Reply(ctx, r, resp, err); err != nil {
				log.Errorf(ctx, "%v", err)
			}

		case "textDocument/codeLens":
			var params CodeLensParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				sendParseError(ctx, log, conn, r, err)
				return
			}
			resp, err := server.CodeLens(ctx, &params)
			if err := conn.Reply(ctx, r, resp, err); err != nil {
				log.Errorf(ctx, "%v", err)
			}

		case "codeLens/resolve":
			var params CodeLens
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				sendParseError(ctx, log, conn, r, err)
				return
			}
			resp, err := server.CodeLensResolve(ctx, &params)
			if err := conn.Reply(ctx, r, resp, err); err != nil {
				log.Errorf(ctx, "%v", err)
			}

		case "textDocument/documentLink":
			var params DocumentLinkParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				sendParseError(ctx, log, conn, r, err)
				return
			}
			resp, err := server.DocumentLink(ctx, &params)
			if err := conn.Reply(ctx, r, resp, err); err != nil {
				log.Errorf(ctx, "%v", err)
			}

		case "documentLink/resolve":
			var params DocumentLink
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				sendParseError(ctx, log, conn, r, err)
				return
			}
			resp, err := server.DocumentLinkResolve(ctx, &params)
			if err := conn.Reply(ctx, r, resp, err); err != nil {
				log.Errorf(ctx, "%v", err)
			}

		case "textDocument/documentColor":
			var params DocumentColorParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				sendParseError(ctx, log, conn, r, err)
				return
			}
			resp, err := server.DocumentColor(ctx, &params)
			if err := conn.Reply(ctx, r, resp, err); err != nil {
				log.Errorf(ctx, "%v", err)
			}

		case "textDocument/colorPresentation":
			var params ColorPresentationParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				sendParseError(ctx, log, conn, r, err)
				return
			}
			resp, err := server.ColorPresentation(ctx, &params)
			if err := conn.Reply(ctx, r, resp, err); err != nil {
				log.Errorf(ctx, "%v", err)
			}

		case "textDocument/formatting":
			var params DocumentFormattingParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				sendParseError(ctx, log, conn, r, err)
				return
			}
			resp, err := server.Formatting(ctx, &params)
			if err := conn.Reply(ctx, r, resp, err); err != nil {
				log.Errorf(ctx, "%v", err)
			}

		case "textDocument/rangeFormatting":
			var params DocumentRangeFormattingParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				sendParseError(ctx, log, conn, r, err)
				return
			}
			resp, err := server.RangeFormatting(ctx, &params)
			if err := conn.Reply(ctx, r, resp, err); err != nil {
				log.Errorf(ctx, "%v", err)
			}

		case "textDocument/onTypeFormatting":
			var params DocumentOnTypeFormattingParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				sendParseError(ctx, log, conn, r, err)
				return
			}
			resp, err := server.OnTypeFormatting(ctx, &params)
			if err := conn.Reply(ctx, r, resp, err); err != nil {
				log.Errorf(ctx, "%v", err)
			}

		case "textDocument/rename":
			var params RenameParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				sendParseError(ctx, log, conn, r, err)
				return
			}
			resp, err := server.Rename(ctx, &params)
			if err := conn.Reply(ctx, r, resp, err); err != nil {
				log.Errorf(ctx, "%v", err)
			}

		case "textDocument/foldingRange":
			var params FoldingRangeParams
			if err := json.Unmarshal(*r.Params, &params); err != nil {
				sendParseError(ctx, log, conn, r, err)
				return
			}
			resp, err := server.FoldingRanges(ctx, &params)
			if err := conn.Reply(ctx, r, resp, err); err != nil {
				log.Errorf(ctx, "%v", err)
			}
		default:
			if r.IsNotify() {
				conn.Reply(ctx, r, nil, jsonrpc2.NewErrorf(jsonrpc2.CodeMethodNotFound, "method %q not found", r.Method))
			}
		}
	}
}

type serverDispatcher struct {
	*jsonrpc2.Conn
}

func (s *serverDispatcher) Initialize(ctx context.Context, params *InitializeParams) (*InitializeResult, error) {
	var result InitializeResult
	if err := s.Conn.Call(ctx, "initialize", params, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

func (s *serverDispatcher) Initialized(ctx context.Context, params *InitializedParams) error {
	return s.Conn.Notify(ctx, "initialized", params)
}

func (s *serverDispatcher) Shutdown(ctx context.Context) error {
	return s.Conn.Call(ctx, "shutdown", nil, nil)
}

func (s *serverDispatcher) Exit(ctx context.Context) error {
	return s.Conn.Notify(ctx, "exit", nil)
}

func (s *serverDispatcher) DidChangeWorkspaceFolders(ctx context.Context, params *DidChangeWorkspaceFoldersParams) error {
	return s.Conn.Notify(ctx, "workspace/didChangeWorkspaceFolders", params)
}

func (s *serverDispatcher) DidChangeConfiguration(ctx context.Context, params *DidChangeConfigurationParams) error {
	return s.Conn.Notify(ctx, "workspace/didChangeConfiguration", params)
}

func (s *serverDispatcher) DidChangeWatchedFiles(ctx context.Context, params *DidChangeWatchedFilesParams) error {
	return s.Conn.Notify(ctx, "workspace/didChangeWatchedFiles", params)
}

func (s *serverDispatcher) Symbols(ctx context.Context, params *WorkspaceSymbolParams) ([]SymbolInformation, error) {
	var result []SymbolInformation
	if err := s.Conn.Call(ctx, "workspace/symbol", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) ExecuteCommand(ctx context.Context, params *ExecuteCommandParams) (interface{}, error) {
	var result interface{}
	if err := s.Conn.Call(ctx, "workspace/executeCommand", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) DidOpen(ctx context.Context, params *DidOpenTextDocumentParams) error {
	return s.Conn.Notify(ctx, "textDocument/didOpen", params)
}

func (s *serverDispatcher) DidChange(ctx context.Context, params *DidChangeTextDocumentParams) error {
	return s.Conn.Notify(ctx, "textDocument/didChange", params)
}

func (s *serverDispatcher) WillSave(ctx context.Context, params *WillSaveTextDocumentParams) error {
	return s.Conn.Notify(ctx, "textDocument/willSave", params)
}

func (s *serverDispatcher) WillSaveWaitUntil(ctx context.Context, params *WillSaveTextDocumentParams) ([]TextEdit, error) {
	var result []TextEdit
	if err := s.Conn.Call(ctx, "textDocument/willSaveWaitUntil", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) DidSave(ctx context.Context, params *DidSaveTextDocumentParams) error {
	return s.Conn.Notify(ctx, "textDocument/didSave", params)
}

func (s *serverDispatcher) DidClose(ctx context.Context, params *DidCloseTextDocumentParams) error {
	return s.Conn.Notify(ctx, "textDocument/didClose", params)
}

func (s *serverDispatcher) Completion(ctx context.Context, params *CompletionParams) (*CompletionList, error) {
	var result CompletionList
	if err := s.Conn.Call(ctx, "textDocument/completion", params, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

func (s *serverDispatcher) CompletionResolve(ctx context.Context, params *CompletionItem) (*CompletionItem, error) {
	var result CompletionItem
	if err := s.Conn.Call(ctx, "completionItem/resolve", params, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

func (s *serverDispatcher) Hover(ctx context.Context, params *TextDocumentPositionParams) (*Hover, error) {
	var result Hover
	if err := s.Conn.Call(ctx, "textDocument/hover", params, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

func (s *serverDispatcher) SignatureHelp(ctx context.Context, params *TextDocumentPositionParams) (*SignatureHelp, error) {
	var result SignatureHelp
	if err := s.Conn.Call(ctx, "textDocument/signatureHelp", params, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

func (s *serverDispatcher) Definition(ctx context.Context, params *TextDocumentPositionParams) ([]Location, error) {
	var result []Location
	if err := s.Conn.Call(ctx, "textDocument/definition", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) TypeDefinition(ctx context.Context, params *TextDocumentPositionParams) ([]Location, error) {
	var result []Location
	if err := s.Conn.Call(ctx, "textDocument/typeDefinition", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) Implementation(ctx context.Context, params *TextDocumentPositionParams) ([]Location, error) {
	var result []Location
	if err := s.Conn.Call(ctx, "textDocument/implementation", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) References(ctx context.Context, params *ReferenceParams) ([]Location, error) {
	var result []Location
	if err := s.Conn.Call(ctx, "textDocument/references", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) DocumentHighlight(ctx context.Context, params *TextDocumentPositionParams) ([]DocumentHighlight, error) {
	var result []DocumentHighlight
	if err := s.Conn.Call(ctx, "textDocument/documentHighlight", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) DocumentSymbol(ctx context.Context, params *DocumentSymbolParams) ([]DocumentSymbol, error) {
	var result []DocumentSymbol
	if err := s.Conn.Call(ctx, "textDocument/documentSymbol", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) CodeAction(ctx context.Context, params *CodeActionParams) ([]CodeAction, error) {
	var result []CodeAction
	if err := s.Conn.Call(ctx, "textDocument/codeAction", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) CodeLens(ctx context.Context, params *CodeLensParams) ([]CodeLens, error) {
	var result []CodeLens
	if err := s.Conn.Call(ctx, "textDocument/codeLens", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) CodeLensResolve(ctx context.Context, params *CodeLens) (*CodeLens, error) {
	var result CodeLens
	if err := s.Conn.Call(ctx, "codeLens/resolve", params, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

func (s *serverDispatcher) DocumentLink(ctx context.Context, params *DocumentLinkParams) ([]DocumentLink, error) {
	var result []DocumentLink
	if err := s.Conn.Call(ctx, "textDocument/documentLink", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) DocumentLinkResolve(ctx context.Context, params *DocumentLink) (*DocumentLink, error) {
	var result DocumentLink
	if err := s.Conn.Call(ctx, "documentLink/resolve", params, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

func (s *serverDispatcher) DocumentColor(ctx context.Context, params *DocumentColorParams) ([]ColorInformation, error) {
	var result []ColorInformation
	if err := s.Conn.Call(ctx, "textDocument/documentColor", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) ColorPresentation(ctx context.Context, params *ColorPresentationParams) ([]ColorPresentation, error) {
	var result []ColorPresentation
	if err := s.Conn.Call(ctx, "textDocument/colorPresentation", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) Formatting(ctx context.Context, params *DocumentFormattingParams) ([]TextEdit, error) {
	var result []TextEdit
	if err := s.Conn.Call(ctx, "textDocument/formatting", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) RangeFormatting(ctx context.Context, params *DocumentRangeFormattingParams) ([]TextEdit, error) {
	var result []TextEdit
	if err := s.Conn.Call(ctx, "textDocument/rangeFormatting", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) OnTypeFormatting(ctx context.Context, params *DocumentOnTypeFormattingParams) ([]TextEdit, error) {
	var result []TextEdit
	if err := s.Conn.Call(ctx, "textDocument/onTypeFormatting", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) Rename(ctx context.Context, params *RenameParams) ([]WorkspaceEdit, error) {
	var result []WorkspaceEdit
	if err := s.Conn.Call(ctx, "textDocument/rename", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) FoldingRanges(ctx context.Context, params *FoldingRangeParams) ([]FoldingRange, error) {
	var result []FoldingRange
	if err := s.Conn.Call(ctx, "textDocument/foldingRanges", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

type CancelParams struct {
	/**
	 * The request id to cancel.
	 */
	ID jsonrpc2.ID `json:"id"`
}
