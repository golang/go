package protocol

// Package protocol contains data types and code for LSP jsonrpcs
// generated automatically from vscode-languageserver-node
// commit: 399de64448129835b53c7efe8962de91681d6cde
// last fetched Tue Jul 28 2020 09:32:20 GMT-0400 (Eastern Daylight Time)

// Code generated (see typescript/README.md) DO NOT EDIT.

import (
	"context"
	"encoding/json"
	"fmt"

	"golang.org/x/tools/internal/jsonrpc2"
)

type Server interface {
	DidChangeWorkspaceFolders(context.Context, *DidChangeWorkspaceFoldersParams) error
	WorkDoneProgressCancel(context.Context, *WorkDoneProgressCancelParams) error
	Initialized(context.Context, *InitializedParams) error
	Exit(context.Context) error
	DidChangeConfiguration(context.Context, *DidChangeConfigurationParams) error
	DidOpen(context.Context, *DidOpenTextDocumentParams) error
	DidChange(context.Context, *DidChangeTextDocumentParams) error
	DidClose(context.Context, *DidCloseTextDocumentParams) error
	DidSave(context.Context, *DidSaveTextDocumentParams) error
	WillSave(context.Context, *WillSaveTextDocumentParams) error
	DidChangeWatchedFiles(context.Context, *DidChangeWatchedFilesParams) error
	SetTrace(context.Context, *SetTraceParams) error
	LogTrace(context.Context, *LogTraceParams) error
	Implementation(context.Context, *ImplementationParams) (Definition /*Definition | DefinitionLink[] | null*/, error)
	TypeDefinition(context.Context, *TypeDefinitionParams) (Definition /*Definition | DefinitionLink[] | null*/, error)
	DocumentColor(context.Context, *DocumentColorParams) ([]ColorInformation, error)
	ColorPresentation(context.Context, *ColorPresentationParams) ([]ColorPresentation, error)
	FoldingRange(context.Context, *FoldingRangeParams) ([]FoldingRange /*FoldingRange[] | null*/, error)
	Declaration(context.Context, *DeclarationParams) (Declaration /*Declaration | DeclarationLink[] | null*/, error)
	SelectionRange(context.Context, *SelectionRangeParams) ([]SelectionRange /*SelectionRange[] | null*/, error)
	PrepareCallHierarchy(context.Context, *CallHierarchyPrepareParams) ([]CallHierarchyItem /*CallHierarchyItem[] | null*/, error)
	IncomingCalls(context.Context, *CallHierarchyIncomingCallsParams) ([]CallHierarchyIncomingCall /*CallHierarchyIncomingCall[] | null*/, error)
	OutgoingCalls(context.Context, *CallHierarchyOutgoingCallsParams) ([]CallHierarchyOutgoingCall /*CallHierarchyOutgoingCall[] | null*/, error)
	Initialize(context.Context, *ParamInitialize) (*InitializeResult, error)
	Shutdown(context.Context) error
	WillSaveWaitUntil(context.Context, *WillSaveTextDocumentParams) ([]TextEdit /*TextEdit[] | null*/, error)
	Completion(context.Context, *CompletionParams) (*CompletionList /*CompletionItem[] | CompletionList | null*/, error)
	Resolve(context.Context, *CompletionItem) (*CompletionItem, error)
	Hover(context.Context, *HoverParams) (*Hover /*Hover | null*/, error)
	SignatureHelp(context.Context, *SignatureHelpParams) (*SignatureHelp /*SignatureHelp | null*/, error)
	Definition(context.Context, *DefinitionParams) (Definition /*Definition | DefinitionLink[] | null*/, error)
	References(context.Context, *ReferenceParams) ([]Location /*Location[] | null*/, error)
	DocumentHighlight(context.Context, *DocumentHighlightParams) ([]DocumentHighlight /*DocumentHighlight[] | null*/, error)
	DocumentSymbol(context.Context, *DocumentSymbolParams) ([]interface{} /*SymbolInformation[] | DocumentSymbol[] | null*/, error)
	CodeAction(context.Context, *CodeActionParams) ([]CodeAction /*(Command | CodeAction)[] | null*/, error)
	Symbol(context.Context, *WorkspaceSymbolParams) ([]SymbolInformation /*SymbolInformation[] | null*/, error)
	CodeLens(context.Context, *CodeLensParams) ([]CodeLens /*CodeLens[] | null*/, error)
	ResolveCodeLens(context.Context, *CodeLens) (*CodeLens, error)
	DocumentLink(context.Context, *DocumentLinkParams) ([]DocumentLink /*DocumentLink[] | null*/, error)
	ResolveDocumentLink(context.Context, *DocumentLink) (*DocumentLink, error)
	Formatting(context.Context, *DocumentFormattingParams) ([]TextEdit /*TextEdit[] | null*/, error)
	RangeFormatting(context.Context, *DocumentRangeFormattingParams) ([]TextEdit /*TextEdit[] | null*/, error)
	OnTypeFormatting(context.Context, *DocumentOnTypeFormattingParams) ([]TextEdit /*TextEdit[] | null*/, error)
	Rename(context.Context, *RenameParams) (*WorkspaceEdit /*WorkspaceEdit | null*/, error)
	PrepareRename(context.Context, *PrepareRenameParams) (*Range /*Range | { range: Range, placeholder: string } | null*/, error)
	ExecuteCommand(context.Context, *ExecuteCommandParams) (interface{} /*any | null*/, error)
	SemanticTokensFull(context.Context, *SemanticTokensParams) (*SemanticTokens /*SemanticTokens | null*/, error)
	SemanticTokensFullDelta(context.Context, *SemanticTokensDeltaParams) (interface{} /* SemanticTokens | SemanticTokensDelta | nil*/, error)
	SemanticTokensRange(context.Context, *SemanticTokensRangeParams) (*SemanticTokens /*SemanticTokens | null*/, error)
	NonstandardRequest(ctx context.Context, method string, params interface{}) (interface{}, error)
}

func serverDispatch(ctx context.Context, server Server, reply jsonrpc2.Replier, r jsonrpc2.Request) (bool, error) {
	switch r.Method() {
	case "workspace/didChangeWorkspaceFolders": // notif
		var params DidChangeWorkspaceFoldersParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := server.DidChangeWorkspaceFolders(ctx, &params)
		return true, reply(ctx, nil, err)
	case "window/workDoneProgress/cancel": // notif
		var params WorkDoneProgressCancelParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := server.WorkDoneProgressCancel(ctx, &params)
		return true, reply(ctx, nil, err)
	case "initialized": // notif
		var params InitializedParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := server.Initialized(ctx, &params)
		return true, reply(ctx, nil, err)
	case "exit": // notif
		err := server.Exit(ctx)
		return true, reply(ctx, nil, err)
	case "workspace/didChangeConfiguration": // notif
		var params DidChangeConfigurationParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := server.DidChangeConfiguration(ctx, &params)
		return true, reply(ctx, nil, err)
	case "textDocument/didOpen": // notif
		var params DidOpenTextDocumentParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := server.DidOpen(ctx, &params)
		return true, reply(ctx, nil, err)
	case "textDocument/didChange": // notif
		var params DidChangeTextDocumentParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := server.DidChange(ctx, &params)
		return true, reply(ctx, nil, err)
	case "textDocument/didClose": // notif
		var params DidCloseTextDocumentParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := server.DidClose(ctx, &params)
		return true, reply(ctx, nil, err)
	case "textDocument/didSave": // notif
		var params DidSaveTextDocumentParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := server.DidSave(ctx, &params)
		return true, reply(ctx, nil, err)
	case "textDocument/willSave": // notif
		var params WillSaveTextDocumentParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := server.WillSave(ctx, &params)
		return true, reply(ctx, nil, err)
	case "workspace/didChangeWatchedFiles": // notif
		var params DidChangeWatchedFilesParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := server.DidChangeWatchedFiles(ctx, &params)
		return true, reply(ctx, nil, err)
	case "$/setTrace": // notif
		var params SetTraceParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := server.SetTrace(ctx, &params)
		return true, reply(ctx, nil, err)
	case "$/logTrace": // notif
		var params LogTraceParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := server.LogTrace(ctx, &params)
		return true, reply(ctx, nil, err)
	case "textDocument/implementation": // req
		var params ImplementationParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.Implementation(ctx, &params)
		return true, reply(ctx, resp, err)
	case "textDocument/typeDefinition": // req
		var params TypeDefinitionParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.TypeDefinition(ctx, &params)
		return true, reply(ctx, resp, err)
	case "textDocument/documentColor": // req
		var params DocumentColorParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.DocumentColor(ctx, &params)
		return true, reply(ctx, resp, err)
	case "textDocument/colorPresentation": // req
		var params ColorPresentationParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.ColorPresentation(ctx, &params)
		return true, reply(ctx, resp, err)
	case "textDocument/foldingRange": // req
		var params FoldingRangeParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.FoldingRange(ctx, &params)
		return true, reply(ctx, resp, err)
	case "textDocument/declaration": // req
		var params DeclarationParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.Declaration(ctx, &params)
		return true, reply(ctx, resp, err)
	case "textDocument/selectionRange": // req
		var params SelectionRangeParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.SelectionRange(ctx, &params)
		return true, reply(ctx, resp, err)
	case "textDocument/prepareCallHierarchy": // req
		var params CallHierarchyPrepareParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.PrepareCallHierarchy(ctx, &params)
		return true, reply(ctx, resp, err)
	case "callHierarchy/incomingCalls": // req
		var params CallHierarchyIncomingCallsParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.IncomingCalls(ctx, &params)
		return true, reply(ctx, resp, err)
	case "callHierarchy/outgoingCalls": // req
		var params CallHierarchyOutgoingCallsParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.OutgoingCalls(ctx, &params)
		return true, reply(ctx, resp, err)
	case "initialize": // req
		var params ParamInitialize
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.Initialize(ctx, &params)
		return true, reply(ctx, resp, err)
	case "shutdown": // req
		if len(r.Params()) > 0 {
			return true, reply(ctx, nil, fmt.Errorf("%w: expected no params", jsonrpc2.ErrInvalidParams))
		}
		err := server.Shutdown(ctx)
		return true, reply(ctx, nil, err)
	case "textDocument/willSaveWaitUntil": // req
		var params WillSaveTextDocumentParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.WillSaveWaitUntil(ctx, &params)
		return true, reply(ctx, resp, err)
	case "textDocument/completion": // req
		var params CompletionParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.Completion(ctx, &params)
		return true, reply(ctx, resp, err)
	case "completionItem/resolve": // req
		var params CompletionItem
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.Resolve(ctx, &params)
		return true, reply(ctx, resp, err)
	case "textDocument/hover": // req
		var params HoverParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.Hover(ctx, &params)
		return true, reply(ctx, resp, err)
	case "textDocument/signatureHelp": // req
		var params SignatureHelpParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.SignatureHelp(ctx, &params)
		return true, reply(ctx, resp, err)
	case "textDocument/definition": // req
		var params DefinitionParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.Definition(ctx, &params)
		return true, reply(ctx, resp, err)
	case "textDocument/references": // req
		var params ReferenceParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.References(ctx, &params)
		return true, reply(ctx, resp, err)
	case "textDocument/documentHighlight": // req
		var params DocumentHighlightParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.DocumentHighlight(ctx, &params)
		return true, reply(ctx, resp, err)
	case "textDocument/documentSymbol": // req
		var params DocumentSymbolParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.DocumentSymbol(ctx, &params)
		return true, reply(ctx, resp, err)
	case "textDocument/codeAction": // req
		var params CodeActionParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.CodeAction(ctx, &params)
		return true, reply(ctx, resp, err)
	case "workspace/symbol": // req
		var params WorkspaceSymbolParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.Symbol(ctx, &params)
		return true, reply(ctx, resp, err)
	case "textDocument/codeLens": // req
		var params CodeLensParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.CodeLens(ctx, &params)
		return true, reply(ctx, resp, err)
	case "codeLens/resolve": // req
		var params CodeLens
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.ResolveCodeLens(ctx, &params)
		return true, reply(ctx, resp, err)
	case "textDocument/documentLink": // req
		var params DocumentLinkParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.DocumentLink(ctx, &params)
		return true, reply(ctx, resp, err)
	case "documentLink/resolve": // req
		var params DocumentLink
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.ResolveDocumentLink(ctx, &params)
		return true, reply(ctx, resp, err)
	case "textDocument/formatting": // req
		var params DocumentFormattingParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.Formatting(ctx, &params)
		return true, reply(ctx, resp, err)
	case "textDocument/rangeFormatting": // req
		var params DocumentRangeFormattingParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.RangeFormatting(ctx, &params)
		return true, reply(ctx, resp, err)
	case "textDocument/onTypeFormatting": // req
		var params DocumentOnTypeFormattingParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.OnTypeFormatting(ctx, &params)
		return true, reply(ctx, resp, err)
	case "textDocument/rename": // req
		var params RenameParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.Rename(ctx, &params)
		return true, reply(ctx, resp, err)
	case "textDocument/prepareRename": // req
		var params PrepareRenameParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.PrepareRename(ctx, &params)
		return true, reply(ctx, resp, err)
	case "workspace/executeCommand": // req
		var params ExecuteCommandParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.ExecuteCommand(ctx, &params)
		return true, reply(ctx, resp, err)
	case "textDocument/semanticTokens/full": // req
		var params SemanticTokensParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.SemanticTokensFull(ctx, &params)
		return true, reply(ctx, resp, err)
	case "textDocument/semanticTokens/full/delta": // req
		var params SemanticTokensDeltaParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.SemanticTokensFullDelta(ctx, &params)
		return true, reply(ctx, resp, err)
	case "textDocument/semanticTokens/range": // req
		var params SemanticTokensRangeParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.SemanticTokensRange(ctx, &params)
		return true, reply(ctx, resp, err)

	default:
		return false, nil
	}
}

func (s *serverDispatcher) DidChangeWorkspaceFolders(ctx context.Context, params *DidChangeWorkspaceFoldersParams) error {
	return s.Conn.Notify(ctx, "workspace/didChangeWorkspaceFolders", params)
}

func (s *serverDispatcher) WorkDoneProgressCancel(ctx context.Context, params *WorkDoneProgressCancelParams) error {
	return s.Conn.Notify(ctx, "window/workDoneProgress/cancel", params)
}

func (s *serverDispatcher) Initialized(ctx context.Context, params *InitializedParams) error {
	return s.Conn.Notify(ctx, "initialized", params)
}

func (s *serverDispatcher) Exit(ctx context.Context) error {
	return s.Conn.Notify(ctx, "exit", nil)
}

func (s *serverDispatcher) DidChangeConfiguration(ctx context.Context, params *DidChangeConfigurationParams) error {
	return s.Conn.Notify(ctx, "workspace/didChangeConfiguration", params)
}

func (s *serverDispatcher) DidOpen(ctx context.Context, params *DidOpenTextDocumentParams) error {
	return s.Conn.Notify(ctx, "textDocument/didOpen", params)
}

func (s *serverDispatcher) DidChange(ctx context.Context, params *DidChangeTextDocumentParams) error {
	return s.Conn.Notify(ctx, "textDocument/didChange", params)
}

func (s *serverDispatcher) DidClose(ctx context.Context, params *DidCloseTextDocumentParams) error {
	return s.Conn.Notify(ctx, "textDocument/didClose", params)
}

func (s *serverDispatcher) DidSave(ctx context.Context, params *DidSaveTextDocumentParams) error {
	return s.Conn.Notify(ctx, "textDocument/didSave", params)
}

func (s *serverDispatcher) WillSave(ctx context.Context, params *WillSaveTextDocumentParams) error {
	return s.Conn.Notify(ctx, "textDocument/willSave", params)
}

func (s *serverDispatcher) DidChangeWatchedFiles(ctx context.Context, params *DidChangeWatchedFilesParams) error {
	return s.Conn.Notify(ctx, "workspace/didChangeWatchedFiles", params)
}

func (s *serverDispatcher) SetTrace(ctx context.Context, params *SetTraceParams) error {
	return s.Conn.Notify(ctx, "$/setTrace", params)
}

func (s *serverDispatcher) LogTrace(ctx context.Context, params *LogTraceParams) error {
	return s.Conn.Notify(ctx, "$/logTrace", params)
}
func (s *serverDispatcher) Implementation(ctx context.Context, params *ImplementationParams) (Definition /*Definition | DefinitionLink[] | null*/, error) {
	var result Definition /*Definition | DefinitionLink[] | null*/
	if err := Call(ctx, s.Conn, "textDocument/implementation", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) TypeDefinition(ctx context.Context, params *TypeDefinitionParams) (Definition /*Definition | DefinitionLink[] | null*/, error) {
	var result Definition /*Definition | DefinitionLink[] | null*/
	if err := Call(ctx, s.Conn, "textDocument/typeDefinition", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) DocumentColor(ctx context.Context, params *DocumentColorParams) ([]ColorInformation, error) {
	var result []ColorInformation
	if err := Call(ctx, s.Conn, "textDocument/documentColor", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) ColorPresentation(ctx context.Context, params *ColorPresentationParams) ([]ColorPresentation, error) {
	var result []ColorPresentation
	if err := Call(ctx, s.Conn, "textDocument/colorPresentation", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) FoldingRange(ctx context.Context, params *FoldingRangeParams) ([]FoldingRange /*FoldingRange[] | null*/, error) {
	var result []FoldingRange /*FoldingRange[] | null*/
	if err := Call(ctx, s.Conn, "textDocument/foldingRange", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) Declaration(ctx context.Context, params *DeclarationParams) (Declaration /*Declaration | DeclarationLink[] | null*/, error) {
	var result Declaration /*Declaration | DeclarationLink[] | null*/
	if err := Call(ctx, s.Conn, "textDocument/declaration", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) SelectionRange(ctx context.Context, params *SelectionRangeParams) ([]SelectionRange /*SelectionRange[] | null*/, error) {
	var result []SelectionRange /*SelectionRange[] | null*/
	if err := Call(ctx, s.Conn, "textDocument/selectionRange", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) PrepareCallHierarchy(ctx context.Context, params *CallHierarchyPrepareParams) ([]CallHierarchyItem /*CallHierarchyItem[] | null*/, error) {
	var result []CallHierarchyItem /*CallHierarchyItem[] | null*/
	if err := Call(ctx, s.Conn, "textDocument/prepareCallHierarchy", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) IncomingCalls(ctx context.Context, params *CallHierarchyIncomingCallsParams) ([]CallHierarchyIncomingCall /*CallHierarchyIncomingCall[] | null*/, error) {
	var result []CallHierarchyIncomingCall /*CallHierarchyIncomingCall[] | null*/
	if err := Call(ctx, s.Conn, "callHierarchy/incomingCalls", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) OutgoingCalls(ctx context.Context, params *CallHierarchyOutgoingCallsParams) ([]CallHierarchyOutgoingCall /*CallHierarchyOutgoingCall[] | null*/, error) {
	var result []CallHierarchyOutgoingCall /*CallHierarchyOutgoingCall[] | null*/
	if err := Call(ctx, s.Conn, "callHierarchy/outgoingCalls", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) Initialize(ctx context.Context, params *ParamInitialize) (*InitializeResult, error) {
	var result *InitializeResult
	if err := Call(ctx, s.Conn, "initialize", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) Shutdown(ctx context.Context) error {
	return Call(ctx, s.Conn, "shutdown", nil, nil)
}

func (s *serverDispatcher) WillSaveWaitUntil(ctx context.Context, params *WillSaveTextDocumentParams) ([]TextEdit /*TextEdit[] | null*/, error) {
	var result []TextEdit /*TextEdit[] | null*/
	if err := Call(ctx, s.Conn, "textDocument/willSaveWaitUntil", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) Completion(ctx context.Context, params *CompletionParams) (*CompletionList /*CompletionItem[] | CompletionList | null*/, error) {
	var result *CompletionList /*CompletionItem[] | CompletionList | null*/
	if err := Call(ctx, s.Conn, "textDocument/completion", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) Resolve(ctx context.Context, params *CompletionItem) (*CompletionItem, error) {
	var result *CompletionItem
	if err := Call(ctx, s.Conn, "completionItem/resolve", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) Hover(ctx context.Context, params *HoverParams) (*Hover /*Hover | null*/, error) {
	var result *Hover /*Hover | null*/
	if err := Call(ctx, s.Conn, "textDocument/hover", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) SignatureHelp(ctx context.Context, params *SignatureHelpParams) (*SignatureHelp /*SignatureHelp | null*/, error) {
	var result *SignatureHelp /*SignatureHelp | null*/
	if err := Call(ctx, s.Conn, "textDocument/signatureHelp", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) Definition(ctx context.Context, params *DefinitionParams) (Definition /*Definition | DefinitionLink[] | null*/, error) {
	var result Definition /*Definition | DefinitionLink[] | null*/
	if err := Call(ctx, s.Conn, "textDocument/definition", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) References(ctx context.Context, params *ReferenceParams) ([]Location /*Location[] | null*/, error) {
	var result []Location /*Location[] | null*/
	if err := Call(ctx, s.Conn, "textDocument/references", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) DocumentHighlight(ctx context.Context, params *DocumentHighlightParams) ([]DocumentHighlight /*DocumentHighlight[] | null*/, error) {
	var result []DocumentHighlight /*DocumentHighlight[] | null*/
	if err := Call(ctx, s.Conn, "textDocument/documentHighlight", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) DocumentSymbol(ctx context.Context, params *DocumentSymbolParams) ([]interface{} /*SymbolInformation[] | DocumentSymbol[] | null*/, error) {
	var result []interface{} /*SymbolInformation[] | DocumentSymbol[] | null*/
	if err := Call(ctx, s.Conn, "textDocument/documentSymbol", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) CodeAction(ctx context.Context, params *CodeActionParams) ([]CodeAction /*(Command | CodeAction)[] | null*/, error) {
	var result []CodeAction /*(Command | CodeAction)[] | null*/
	if err := Call(ctx, s.Conn, "textDocument/codeAction", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) Symbol(ctx context.Context, params *WorkspaceSymbolParams) ([]SymbolInformation /*SymbolInformation[] | null*/, error) {
	var result []SymbolInformation /*SymbolInformation[] | null*/
	if err := Call(ctx, s.Conn, "workspace/symbol", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) CodeLens(ctx context.Context, params *CodeLensParams) ([]CodeLens /*CodeLens[] | null*/, error) {
	var result []CodeLens /*CodeLens[] | null*/
	if err := Call(ctx, s.Conn, "textDocument/codeLens", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) ResolveCodeLens(ctx context.Context, params *CodeLens) (*CodeLens, error) {
	var result *CodeLens
	if err := Call(ctx, s.Conn, "codeLens/resolve", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) DocumentLink(ctx context.Context, params *DocumentLinkParams) ([]DocumentLink /*DocumentLink[] | null*/, error) {
	var result []DocumentLink /*DocumentLink[] | null*/
	if err := Call(ctx, s.Conn, "textDocument/documentLink", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) ResolveDocumentLink(ctx context.Context, params *DocumentLink) (*DocumentLink, error) {
	var result *DocumentLink
	if err := Call(ctx, s.Conn, "documentLink/resolve", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) Formatting(ctx context.Context, params *DocumentFormattingParams) ([]TextEdit /*TextEdit[] | null*/, error) {
	var result []TextEdit /*TextEdit[] | null*/
	if err := Call(ctx, s.Conn, "textDocument/formatting", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) RangeFormatting(ctx context.Context, params *DocumentRangeFormattingParams) ([]TextEdit /*TextEdit[] | null*/, error) {
	var result []TextEdit /*TextEdit[] | null*/
	if err := Call(ctx, s.Conn, "textDocument/rangeFormatting", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) OnTypeFormatting(ctx context.Context, params *DocumentOnTypeFormattingParams) ([]TextEdit /*TextEdit[] | null*/, error) {
	var result []TextEdit /*TextEdit[] | null*/
	if err := Call(ctx, s.Conn, "textDocument/onTypeFormatting", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) Rename(ctx context.Context, params *RenameParams) (*WorkspaceEdit /*WorkspaceEdit | null*/, error) {
	var result *WorkspaceEdit /*WorkspaceEdit | null*/
	if err := Call(ctx, s.Conn, "textDocument/rename", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) PrepareRename(ctx context.Context, params *PrepareRenameParams) (*Range /*Range | { range: Range, placeholder: string } | null*/, error) {
	var result *Range /*Range | { range: Range, placeholder: string } | null*/
	if err := Call(ctx, s.Conn, "textDocument/prepareRename", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) ExecuteCommand(ctx context.Context, params *ExecuteCommandParams) (interface{} /*any | null*/, error) {
	var result interface{} /*any | null*/
	if err := Call(ctx, s.Conn, "workspace/executeCommand", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) SemanticTokensFull(ctx context.Context, params *SemanticTokensParams) (*SemanticTokens /*SemanticTokens | null*/, error) {
	var result *SemanticTokens /*SemanticTokens | null*/
	if err := Call(ctx, s.Conn, "textDocument/semanticTokens/full", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) SemanticTokensFullDelta(ctx context.Context, params *SemanticTokensDeltaParams) (interface{} /* SemanticTokens | SemanticTokensDelta | nil*/, error) {
	var result interface{} /* SemanticTokens | SemanticTokensDelta | nil*/
	if err := Call(ctx, s.Conn, "textDocument/semanticTokens/full/delta", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) SemanticTokensRange(ctx context.Context, params *SemanticTokensRangeParams) (*SemanticTokens /*SemanticTokens | null*/, error) {
	var result *SemanticTokens /*SemanticTokens | null*/
	if err := Call(ctx, s.Conn, "textDocument/semanticTokens/range", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) NonstandardRequest(ctx context.Context, method string, params interface{}) (interface{}, error) {
	var result interface{}
	if err := Call(ctx, s.Conn, method, params, &result); err != nil {
		return nil, err
	}
	return result, nil
}
