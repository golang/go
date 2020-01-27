package protocol

// Package protocol contains data types and code for LSP jsonrpcs
// generated automatically from vscode-languageserver-node
// commit: 7b90c29d0cb5cd7b9c41084f6cb3781a955adeba
// last fetched Thu Jan 23 2020 11:10:31 GMT-0500 (Eastern Standard Time)

// Code generated (see typescript/README.md) DO NOT EDIT.

import (
	"context"
	"encoding/json"

	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/telemetry/log"
	"golang.org/x/tools/internal/xcontext"
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
	Progress(context.Context, *ProgressParams) error
	SetTraceNotification(context.Context, *SetTraceParams) error
	LogTraceNotification(context.Context, *LogTraceParams) error
	Implementation(context.Context, *ImplementationParams) (Definition /*Definition | DefinitionLink[] | null*/, error)
	TypeDefinition(context.Context, *TypeDefinitionParams) (Definition /*Definition | DefinitionLink[] | null*/, error)
	DocumentColor(context.Context, *DocumentColorParams) ([]ColorInformation, error)
	ColorPresentation(context.Context, *ColorPresentationParams) ([]ColorPresentation, error)
	FoldingRange(context.Context, *FoldingRangeParams) ([]FoldingRange /*FoldingRange[] | null*/, error)
	Declaration(context.Context, *DeclarationParams) (Declaration /*Declaration | DeclarationLink[] | null*/, error)
	SelectionRange(context.Context, *SelectionRangeParams) ([]SelectionRange /*SelectionRange[] | null*/, error)
	WorkDoneProgressCreate(context.Context, *WorkDoneProgressCreateParams) error
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
	DocumentSymbol(context.Context, *DocumentSymbolParams) ([]DocumentSymbol /*SymbolInformation[] | DocumentSymbol[] | null*/, error)
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
	PrepareRename(context.Context, *PrepareRenameParams) (interface{} /* Range | struct{;  Range Range`json:"range"`;  Placeholder string`json:"placeholder"`; } | nil*/, error)
	ExecuteCommand(context.Context, *ExecuteCommandParams) (interface{} /*any | null*/, error)
	PrepareCallHierarchy(context.Context, *CallHierarchyPrepareParams) ([]CallHierarchyItem /*CallHierarchyItem[] | null*/, error)
	IncomingCalls(context.Context, *CallHierarchyIncomingCallsParams) ([]CallHierarchyIncomingCall /*CallHierarchyIncomingCall[] | null*/, error)
	OutgoingCalls(context.Context, *CallHierarchyOutgoingCallsParams) ([]CallHierarchyOutgoingCall /*CallHierarchyOutgoingCall[] | null*/, error)
	SemanticTokens(context.Context, *SemanticTokensParams) (*SemanticTokens /*SemanticTokens | null*/, error)
	SemanticTokensEdits(context.Context, *SemanticTokensEditsParams) (interface{} /* SemanticTokens | SemanticTokensEdits | nil*/, error)
	SemanticTokensRange(context.Context, *SemanticTokensRangeParams) (*SemanticTokens /*SemanticTokens | null*/, error)
	NonstandardRequest(ctx context.Context, method string, params interface{}) (interface{}, error)
}

func (h serverHandler) Deliver(ctx context.Context, r *jsonrpc2.Request, delivered bool) bool {
	if delivered {
		return false
	}
	if ctx.Err() != nil {
		ctx := xcontext.Detach(ctx)
		r.Reply(ctx, nil, jsonrpc2.NewErrorf(RequestCancelledError, ""))
		return true
	}
	switch r.Method {
	case "workspace/didChangeWorkspaceFolders": // notif
		var params DidChangeWorkspaceFoldersParams
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		if err := h.server.DidChangeWorkspaceFolders(ctx, &params); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "window/workDoneProgress/cancel": // notif
		var params WorkDoneProgressCancelParams
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		if err := h.server.WorkDoneProgressCancel(ctx, &params); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "initialized": // notif
		var params InitializedParams
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		if err := h.server.Initialized(ctx, &params); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "exit": // notif
		if err := h.server.Exit(ctx); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "workspace/didChangeConfiguration": // notif
		var params DidChangeConfigurationParams
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		if err := h.server.DidChangeConfiguration(ctx, &params); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "textDocument/didOpen": // notif
		var params DidOpenTextDocumentParams
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		if err := h.server.DidOpen(ctx, &params); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "textDocument/didChange": // notif
		var params DidChangeTextDocumentParams
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		if err := h.server.DidChange(ctx, &params); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "textDocument/didClose": // notif
		var params DidCloseTextDocumentParams
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		if err := h.server.DidClose(ctx, &params); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "textDocument/didSave": // notif
		var params DidSaveTextDocumentParams
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		if err := h.server.DidSave(ctx, &params); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "textDocument/willSave": // notif
		var params WillSaveTextDocumentParams
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		if err := h.server.WillSave(ctx, &params); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "workspace/didChangeWatchedFiles": // notif
		var params DidChangeWatchedFilesParams
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		if err := h.server.DidChangeWatchedFiles(ctx, &params); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "$/progress": // notif
		var params ProgressParams
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		if err := h.server.Progress(ctx, &params); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "$/setTraceNotification": // notif
		var params SetTraceParams
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		if err := h.server.SetTraceNotification(ctx, &params); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "$/logTraceNotification": // notif
		var params LogTraceParams
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		if err := h.server.LogTraceNotification(ctx, &params); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "textDocument/implementation": // req
		var params ImplementationParams
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		resp, err := h.server.Implementation(ctx, &params)
		if err := r.Reply(ctx, resp, err); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "textDocument/typeDefinition": // req
		var params TypeDefinitionParams
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		resp, err := h.server.TypeDefinition(ctx, &params)
		if err := r.Reply(ctx, resp, err); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "textDocument/documentColor": // req
		var params DocumentColorParams
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		resp, err := h.server.DocumentColor(ctx, &params)
		if err := r.Reply(ctx, resp, err); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "textDocument/colorPresentation": // req
		var params ColorPresentationParams
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		resp, err := h.server.ColorPresentation(ctx, &params)
		if err := r.Reply(ctx, resp, err); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "textDocument/foldingRange": // req
		var params FoldingRangeParams
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		resp, err := h.server.FoldingRange(ctx, &params)
		if err := r.Reply(ctx, resp, err); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "textDocument/declaration": // req
		var params DeclarationParams
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		resp, err := h.server.Declaration(ctx, &params)
		if err := r.Reply(ctx, resp, err); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "textDocument/selectionRange": // req
		var params SelectionRangeParams
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		resp, err := h.server.SelectionRange(ctx, &params)
		if err := r.Reply(ctx, resp, err); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "window/workDoneProgress/create": // req
		var params WorkDoneProgressCreateParams
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		err := h.server.WorkDoneProgressCreate(ctx, &params)
		if err := r.Reply(ctx, nil, err); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "initialize": // req
		var params ParamInitialize
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		resp, err := h.server.Initialize(ctx, &params)
		if err := r.Reply(ctx, resp, err); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "shutdown": // req
		if r.Params != nil {
			r.Reply(ctx, nil, jsonrpc2.NewErrorf(jsonrpc2.CodeInvalidParams, "Expected no params"))
			return true
		}
		err := h.server.Shutdown(ctx)
		if err := r.Reply(ctx, nil, err); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "textDocument/willSaveWaitUntil": // req
		var params WillSaveTextDocumentParams
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		resp, err := h.server.WillSaveWaitUntil(ctx, &params)
		if err := r.Reply(ctx, resp, err); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "textDocument/completion": // req
		var params CompletionParams
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		resp, err := h.server.Completion(ctx, &params)
		if err := r.Reply(ctx, resp, err); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "completionItem/resolve": // req
		var params CompletionItem
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		resp, err := h.server.Resolve(ctx, &params)
		if err := r.Reply(ctx, resp, err); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "textDocument/hover": // req
		var params HoverParams
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		resp, err := h.server.Hover(ctx, &params)
		if err := r.Reply(ctx, resp, err); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "textDocument/signatureHelp": // req
		var params SignatureHelpParams
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		resp, err := h.server.SignatureHelp(ctx, &params)
		if err := r.Reply(ctx, resp, err); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "textDocument/definition": // req
		var params DefinitionParams
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		resp, err := h.server.Definition(ctx, &params)
		if err := r.Reply(ctx, resp, err); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "textDocument/references": // req
		var params ReferenceParams
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		resp, err := h.server.References(ctx, &params)
		if err := r.Reply(ctx, resp, err); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "textDocument/documentHighlight": // req
		var params DocumentHighlightParams
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		resp, err := h.server.DocumentHighlight(ctx, &params)
		if err := r.Reply(ctx, resp, err); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "textDocument/documentSymbol": // req
		var params DocumentSymbolParams
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		resp, err := h.server.DocumentSymbol(ctx, &params)
		if err := r.Reply(ctx, resp, err); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "textDocument/codeAction": // req
		var params CodeActionParams
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		resp, err := h.server.CodeAction(ctx, &params)
		if err := r.Reply(ctx, resp, err); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "workspace/symbol": // req
		var params WorkspaceSymbolParams
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		resp, err := h.server.Symbol(ctx, &params)
		if err := r.Reply(ctx, resp, err); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "textDocument/codeLens": // req
		var params CodeLensParams
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		resp, err := h.server.CodeLens(ctx, &params)
		if err := r.Reply(ctx, resp, err); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "codeLens/resolve": // req
		var params CodeLens
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		resp, err := h.server.ResolveCodeLens(ctx, &params)
		if err := r.Reply(ctx, resp, err); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "textDocument/documentLink": // req
		var params DocumentLinkParams
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		resp, err := h.server.DocumentLink(ctx, &params)
		if err := r.Reply(ctx, resp, err); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "documentLink/resolve": // req
		var params DocumentLink
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		resp, err := h.server.ResolveDocumentLink(ctx, &params)
		if err := r.Reply(ctx, resp, err); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "textDocument/formatting": // req
		var params DocumentFormattingParams
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		resp, err := h.server.Formatting(ctx, &params)
		if err := r.Reply(ctx, resp, err); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "textDocument/rangeFormatting": // req
		var params DocumentRangeFormattingParams
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		resp, err := h.server.RangeFormatting(ctx, &params)
		if err := r.Reply(ctx, resp, err); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "textDocument/onTypeFormatting": // req
		var params DocumentOnTypeFormattingParams
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		resp, err := h.server.OnTypeFormatting(ctx, &params)
		if err := r.Reply(ctx, resp, err); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "textDocument/rename": // req
		var params RenameParams
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		resp, err := h.server.Rename(ctx, &params)
		if err := r.Reply(ctx, resp, err); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "textDocument/prepareRename": // req
		var params PrepareRenameParams
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		resp, err := h.server.PrepareRename(ctx, &params)
		if err := r.Reply(ctx, resp, err); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "workspace/executeCommand": // req
		var params ExecuteCommandParams
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		resp, err := h.server.ExecuteCommand(ctx, &params)
		if err := r.Reply(ctx, resp, err); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "textDocument/prepareCallHierarchy": // req
		var params CallHierarchyPrepareParams
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		resp, err := h.server.PrepareCallHierarchy(ctx, &params)
		if err := r.Reply(ctx, resp, err); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "callHierarchy/incomingCalls": // req
		var params CallHierarchyIncomingCallsParams
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		resp, err := h.server.IncomingCalls(ctx, &params)
		if err := r.Reply(ctx, resp, err); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "callHierarchy/outgoingCalls": // req
		var params CallHierarchyOutgoingCallsParams
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		resp, err := h.server.OutgoingCalls(ctx, &params)
		if err := r.Reply(ctx, resp, err); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "textDocument/semanticTokens": // req
		var params SemanticTokensParams
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		resp, err := h.server.SemanticTokens(ctx, &params)
		if err := r.Reply(ctx, resp, err); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "textDocument/semanticTokens/edits": // req
		var params SemanticTokensEditsParams
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		resp, err := h.server.SemanticTokensEdits(ctx, &params)
		if err := r.Reply(ctx, resp, err); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	case "textDocument/semanticTokens/range": // req
		var params SemanticTokensRangeParams
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		resp, err := h.server.SemanticTokensRange(ctx, &params)
		if err := r.Reply(ctx, resp, err); err != nil {
			log.Error(ctx, "", err)
		}
		return true
	default:
		var params interface{}
		if err := json.Unmarshal(*r.Params, &params); err != nil {
			sendParseError(ctx, r, err)
			return true
		}
		resp, err := h.server.NonstandardRequest(ctx, r.Method, params)
		if err := r.Reply(ctx, resp, err); err != nil {
			log.Error(ctx, "", err)
		}
		return true

	}
}

type serverDispatcher struct {
	*jsonrpc2.Conn
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

func (s *serverDispatcher) Progress(ctx context.Context, params *ProgressParams) error {
	return s.Conn.Notify(ctx, "$/progress", params)
}

func (s *serverDispatcher) SetTraceNotification(ctx context.Context, params *SetTraceParams) error {
	return s.Conn.Notify(ctx, "$/setTraceNotification", params)
}

func (s *serverDispatcher) LogTraceNotification(ctx context.Context, params *LogTraceParams) error {
	return s.Conn.Notify(ctx, "$/logTraceNotification", params)
}
func (s *serverDispatcher) Implementation(ctx context.Context, params *ImplementationParams) (Definition /*Definition | DefinitionLink[] | null*/, error) {
	var result Definition /*Definition | DefinitionLink[] | null*/
	if err := s.Conn.Call(ctx, "textDocument/implementation", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) TypeDefinition(ctx context.Context, params *TypeDefinitionParams) (Definition /*Definition | DefinitionLink[] | null*/, error) {
	var result Definition /*Definition | DefinitionLink[] | null*/
	if err := s.Conn.Call(ctx, "textDocument/typeDefinition", params, &result); err != nil {
		return nil, err
	}
	return result, nil
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

func (s *serverDispatcher) FoldingRange(ctx context.Context, params *FoldingRangeParams) ([]FoldingRange /*FoldingRange[] | null*/, error) {
	var result []FoldingRange /*FoldingRange[] | null*/
	if err := s.Conn.Call(ctx, "textDocument/foldingRange", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) Declaration(ctx context.Context, params *DeclarationParams) (Declaration /*Declaration | DeclarationLink[] | null*/, error) {
	var result Declaration /*Declaration | DeclarationLink[] | null*/
	if err := s.Conn.Call(ctx, "textDocument/declaration", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) SelectionRange(ctx context.Context, params *SelectionRangeParams) ([]SelectionRange /*SelectionRange[] | null*/, error) {
	var result []SelectionRange /*SelectionRange[] | null*/
	if err := s.Conn.Call(ctx, "textDocument/selectionRange", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) WorkDoneProgressCreate(ctx context.Context, params *WorkDoneProgressCreateParams) error {
	return s.Conn.Call(ctx, "window/workDoneProgress/create", params, nil) // Call, not Notify
}

func (s *serverDispatcher) Initialize(ctx context.Context, params *ParamInitialize) (*InitializeResult, error) {
	var result InitializeResult
	if err := s.Conn.Call(ctx, "initialize", params, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

func (s *serverDispatcher) Shutdown(ctx context.Context) error {
	return s.Conn.Call(ctx, "shutdown", nil, nil)
}

func (s *serverDispatcher) WillSaveWaitUntil(ctx context.Context, params *WillSaveTextDocumentParams) ([]TextEdit /*TextEdit[] | null*/, error) {
	var result []TextEdit /*TextEdit[] | null*/
	if err := s.Conn.Call(ctx, "textDocument/willSaveWaitUntil", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) Completion(ctx context.Context, params *CompletionParams) (*CompletionList /*CompletionItem[] | CompletionList | null*/, error) {
	var result CompletionList /*CompletionItem[] | CompletionList | null*/
	if err := s.Conn.Call(ctx, "textDocument/completion", params, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

func (s *serverDispatcher) Resolve(ctx context.Context, params *CompletionItem) (*CompletionItem, error) {
	var result CompletionItem
	if err := s.Conn.Call(ctx, "completionItem/resolve", params, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

func (s *serverDispatcher) Hover(ctx context.Context, params *HoverParams) (*Hover /*Hover | null*/, error) {
	var result Hover /*Hover | null*/
	if err := s.Conn.Call(ctx, "textDocument/hover", params, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

func (s *serverDispatcher) SignatureHelp(ctx context.Context, params *SignatureHelpParams) (*SignatureHelp /*SignatureHelp | null*/, error) {
	var result SignatureHelp /*SignatureHelp | null*/
	if err := s.Conn.Call(ctx, "textDocument/signatureHelp", params, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

func (s *serverDispatcher) Definition(ctx context.Context, params *DefinitionParams) (Definition /*Definition | DefinitionLink[] | null*/, error) {
	var result Definition /*Definition | DefinitionLink[] | null*/
	if err := s.Conn.Call(ctx, "textDocument/definition", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) References(ctx context.Context, params *ReferenceParams) ([]Location /*Location[] | null*/, error) {
	var result []Location /*Location[] | null*/
	if err := s.Conn.Call(ctx, "textDocument/references", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) DocumentHighlight(ctx context.Context, params *DocumentHighlightParams) ([]DocumentHighlight /*DocumentHighlight[] | null*/, error) {
	var result []DocumentHighlight /*DocumentHighlight[] | null*/
	if err := s.Conn.Call(ctx, "textDocument/documentHighlight", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) DocumentSymbol(ctx context.Context, params *DocumentSymbolParams) ([]DocumentSymbol /*SymbolInformation[] | DocumentSymbol[] | null*/, error) {
	var result []DocumentSymbol /*SymbolInformation[] | DocumentSymbol[] | null*/
	if err := s.Conn.Call(ctx, "textDocument/documentSymbol", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) CodeAction(ctx context.Context, params *CodeActionParams) ([]CodeAction /*(Command | CodeAction)[] | null*/, error) {
	var result []CodeAction /*(Command | CodeAction)[] | null*/
	if err := s.Conn.Call(ctx, "textDocument/codeAction", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) Symbol(ctx context.Context, params *WorkspaceSymbolParams) ([]SymbolInformation /*SymbolInformation[] | null*/, error) {
	var result []SymbolInformation /*SymbolInformation[] | null*/
	if err := s.Conn.Call(ctx, "workspace/symbol", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) CodeLens(ctx context.Context, params *CodeLensParams) ([]CodeLens /*CodeLens[] | null*/, error) {
	var result []CodeLens /*CodeLens[] | null*/
	if err := s.Conn.Call(ctx, "textDocument/codeLens", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) ResolveCodeLens(ctx context.Context, params *CodeLens) (*CodeLens, error) {
	var result CodeLens
	if err := s.Conn.Call(ctx, "codeLens/resolve", params, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

func (s *serverDispatcher) DocumentLink(ctx context.Context, params *DocumentLinkParams) ([]DocumentLink /*DocumentLink[] | null*/, error) {
	var result []DocumentLink /*DocumentLink[] | null*/
	if err := s.Conn.Call(ctx, "textDocument/documentLink", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) ResolveDocumentLink(ctx context.Context, params *DocumentLink) (*DocumentLink, error) {
	var result DocumentLink
	if err := s.Conn.Call(ctx, "documentLink/resolve", params, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

func (s *serverDispatcher) Formatting(ctx context.Context, params *DocumentFormattingParams) ([]TextEdit /*TextEdit[] | null*/, error) {
	var result []TextEdit /*TextEdit[] | null*/
	if err := s.Conn.Call(ctx, "textDocument/formatting", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) RangeFormatting(ctx context.Context, params *DocumentRangeFormattingParams) ([]TextEdit /*TextEdit[] | null*/, error) {
	var result []TextEdit /*TextEdit[] | null*/
	if err := s.Conn.Call(ctx, "textDocument/rangeFormatting", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) OnTypeFormatting(ctx context.Context, params *DocumentOnTypeFormattingParams) ([]TextEdit /*TextEdit[] | null*/, error) {
	var result []TextEdit /*TextEdit[] | null*/
	if err := s.Conn.Call(ctx, "textDocument/onTypeFormatting", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) Rename(ctx context.Context, params *RenameParams) (*WorkspaceEdit /*WorkspaceEdit | null*/, error) {
	var result WorkspaceEdit /*WorkspaceEdit | null*/
	if err := s.Conn.Call(ctx, "textDocument/rename", params, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

func (s *serverDispatcher) PrepareRename(ctx context.Context, params *PrepareRenameParams) (interface{} /* Range | struct{;  Range Range`json:"range"`;  Placeholder string`json:"placeholder"`; } | nil*/, error) {
	var result interface{} /* Range | struct{;  Range Range`json:"range"`;  Placeholder string`json:"placeholder"`; } | nil*/
	if err := s.Conn.Call(ctx, "textDocument/prepareRename", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) ExecuteCommand(ctx context.Context, params *ExecuteCommandParams) (interface{} /*any | null*/, error) {
	var result interface{} /*any | null*/
	if err := s.Conn.Call(ctx, "workspace/executeCommand", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) PrepareCallHierarchy(ctx context.Context, params *CallHierarchyPrepareParams) ([]CallHierarchyItem /*CallHierarchyItem[] | null*/, error) {
	var result []CallHierarchyItem /*CallHierarchyItem[] | null*/
	if err := s.Conn.Call(ctx, "textDocument/prepareCallHierarchy", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) IncomingCalls(ctx context.Context, params *CallHierarchyIncomingCallsParams) ([]CallHierarchyIncomingCall /*CallHierarchyIncomingCall[] | null*/, error) {
	var result []CallHierarchyIncomingCall /*CallHierarchyIncomingCall[] | null*/
	if err := s.Conn.Call(ctx, "callHierarchy/incomingCalls", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) OutgoingCalls(ctx context.Context, params *CallHierarchyOutgoingCallsParams) ([]CallHierarchyOutgoingCall /*CallHierarchyOutgoingCall[] | null*/, error) {
	var result []CallHierarchyOutgoingCall /*CallHierarchyOutgoingCall[] | null*/
	if err := s.Conn.Call(ctx, "callHierarchy/outgoingCalls", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) SemanticTokens(ctx context.Context, params *SemanticTokensParams) (*SemanticTokens /*SemanticTokens | null*/, error) {
	var result SemanticTokens /*SemanticTokens | null*/
	if err := s.Conn.Call(ctx, "textDocument/semanticTokens", params, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

func (s *serverDispatcher) SemanticTokensEdits(ctx context.Context, params *SemanticTokensEditsParams) (interface{} /* SemanticTokens | SemanticTokensEdits | nil*/, error) {
	var result interface{} /* SemanticTokens | SemanticTokensEdits | nil*/
	if err := s.Conn.Call(ctx, "textDocument/semanticTokens/edits", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) SemanticTokensRange(ctx context.Context, params *SemanticTokensRangeParams) (*SemanticTokens /*SemanticTokens | null*/, error) {
	var result SemanticTokens /*SemanticTokens | null*/
	if err := s.Conn.Call(ctx, "textDocument/semanticTokens/range", params, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

func (s *serverDispatcher) NonstandardRequest(ctx context.Context, method string, params interface{}) (interface{}, error) {
	var result interface{}
	if err := s.Conn.Call(ctx, method, params, &result); err != nil {
		return nil, err
	}
	return result, nil
}
