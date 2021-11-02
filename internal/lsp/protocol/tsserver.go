// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protocol

// Package protocol contains data types and code for LSP jsonrpcs
// generated automatically from vscode-languageserver-node
// commit: f17727af04704c0e2ede73dfdbeb463156e94561
// last fetched Thu Feb 10 2022 14:34:11 GMT-0700 (Mountain Standard Time)

// Code generated (see typescript/README.md) DO NOT EDIT.

import (
	"context"
	"encoding/json"

	"golang.org/x/tools/internal/jsonrpc2"
	errors "golang.org/x/xerrors"
)

type Server interface {
	DidChangeWorkspaceFolders(context.Context, *DidChangeWorkspaceFoldersParams) error
	WorkDoneProgressCancel(context.Context, *WorkDoneProgressCancelParams) error
	DidCreateFiles(context.Context, *CreateFilesParams) error
	DidRenameFiles(context.Context, *RenameFilesParams) error
	DidDeleteFiles(context.Context, *DeleteFilesParams) error
	Initialized(context.Context, *InitializedParams) error
	Exit(context.Context) error
	DidChangeConfiguration(context.Context, *DidChangeConfigurationParams) error
	DidOpen(context.Context, *DidOpenTextDocumentParams) error
	DidChange(context.Context, *DidChangeTextDocumentParams) error
	DidClose(context.Context, *DidCloseTextDocumentParams) error
	DidSave(context.Context, *DidSaveTextDocumentParams) error
	WillSave(context.Context, *WillSaveTextDocumentParams) error
	DidChangeWatchedFiles(context.Context, *DidChangeWatchedFilesParams) error
	DidOpenNotebookDocument(context.Context, *DidOpenNotebookDocumentParams) error
	DidChangeNotebookDocument(context.Context, *DidChangeNotebookDocumentParams) error
	DidCloseNotebookDocument(context.Context, *DidCloseNotebookDocumentParams) error
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
	SemanticTokensFull(context.Context, *SemanticTokensParams) (*SemanticTokens /*SemanticTokens | null*/, error)
	SemanticTokensFullDelta(context.Context, *SemanticTokensDeltaParams) (interface{} /* SemanticTokens | SemanticTokensDelta | float64*/, error)
	SemanticTokensRange(context.Context, *SemanticTokensRangeParams) (*SemanticTokens /*SemanticTokens | null*/, error)
	SemanticTokensRefresh(context.Context) error
	LinkedEditingRange(context.Context, *LinkedEditingRangeParams) (*LinkedEditingRanges /*LinkedEditingRanges | null*/, error)
	WillCreateFiles(context.Context, *CreateFilesParams) (*WorkspaceEdit /*WorkspaceEdit | null*/, error)
	WillRenameFiles(context.Context, *RenameFilesParams) (*WorkspaceEdit /*WorkspaceEdit | null*/, error)
	WillDeleteFiles(context.Context, *DeleteFilesParams) (*WorkspaceEdit /*WorkspaceEdit | null*/, error)
	Moniker(context.Context, *MonikerParams) ([]Moniker /*Moniker[] | null*/, error)
	PrepareTypeHierarchy(context.Context, *TypeHierarchyPrepareParams) ([]TypeHierarchyItem /*TypeHierarchyItem[] | null*/, error)
	Supertypes(context.Context, *TypeHierarchySupertypesParams) ([]TypeHierarchyItem /*TypeHierarchyItem[] | null*/, error)
	Subtypes(context.Context, *TypeHierarchySubtypesParams) ([]TypeHierarchyItem /*TypeHierarchyItem[] | null*/, error)
	InlineValues(context.Context, *InlineValuesParams) ([]InlineValue /*InlineValue[] | null*/, error)
	InlineValuesRefresh(context.Context) error
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
	ResolveCodeAction(context.Context, *CodeAction) (*CodeAction, error)
	Symbol(context.Context, *WorkspaceSymbolParams) ([]SymbolInformation /*SymbolInformation[] | WorkspaceSymbol[] | null*/, error)
	ResolveWorkspaceSymbol(context.Context, *WorkspaceSymbol) (*WorkspaceSymbol, error)
	CodeLens(context.Context, *CodeLensParams) ([]CodeLens /*CodeLens[] | null*/, error)
	ResolveCodeLens(context.Context, *CodeLens) (*CodeLens, error)
	CodeLensRefresh(context.Context) error
	DocumentLink(context.Context, *DocumentLinkParams) ([]DocumentLink /*DocumentLink[] | null*/, error)
	ResolveDocumentLink(context.Context, *DocumentLink) (*DocumentLink, error)
	Formatting(context.Context, *DocumentFormattingParams) ([]TextEdit /*TextEdit[] | null*/, error)
	RangeFormatting(context.Context, *DocumentRangeFormattingParams) ([]TextEdit /*TextEdit[] | null*/, error)
	OnTypeFormatting(context.Context, *DocumentOnTypeFormattingParams) ([]TextEdit /*TextEdit[] | null*/, error)
	Rename(context.Context, *RenameParams) (*WorkspaceEdit /*WorkspaceEdit | null*/, error)
	PrepareRename(context.Context, *PrepareRenameParams) (*PrepareRename2Gn /*Range | { range: Range, placeholder: string } | { defaultBehavior: boolean } | null*/, error)
	ExecuteCommand(context.Context, *ExecuteCommandParams) (interface{} /* LSPAny | void | float64*/, error)
	Diagnostic(context.Context, *string) (*string, error)
	DiagnosticWorkspace(context.Context, *WorkspaceDiagnosticParams) (*WorkspaceDiagnosticReport, error)
	DiagnosticRefresh(context.Context) error
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
	case "workspace/didCreateFiles": // notif
		var params CreateFilesParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := server.DidCreateFiles(ctx, &params)
		return true, reply(ctx, nil, err)
	case "workspace/didRenameFiles": // notif
		var params RenameFilesParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := server.DidRenameFiles(ctx, &params)
		return true, reply(ctx, nil, err)
	case "workspace/didDeleteFiles": // notif
		var params DeleteFilesParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := server.DidDeleteFiles(ctx, &params)
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
	case "notebookDocument/didOpen": // notif
		var params DidOpenNotebookDocumentParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := server.DidOpenNotebookDocument(ctx, &params)
		return true, reply(ctx, nil, err)
	case "notebookDocument/didChange": // notif
		var params DidChangeNotebookDocumentParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := server.DidChangeNotebookDocument(ctx, &params)
		return true, reply(ctx, nil, err)
	case "notebookDocument/didClose": // notif
		var params DidCloseNotebookDocumentParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := server.DidCloseNotebookDocument(ctx, &params)
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
	case "workspace/semanticTokens/refresh": // req
		if len(r.Params()) > 0 {
			return true, reply(ctx, nil, errors.Errorf("%w: expected no params", jsonrpc2.ErrInvalidParams))
		}
		err := server.SemanticTokensRefresh(ctx)
		return true, reply(ctx, nil, err)
	case "textDocument/linkedEditingRange": // req
		var params LinkedEditingRangeParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.LinkedEditingRange(ctx, &params)
		return true, reply(ctx, resp, err)
	case "workspace/willCreateFiles": // req
		var params CreateFilesParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.WillCreateFiles(ctx, &params)
		return true, reply(ctx, resp, err)
	case "workspace/willRenameFiles": // req
		var params RenameFilesParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.WillRenameFiles(ctx, &params)
		return true, reply(ctx, resp, err)
	case "workspace/willDeleteFiles": // req
		var params DeleteFilesParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.WillDeleteFiles(ctx, &params)
		return true, reply(ctx, resp, err)
	case "textDocument/moniker": // req
		var params MonikerParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.Moniker(ctx, &params)
		return true, reply(ctx, resp, err)
	case "textDocument/prepareTypeHierarchy": // req
		var params TypeHierarchyPrepareParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.PrepareTypeHierarchy(ctx, &params)
		return true, reply(ctx, resp, err)
	case "typeHierarchy/supertypes": // req
		var params TypeHierarchySupertypesParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.Supertypes(ctx, &params)
		return true, reply(ctx, resp, err)
	case "typeHierarchy/subtypes": // req
		var params TypeHierarchySubtypesParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.Subtypes(ctx, &params)
		return true, reply(ctx, resp, err)
	case "textDocument/inlineValues": // req
		var params InlineValuesParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.InlineValues(ctx, &params)
		return true, reply(ctx, resp, err)
	case "workspace/inlineValues/refresh": // req
		if len(r.Params()) > 0 {
			return true, reply(ctx, nil, errors.Errorf("%w: expected no params", jsonrpc2.ErrInvalidParams))
		}
		err := server.InlineValuesRefresh(ctx)
		return true, reply(ctx, nil, err)
	case "initialize": // req
		var params ParamInitialize
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			if _, ok := err.(*json.UnmarshalTypeError); !ok {
				return true, sendParseError(ctx, reply, err)
			}
		}
		resp, err := server.Initialize(ctx, &params)
		return true, reply(ctx, resp, err)
	case "shutdown": // req
		if len(r.Params()) > 0 {
			return true, reply(ctx, nil, errors.Errorf("%w: expected no params", jsonrpc2.ErrInvalidParams))
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
	case "codeAction/resolve": // req
		var params CodeAction
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.ResolveCodeAction(ctx, &params)
		return true, reply(ctx, resp, err)
	case "workspace/symbol": // req
		var params WorkspaceSymbolParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.Symbol(ctx, &params)
		return true, reply(ctx, resp, err)
	case "workspaceSymbol/resolve": // req
		var params WorkspaceSymbol
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.ResolveWorkspaceSymbol(ctx, &params)
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
	case "workspace/codeLens/refresh": // req
		if len(r.Params()) > 0 {
			return true, reply(ctx, nil, errors.Errorf("%w: expected no params", jsonrpc2.ErrInvalidParams))
		}
		err := server.CodeLensRefresh(ctx)
		return true, reply(ctx, nil, err)
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
	case "textDocument/diagnostic": // req
		var params string
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.Diagnostic(ctx, &params)
		return true, reply(ctx, resp, err)
	case "workspace/diagnostic": // req
		var params WorkspaceDiagnosticParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.DiagnosticWorkspace(ctx, &params)
		return true, reply(ctx, resp, err)
	case "workspace/diagnostic/refresh": // req
		if len(r.Params()) > 0 {
			return true, reply(ctx, nil, errors.Errorf("%w: expected no params", jsonrpc2.ErrInvalidParams))
		}
		err := server.DiagnosticRefresh(ctx)
		return true, reply(ctx, nil, err)

	default:
		return false, nil
	}
}

func (s *serverDispatcher) DidChangeWorkspaceFolders(ctx context.Context, params *DidChangeWorkspaceFoldersParams) error {
	return s.sender.Notify(ctx, "workspace/didChangeWorkspaceFolders", params)
}

func (s *serverDispatcher) WorkDoneProgressCancel(ctx context.Context, params *WorkDoneProgressCancelParams) error {
	return s.sender.Notify(ctx, "window/workDoneProgress/cancel", params)
}

func (s *serverDispatcher) DidCreateFiles(ctx context.Context, params *CreateFilesParams) error {
	return s.sender.Notify(ctx, "workspace/didCreateFiles", params)
}

func (s *serverDispatcher) DidRenameFiles(ctx context.Context, params *RenameFilesParams) error {
	return s.sender.Notify(ctx, "workspace/didRenameFiles", params)
}

func (s *serverDispatcher) DidDeleteFiles(ctx context.Context, params *DeleteFilesParams) error {
	return s.sender.Notify(ctx, "workspace/didDeleteFiles", params)
}

func (s *serverDispatcher) Initialized(ctx context.Context, params *InitializedParams) error {
	return s.sender.Notify(ctx, "initialized", params)
}

func (s *serverDispatcher) Exit(ctx context.Context) error {
	return s.sender.Notify(ctx, "exit", nil)
}

func (s *serverDispatcher) DidChangeConfiguration(ctx context.Context, params *DidChangeConfigurationParams) error {
	return s.sender.Notify(ctx, "workspace/didChangeConfiguration", params)
}

func (s *serverDispatcher) DidOpen(ctx context.Context, params *DidOpenTextDocumentParams) error {
	return s.sender.Notify(ctx, "textDocument/didOpen", params)
}

func (s *serverDispatcher) DidChange(ctx context.Context, params *DidChangeTextDocumentParams) error {
	return s.sender.Notify(ctx, "textDocument/didChange", params)
}

func (s *serverDispatcher) DidClose(ctx context.Context, params *DidCloseTextDocumentParams) error {
	return s.sender.Notify(ctx, "textDocument/didClose", params)
}

func (s *serverDispatcher) DidSave(ctx context.Context, params *DidSaveTextDocumentParams) error {
	return s.sender.Notify(ctx, "textDocument/didSave", params)
}

func (s *serverDispatcher) WillSave(ctx context.Context, params *WillSaveTextDocumentParams) error {
	return s.sender.Notify(ctx, "textDocument/willSave", params)
}

func (s *serverDispatcher) DidChangeWatchedFiles(ctx context.Context, params *DidChangeWatchedFilesParams) error {
	return s.sender.Notify(ctx, "workspace/didChangeWatchedFiles", params)
}

func (s *serverDispatcher) DidOpenNotebookDocument(ctx context.Context, params *DidOpenNotebookDocumentParams) error {
	return s.sender.Notify(ctx, "notebookDocument/didOpen", params)
}

func (s *serverDispatcher) DidChangeNotebookDocument(ctx context.Context, params *DidChangeNotebookDocumentParams) error {
	return s.sender.Notify(ctx, "notebookDocument/didChange", params)
}

func (s *serverDispatcher) DidCloseNotebookDocument(ctx context.Context, params *DidCloseNotebookDocumentParams) error {
	return s.sender.Notify(ctx, "notebookDocument/didClose", params)
}

func (s *serverDispatcher) SetTrace(ctx context.Context, params *SetTraceParams) error {
	return s.sender.Notify(ctx, "$/setTrace", params)
}

func (s *serverDispatcher) LogTrace(ctx context.Context, params *LogTraceParams) error {
	return s.sender.Notify(ctx, "$/logTrace", params)
}
func (s *serverDispatcher) Implementation(ctx context.Context, params *ImplementationParams) (Definition /*Definition | DefinitionLink[] | null*/, error) {
	var result Definition /*Definition | DefinitionLink[] | null*/
	if err := s.sender.Call(ctx, "textDocument/implementation", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) TypeDefinition(ctx context.Context, params *TypeDefinitionParams) (Definition /*Definition | DefinitionLink[] | null*/, error) {
	var result Definition /*Definition | DefinitionLink[] | null*/
	if err := s.sender.Call(ctx, "textDocument/typeDefinition", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) DocumentColor(ctx context.Context, params *DocumentColorParams) ([]ColorInformation, error) {
	var result []ColorInformation
	if err := s.sender.Call(ctx, "textDocument/documentColor", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) ColorPresentation(ctx context.Context, params *ColorPresentationParams) ([]ColorPresentation, error) {
	var result []ColorPresentation
	if err := s.sender.Call(ctx, "textDocument/colorPresentation", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) FoldingRange(ctx context.Context, params *FoldingRangeParams) ([]FoldingRange /*FoldingRange[] | null*/, error) {
	var result []FoldingRange /*FoldingRange[] | null*/
	if err := s.sender.Call(ctx, "textDocument/foldingRange", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) Declaration(ctx context.Context, params *DeclarationParams) (Declaration /*Declaration | DeclarationLink[] | null*/, error) {
	var result Declaration /*Declaration | DeclarationLink[] | null*/
	if err := s.sender.Call(ctx, "textDocument/declaration", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) SelectionRange(ctx context.Context, params *SelectionRangeParams) ([]SelectionRange /*SelectionRange[] | null*/, error) {
	var result []SelectionRange /*SelectionRange[] | null*/
	if err := s.sender.Call(ctx, "textDocument/selectionRange", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) PrepareCallHierarchy(ctx context.Context, params *CallHierarchyPrepareParams) ([]CallHierarchyItem /*CallHierarchyItem[] | null*/, error) {
	var result []CallHierarchyItem /*CallHierarchyItem[] | null*/
	if err := s.sender.Call(ctx, "textDocument/prepareCallHierarchy", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) IncomingCalls(ctx context.Context, params *CallHierarchyIncomingCallsParams) ([]CallHierarchyIncomingCall /*CallHierarchyIncomingCall[] | null*/, error) {
	var result []CallHierarchyIncomingCall /*CallHierarchyIncomingCall[] | null*/
	if err := s.sender.Call(ctx, "callHierarchy/incomingCalls", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) OutgoingCalls(ctx context.Context, params *CallHierarchyOutgoingCallsParams) ([]CallHierarchyOutgoingCall /*CallHierarchyOutgoingCall[] | null*/, error) {
	var result []CallHierarchyOutgoingCall /*CallHierarchyOutgoingCall[] | null*/
	if err := s.sender.Call(ctx, "callHierarchy/outgoingCalls", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) SemanticTokensFull(ctx context.Context, params *SemanticTokensParams) (*SemanticTokens /*SemanticTokens | null*/, error) {
	var result *SemanticTokens /*SemanticTokens | null*/
	if err := s.sender.Call(ctx, "textDocument/semanticTokens/full", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) SemanticTokensFullDelta(ctx context.Context, params *SemanticTokensDeltaParams) (interface{} /* SemanticTokens | SemanticTokensDelta | float64*/, error) {
	var result interface{} /* SemanticTokens | SemanticTokensDelta | float64*/
	if err := s.sender.Call(ctx, "textDocument/semanticTokens/full/delta", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) SemanticTokensRange(ctx context.Context, params *SemanticTokensRangeParams) (*SemanticTokens /*SemanticTokens | null*/, error) {
	var result *SemanticTokens /*SemanticTokens | null*/
	if err := s.sender.Call(ctx, "textDocument/semanticTokens/range", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) SemanticTokensRefresh(ctx context.Context) error {
	return s.sender.Call(ctx, "workspace/semanticTokens/refresh", nil, nil)
}

func (s *serverDispatcher) LinkedEditingRange(ctx context.Context, params *LinkedEditingRangeParams) (*LinkedEditingRanges /*LinkedEditingRanges | null*/, error) {
	var result *LinkedEditingRanges /*LinkedEditingRanges | null*/
	if err := s.sender.Call(ctx, "textDocument/linkedEditingRange", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) WillCreateFiles(ctx context.Context, params *CreateFilesParams) (*WorkspaceEdit /*WorkspaceEdit | null*/, error) {
	var result *WorkspaceEdit /*WorkspaceEdit | null*/
	if err := s.sender.Call(ctx, "workspace/willCreateFiles", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) WillRenameFiles(ctx context.Context, params *RenameFilesParams) (*WorkspaceEdit /*WorkspaceEdit | null*/, error) {
	var result *WorkspaceEdit /*WorkspaceEdit | null*/
	if err := s.sender.Call(ctx, "workspace/willRenameFiles", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) WillDeleteFiles(ctx context.Context, params *DeleteFilesParams) (*WorkspaceEdit /*WorkspaceEdit | null*/, error) {
	var result *WorkspaceEdit /*WorkspaceEdit | null*/
	if err := s.sender.Call(ctx, "workspace/willDeleteFiles", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) Moniker(ctx context.Context, params *MonikerParams) ([]Moniker /*Moniker[] | null*/, error) {
	var result []Moniker /*Moniker[] | null*/
	if err := s.sender.Call(ctx, "textDocument/moniker", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) PrepareTypeHierarchy(ctx context.Context, params *TypeHierarchyPrepareParams) ([]TypeHierarchyItem /*TypeHierarchyItem[] | null*/, error) {
	var result []TypeHierarchyItem /*TypeHierarchyItem[] | null*/
	if err := s.sender.Call(ctx, "textDocument/prepareTypeHierarchy", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) Supertypes(ctx context.Context, params *TypeHierarchySupertypesParams) ([]TypeHierarchyItem /*TypeHierarchyItem[] | null*/, error) {
	var result []TypeHierarchyItem /*TypeHierarchyItem[] | null*/
	if err := s.sender.Call(ctx, "typeHierarchy/supertypes", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) Subtypes(ctx context.Context, params *TypeHierarchySubtypesParams) ([]TypeHierarchyItem /*TypeHierarchyItem[] | null*/, error) {
	var result []TypeHierarchyItem /*TypeHierarchyItem[] | null*/
	if err := s.sender.Call(ctx, "typeHierarchy/subtypes", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) InlineValues(ctx context.Context, params *InlineValuesParams) ([]InlineValue /*InlineValue[] | null*/, error) {
	var result []InlineValue /*InlineValue[] | null*/
	if err := s.sender.Call(ctx, "textDocument/inlineValues", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) InlineValuesRefresh(ctx context.Context) error {
	return s.sender.Call(ctx, "workspace/inlineValues/refresh", nil, nil)
}

func (s *serverDispatcher) Initialize(ctx context.Context, params *ParamInitialize) (*InitializeResult, error) {
	var result *InitializeResult
	if err := s.sender.Call(ctx, "initialize", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) Shutdown(ctx context.Context) error {
	return s.sender.Call(ctx, "shutdown", nil, nil)
}

func (s *serverDispatcher) WillSaveWaitUntil(ctx context.Context, params *WillSaveTextDocumentParams) ([]TextEdit /*TextEdit[] | null*/, error) {
	var result []TextEdit /*TextEdit[] | null*/
	if err := s.sender.Call(ctx, "textDocument/willSaveWaitUntil", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) Completion(ctx context.Context, params *CompletionParams) (*CompletionList /*CompletionItem[] | CompletionList | null*/, error) {
	var result *CompletionList /*CompletionItem[] | CompletionList | null*/
	if err := s.sender.Call(ctx, "textDocument/completion", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) Resolve(ctx context.Context, params *CompletionItem) (*CompletionItem, error) {
	var result *CompletionItem
	if err := s.sender.Call(ctx, "completionItem/resolve", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) Hover(ctx context.Context, params *HoverParams) (*Hover /*Hover | null*/, error) {
	var result *Hover /*Hover | null*/
	if err := s.sender.Call(ctx, "textDocument/hover", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) SignatureHelp(ctx context.Context, params *SignatureHelpParams) (*SignatureHelp /*SignatureHelp | null*/, error) {
	var result *SignatureHelp /*SignatureHelp | null*/
	if err := s.sender.Call(ctx, "textDocument/signatureHelp", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) Definition(ctx context.Context, params *DefinitionParams) (Definition /*Definition | DefinitionLink[] | null*/, error) {
	var result Definition /*Definition | DefinitionLink[] | null*/
	if err := s.sender.Call(ctx, "textDocument/definition", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) References(ctx context.Context, params *ReferenceParams) ([]Location /*Location[] | null*/, error) {
	var result []Location /*Location[] | null*/
	if err := s.sender.Call(ctx, "textDocument/references", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) DocumentHighlight(ctx context.Context, params *DocumentHighlightParams) ([]DocumentHighlight /*DocumentHighlight[] | null*/, error) {
	var result []DocumentHighlight /*DocumentHighlight[] | null*/
	if err := s.sender.Call(ctx, "textDocument/documentHighlight", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) DocumentSymbol(ctx context.Context, params *DocumentSymbolParams) ([]interface{} /*SymbolInformation[] | DocumentSymbol[] | null*/, error) {
	var result []interface{} /*SymbolInformation[] | DocumentSymbol[] | null*/
	if err := s.sender.Call(ctx, "textDocument/documentSymbol", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) CodeAction(ctx context.Context, params *CodeActionParams) ([]CodeAction /*(Command | CodeAction)[] | null*/, error) {
	var result []CodeAction /*(Command | CodeAction)[] | null*/
	if err := s.sender.Call(ctx, "textDocument/codeAction", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) ResolveCodeAction(ctx context.Context, params *CodeAction) (*CodeAction, error) {
	var result *CodeAction
	if err := s.sender.Call(ctx, "codeAction/resolve", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) Symbol(ctx context.Context, params *WorkspaceSymbolParams) ([]SymbolInformation /*SymbolInformation[] | WorkspaceSymbol[] | null*/, error) {
	var result []SymbolInformation /*SymbolInformation[] | WorkspaceSymbol[] | null*/
	if err := s.sender.Call(ctx, "workspace/symbol", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) ResolveWorkspaceSymbol(ctx context.Context, params *WorkspaceSymbol) (*WorkspaceSymbol, error) {
	var result *WorkspaceSymbol
	if err := s.sender.Call(ctx, "workspaceSymbol/resolve", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) CodeLens(ctx context.Context, params *CodeLensParams) ([]CodeLens /*CodeLens[] | null*/, error) {
	var result []CodeLens /*CodeLens[] | null*/
	if err := s.sender.Call(ctx, "textDocument/codeLens", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) ResolveCodeLens(ctx context.Context, params *CodeLens) (*CodeLens, error) {
	var result *CodeLens
	if err := s.sender.Call(ctx, "codeLens/resolve", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) CodeLensRefresh(ctx context.Context) error {
	return s.sender.Call(ctx, "workspace/codeLens/refresh", nil, nil)
}

func (s *serverDispatcher) DocumentLink(ctx context.Context, params *DocumentLinkParams) ([]DocumentLink /*DocumentLink[] | null*/, error) {
	var result []DocumentLink /*DocumentLink[] | null*/
	if err := s.sender.Call(ctx, "textDocument/documentLink", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) ResolveDocumentLink(ctx context.Context, params *DocumentLink) (*DocumentLink, error) {
	var result *DocumentLink
	if err := s.sender.Call(ctx, "documentLink/resolve", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) Formatting(ctx context.Context, params *DocumentFormattingParams) ([]TextEdit /*TextEdit[] | null*/, error) {
	var result []TextEdit /*TextEdit[] | null*/
	if err := s.sender.Call(ctx, "textDocument/formatting", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) RangeFormatting(ctx context.Context, params *DocumentRangeFormattingParams) ([]TextEdit /*TextEdit[] | null*/, error) {
	var result []TextEdit /*TextEdit[] | null*/
	if err := s.sender.Call(ctx, "textDocument/rangeFormatting", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) OnTypeFormatting(ctx context.Context, params *DocumentOnTypeFormattingParams) ([]TextEdit /*TextEdit[] | null*/, error) {
	var result []TextEdit /*TextEdit[] | null*/
	if err := s.sender.Call(ctx, "textDocument/onTypeFormatting", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) Rename(ctx context.Context, params *RenameParams) (*WorkspaceEdit /*WorkspaceEdit | null*/, error) {
	var result *WorkspaceEdit /*WorkspaceEdit | null*/
	if err := s.sender.Call(ctx, "textDocument/rename", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) PrepareRename(ctx context.Context, params *PrepareRenameParams) (*PrepareRename2Gn /*Range | { range: Range, placeholder: string } | { defaultBehavior: boolean } | null*/, error) {
	var result *PrepareRename2Gn /*Range | { range: Range, placeholder: string } | { defaultBehavior: boolean } | null*/
	if err := s.sender.Call(ctx, "textDocument/prepareRename", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) ExecuteCommand(ctx context.Context, params *ExecuteCommandParams) (interface{} /* LSPAny | void | float64*/, error) {
	var result interface{} /* LSPAny | void | float64*/
	if err := s.sender.Call(ctx, "workspace/executeCommand", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) Diagnostic(ctx context.Context, params *string) (*string, error) {
	var result *string
	if err := s.sender.Call(ctx, "textDocument/diagnostic", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) DiagnosticWorkspace(ctx context.Context, params *WorkspaceDiagnosticParams) (*WorkspaceDiagnosticReport, error) {
	var result *WorkspaceDiagnosticReport
	if err := s.sender.Call(ctx, "workspace/diagnostic", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (s *serverDispatcher) DiagnosticRefresh(ctx context.Context) error {
	return s.sender.Call(ctx, "workspace/diagnostic/refresh", nil, nil)
}

func (s *serverDispatcher) NonstandardRequest(ctx context.Context, method string, params interface{}) (interface{}, error) {
	var result interface{}
	if err := s.sender.Call(ctx, method, params, &result); err != nil {
		return nil, err
	}
	return result, nil
}
