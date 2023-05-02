// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Code generated for LSP. DO NOT EDIT.

package protocol

// Code generated from protocol/metaModel.json at ref release/protocol/3.17.4-next.0 (hash 5c6ec4f537f304aa1ad645b5fd2bbb757fc40ed1).
// https://github.com/microsoft/vscode-languageserver-node/blob/release/protocol/3.17.4-next.0/protocol/metaModel.json
// LSP metaData.version = 3.17.0.

import (
	"context"
	"encoding/json"

	"golang.org/x/tools/internal/jsonrpc2"
)

type Server interface {
	Progress(context.Context, *ProgressParams) error                                                       // $/progress
	SetTrace(context.Context, *SetTraceParams) error                                                       // $/setTrace
	IncomingCalls(context.Context, *CallHierarchyIncomingCallsParams) ([]CallHierarchyIncomingCall, error) // callHierarchy/incomingCalls
	OutgoingCalls(context.Context, *CallHierarchyOutgoingCallsParams) ([]CallHierarchyOutgoingCall, error) // callHierarchy/outgoingCalls
	ResolveCodeAction(context.Context, *CodeAction) (*CodeAction, error)                                   // codeAction/resolve
	ResolveCodeLens(context.Context, *CodeLens) (*CodeLens, error)                                         // codeLens/resolve
	ResolveCompletionItem(context.Context, *CompletionItem) (*CompletionItem, error)                       // completionItem/resolve
	ResolveDocumentLink(context.Context, *DocumentLink) (*DocumentLink, error)                             // documentLink/resolve
	Exit(context.Context) error                                                                            // exit
	Initialize(context.Context, *ParamInitialize) (*InitializeResult, error)                               // initialize
	Initialized(context.Context, *InitializedParams) error                                                 // initialized
	Resolve(context.Context, *InlayHint) (*InlayHint, error)                                               // inlayHint/resolve
	DidChangeNotebookDocument(context.Context, *DidChangeNotebookDocumentParams) error                     // notebookDocument/didChange
	DidCloseNotebookDocument(context.Context, *DidCloseNotebookDocumentParams) error                       // notebookDocument/didClose
	DidOpenNotebookDocument(context.Context, *DidOpenNotebookDocumentParams) error                         // notebookDocument/didOpen
	DidSaveNotebookDocument(context.Context, *DidSaveNotebookDocumentParams) error                         // notebookDocument/didSave
	Shutdown(context.Context) error                                                                        // shutdown
	CodeAction(context.Context, *CodeActionParams) ([]CodeAction, error)                                   // textDocument/codeAction
	CodeLens(context.Context, *CodeLensParams) ([]CodeLens, error)                                         // textDocument/codeLens
	ColorPresentation(context.Context, *ColorPresentationParams) ([]ColorPresentation, error)              // textDocument/colorPresentation
	Completion(context.Context, *CompletionParams) (*CompletionList, error)                                // textDocument/completion
	Declaration(context.Context, *DeclarationParams) (*Or_textDocument_declaration, error)                 // textDocument/declaration
	Definition(context.Context, *DefinitionParams) ([]Location, error)                                     // textDocument/definition
	Diagnostic(context.Context, *string) (*string, error)                                                  // textDocument/diagnostic
	DidChange(context.Context, *DidChangeTextDocumentParams) error                                         // textDocument/didChange
	DidClose(context.Context, *DidCloseTextDocumentParams) error                                           // textDocument/didClose
	DidOpen(context.Context, *DidOpenTextDocumentParams) error                                             // textDocument/didOpen
	DidSave(context.Context, *DidSaveTextDocumentParams) error                                             // textDocument/didSave
	DocumentColor(context.Context, *DocumentColorParams) ([]ColorInformation, error)                       // textDocument/documentColor
	DocumentHighlight(context.Context, *DocumentHighlightParams) ([]DocumentHighlight, error)              // textDocument/documentHighlight
	DocumentLink(context.Context, *DocumentLinkParams) ([]DocumentLink, error)                             // textDocument/documentLink
	DocumentSymbol(context.Context, *DocumentSymbolParams) ([]interface{}, error)                          // textDocument/documentSymbol
	FoldingRange(context.Context, *FoldingRangeParams) ([]FoldingRange, error)                             // textDocument/foldingRange
	Formatting(context.Context, *DocumentFormattingParams) ([]TextEdit, error)                             // textDocument/formatting
	Hover(context.Context, *HoverParams) (*Hover, error)                                                   // textDocument/hover
	Implementation(context.Context, *ImplementationParams) ([]Location, error)                             // textDocument/implementation
	InlayHint(context.Context, *InlayHintParams) ([]InlayHint, error)                                      // textDocument/inlayHint
	InlineValue(context.Context, *InlineValueParams) ([]InlineValue, error)                                // textDocument/inlineValue
	LinkedEditingRange(context.Context, *LinkedEditingRangeParams) (*LinkedEditingRanges, error)           // textDocument/linkedEditingRange
	Moniker(context.Context, *MonikerParams) ([]Moniker, error)                                            // textDocument/moniker
	OnTypeFormatting(context.Context, *DocumentOnTypeFormattingParams) ([]TextEdit, error)                 // textDocument/onTypeFormatting
	PrepareCallHierarchy(context.Context, *CallHierarchyPrepareParams) ([]CallHierarchyItem, error)        // textDocument/prepareCallHierarchy
	PrepareRename(context.Context, *PrepareRenameParams) (*PrepareRename2Gn, error)                        // textDocument/prepareRename
	PrepareTypeHierarchy(context.Context, *TypeHierarchyPrepareParams) ([]TypeHierarchyItem, error)        // textDocument/prepareTypeHierarchy
	RangeFormatting(context.Context, *DocumentRangeFormattingParams) ([]TextEdit, error)                   // textDocument/rangeFormatting
	References(context.Context, *ReferenceParams) ([]Location, error)                                      // textDocument/references
	Rename(context.Context, *RenameParams) (*WorkspaceEdit, error)                                         // textDocument/rename
	SelectionRange(context.Context, *SelectionRangeParams) ([]SelectionRange, error)                       // textDocument/selectionRange
	SemanticTokensFull(context.Context, *SemanticTokensParams) (*SemanticTokens, error)                    // textDocument/semanticTokens/full
	SemanticTokensFullDelta(context.Context, *SemanticTokensDeltaParams) (interface{}, error)              // textDocument/semanticTokens/full/delta
	SemanticTokensRange(context.Context, *SemanticTokensRangeParams) (*SemanticTokens, error)              // textDocument/semanticTokens/range
	SignatureHelp(context.Context, *SignatureHelpParams) (*SignatureHelp, error)                           // textDocument/signatureHelp
	TypeDefinition(context.Context, *TypeDefinitionParams) ([]Location, error)                             // textDocument/typeDefinition
	WillSave(context.Context, *WillSaveTextDocumentParams) error                                           // textDocument/willSave
	WillSaveWaitUntil(context.Context, *WillSaveTextDocumentParams) ([]TextEdit, error)                    // textDocument/willSaveWaitUntil
	Subtypes(context.Context, *TypeHierarchySubtypesParams) ([]TypeHierarchyItem, error)                   // typeHierarchy/subtypes
	Supertypes(context.Context, *TypeHierarchySupertypesParams) ([]TypeHierarchyItem, error)               // typeHierarchy/supertypes
	WorkDoneProgressCancel(context.Context, *WorkDoneProgressCancelParams) error                           // window/workDoneProgress/cancel
	DiagnosticWorkspace(context.Context, *WorkspaceDiagnosticParams) (*WorkspaceDiagnosticReport, error)   // workspace/diagnostic
	DidChangeConfiguration(context.Context, *DidChangeConfigurationParams) error                           // workspace/didChangeConfiguration
	DidChangeWatchedFiles(context.Context, *DidChangeWatchedFilesParams) error                             // workspace/didChangeWatchedFiles
	DidChangeWorkspaceFolders(context.Context, *DidChangeWorkspaceFoldersParams) error                     // workspace/didChangeWorkspaceFolders
	DidCreateFiles(context.Context, *CreateFilesParams) error                                              // workspace/didCreateFiles
	DidDeleteFiles(context.Context, *DeleteFilesParams) error                                              // workspace/didDeleteFiles
	DidRenameFiles(context.Context, *RenameFilesParams) error                                              // workspace/didRenameFiles
	ExecuteCommand(context.Context, *ExecuteCommandParams) (interface{}, error)                            // workspace/executeCommand
	Symbol(context.Context, *WorkspaceSymbolParams) ([]SymbolInformation, error)                           // workspace/symbol
	WillCreateFiles(context.Context, *CreateFilesParams) (*WorkspaceEdit, error)                           // workspace/willCreateFiles
	WillDeleteFiles(context.Context, *DeleteFilesParams) (*WorkspaceEdit, error)                           // workspace/willDeleteFiles
	WillRenameFiles(context.Context, *RenameFilesParams) (*WorkspaceEdit, error)                           // workspace/willRenameFiles
	ResolveWorkspaceSymbol(context.Context, *WorkspaceSymbol) (*WorkspaceSymbol, error)                    // workspaceSymbol/resolve
	NonstandardRequest(ctx context.Context, method string, params interface{}) (interface{}, error)
}

func serverDispatch(ctx context.Context, server Server, reply jsonrpc2.Replier, r jsonrpc2.Request) (bool, error) {
	switch r.Method() {
	case "$/progress":
		var params ProgressParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := server.Progress(ctx, &params)
		return true, reply(ctx, nil, err)
	case "$/setTrace":
		var params SetTraceParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := server.SetTrace(ctx, &params)
		return true, reply(ctx, nil, err)
	case "callHierarchy/incomingCalls":
		var params CallHierarchyIncomingCallsParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.IncomingCalls(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "callHierarchy/outgoingCalls":
		var params CallHierarchyOutgoingCallsParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.OutgoingCalls(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "codeAction/resolve":
		var params CodeAction
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.ResolveCodeAction(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "codeLens/resolve":
		var params CodeLens
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.ResolveCodeLens(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "completionItem/resolve":
		var params CompletionItem
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.ResolveCompletionItem(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "documentLink/resolve":
		var params DocumentLink
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.ResolveDocumentLink(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "exit":
		err := server.Exit(ctx)
		return true, reply(ctx, nil, err)
	case "initialize":
		var params ParamInitialize
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.Initialize(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "initialized":
		var params InitializedParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := server.Initialized(ctx, &params)
		return true, reply(ctx, nil, err)
	case "inlayHint/resolve":
		var params InlayHint
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.Resolve(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "notebookDocument/didChange":
		var params DidChangeNotebookDocumentParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := server.DidChangeNotebookDocument(ctx, &params)
		return true, reply(ctx, nil, err)
	case "notebookDocument/didClose":
		var params DidCloseNotebookDocumentParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := server.DidCloseNotebookDocument(ctx, &params)
		return true, reply(ctx, nil, err)
	case "notebookDocument/didOpen":
		var params DidOpenNotebookDocumentParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := server.DidOpenNotebookDocument(ctx, &params)
		return true, reply(ctx, nil, err)
	case "notebookDocument/didSave":
		var params DidSaveNotebookDocumentParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := server.DidSaveNotebookDocument(ctx, &params)
		return true, reply(ctx, nil, err)
	case "shutdown":
		err := server.Shutdown(ctx)
		return true, reply(ctx, nil, err)
	case "textDocument/codeAction":
		var params CodeActionParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.CodeAction(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "textDocument/codeLens":
		var params CodeLensParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.CodeLens(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "textDocument/colorPresentation":
		var params ColorPresentationParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.ColorPresentation(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "textDocument/completion":
		var params CompletionParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.Completion(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "textDocument/declaration":
		var params DeclarationParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.Declaration(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "textDocument/definition":
		var params DefinitionParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.Definition(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "textDocument/diagnostic":
		var params string
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.Diagnostic(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "textDocument/didChange":
		var params DidChangeTextDocumentParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := server.DidChange(ctx, &params)
		return true, reply(ctx, nil, err)
	case "textDocument/didClose":
		var params DidCloseTextDocumentParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := server.DidClose(ctx, &params)
		return true, reply(ctx, nil, err)
	case "textDocument/didOpen":
		var params DidOpenTextDocumentParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := server.DidOpen(ctx, &params)
		return true, reply(ctx, nil, err)
	case "textDocument/didSave":
		var params DidSaveTextDocumentParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := server.DidSave(ctx, &params)
		return true, reply(ctx, nil, err)
	case "textDocument/documentColor":
		var params DocumentColorParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.DocumentColor(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "textDocument/documentHighlight":
		var params DocumentHighlightParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.DocumentHighlight(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "textDocument/documentLink":
		var params DocumentLinkParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.DocumentLink(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "textDocument/documentSymbol":
		var params DocumentSymbolParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.DocumentSymbol(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "textDocument/foldingRange":
		var params FoldingRangeParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.FoldingRange(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "textDocument/formatting":
		var params DocumentFormattingParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.Formatting(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "textDocument/hover":
		var params HoverParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.Hover(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "textDocument/implementation":
		var params ImplementationParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.Implementation(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "textDocument/inlayHint":
		var params InlayHintParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.InlayHint(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "textDocument/inlineValue":
		var params InlineValueParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.InlineValue(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "textDocument/linkedEditingRange":
		var params LinkedEditingRangeParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.LinkedEditingRange(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "textDocument/moniker":
		var params MonikerParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.Moniker(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "textDocument/onTypeFormatting":
		var params DocumentOnTypeFormattingParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.OnTypeFormatting(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "textDocument/prepareCallHierarchy":
		var params CallHierarchyPrepareParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.PrepareCallHierarchy(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "textDocument/prepareRename":
		var params PrepareRenameParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.PrepareRename(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "textDocument/prepareTypeHierarchy":
		var params TypeHierarchyPrepareParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.PrepareTypeHierarchy(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "textDocument/rangeFormatting":
		var params DocumentRangeFormattingParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.RangeFormatting(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "textDocument/references":
		var params ReferenceParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.References(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "textDocument/rename":
		var params RenameParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.Rename(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "textDocument/selectionRange":
		var params SelectionRangeParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.SelectionRange(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "textDocument/semanticTokens/full":
		var params SemanticTokensParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.SemanticTokensFull(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "textDocument/semanticTokens/full/delta":
		var params SemanticTokensDeltaParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.SemanticTokensFullDelta(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "textDocument/semanticTokens/range":
		var params SemanticTokensRangeParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.SemanticTokensRange(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "textDocument/signatureHelp":
		var params SignatureHelpParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.SignatureHelp(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "textDocument/typeDefinition":
		var params TypeDefinitionParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.TypeDefinition(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "textDocument/willSave":
		var params WillSaveTextDocumentParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := server.WillSave(ctx, &params)
		return true, reply(ctx, nil, err)
	case "textDocument/willSaveWaitUntil":
		var params WillSaveTextDocumentParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.WillSaveWaitUntil(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "typeHierarchy/subtypes":
		var params TypeHierarchySubtypesParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.Subtypes(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "typeHierarchy/supertypes":
		var params TypeHierarchySupertypesParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.Supertypes(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "window/workDoneProgress/cancel":
		var params WorkDoneProgressCancelParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := server.WorkDoneProgressCancel(ctx, &params)
		return true, reply(ctx, nil, err)
	case "workspace/diagnostic":
		var params WorkspaceDiagnosticParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.DiagnosticWorkspace(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "workspace/didChangeConfiguration":
		var params DidChangeConfigurationParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := server.DidChangeConfiguration(ctx, &params)
		return true, reply(ctx, nil, err)
	case "workspace/didChangeWatchedFiles":
		var params DidChangeWatchedFilesParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := server.DidChangeWatchedFiles(ctx, &params)
		return true, reply(ctx, nil, err)
	case "workspace/didChangeWorkspaceFolders":
		var params DidChangeWorkspaceFoldersParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := server.DidChangeWorkspaceFolders(ctx, &params)
		return true, reply(ctx, nil, err)
	case "workspace/didCreateFiles":
		var params CreateFilesParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := server.DidCreateFiles(ctx, &params)
		return true, reply(ctx, nil, err)
	case "workspace/didDeleteFiles":
		var params DeleteFilesParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := server.DidDeleteFiles(ctx, &params)
		return true, reply(ctx, nil, err)
	case "workspace/didRenameFiles":
		var params RenameFilesParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		err := server.DidRenameFiles(ctx, &params)
		return true, reply(ctx, nil, err)
	case "workspace/executeCommand":
		var params ExecuteCommandParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.ExecuteCommand(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "workspace/symbol":
		var params WorkspaceSymbolParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.Symbol(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "workspace/willCreateFiles":
		var params CreateFilesParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.WillCreateFiles(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "workspace/willDeleteFiles":
		var params DeleteFilesParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.WillDeleteFiles(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "workspace/willRenameFiles":
		var params RenameFilesParams
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.WillRenameFiles(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	case "workspaceSymbol/resolve":
		var params WorkspaceSymbol
		if err := json.Unmarshal(r.Params(), &params); err != nil {
			return true, sendParseError(ctx, reply, err)
		}
		resp, err := server.ResolveWorkspaceSymbol(ctx, &params)
		if err != nil {
			return true, reply(ctx, nil, err)
		}
		return true, reply(ctx, resp, nil)
	default:
		return false, nil
	}
}

func (s *serverDispatcher) Progress(ctx context.Context, params *ProgressParams) error {
	return s.sender.Notify(ctx, "$/progress", params)
}
func (s *serverDispatcher) SetTrace(ctx context.Context, params *SetTraceParams) error {
	return s.sender.Notify(ctx, "$/setTrace", params)
}
func (s *serverDispatcher) IncomingCalls(ctx context.Context, params *CallHierarchyIncomingCallsParams) ([]CallHierarchyIncomingCall, error) {
	var result []CallHierarchyIncomingCall
	if err := s.sender.Call(ctx, "callHierarchy/incomingCalls", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}
func (s *serverDispatcher) OutgoingCalls(ctx context.Context, params *CallHierarchyOutgoingCallsParams) ([]CallHierarchyOutgoingCall, error) {
	var result []CallHierarchyOutgoingCall
	if err := s.sender.Call(ctx, "callHierarchy/outgoingCalls", params, &result); err != nil {
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
func (s *serverDispatcher) ResolveCodeLens(ctx context.Context, params *CodeLens) (*CodeLens, error) {
	var result *CodeLens
	if err := s.sender.Call(ctx, "codeLens/resolve", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}
func (s *serverDispatcher) ResolveCompletionItem(ctx context.Context, params *CompletionItem) (*CompletionItem, error) {
	var result *CompletionItem
	if err := s.sender.Call(ctx, "completionItem/resolve", params, &result); err != nil {
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
func (s *serverDispatcher) Exit(ctx context.Context) error {
	return s.sender.Notify(ctx, "exit", nil)
}
func (s *serverDispatcher) Initialize(ctx context.Context, params *ParamInitialize) (*InitializeResult, error) {
	var result *InitializeResult
	if err := s.sender.Call(ctx, "initialize", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}
func (s *serverDispatcher) Initialized(ctx context.Context, params *InitializedParams) error {
	return s.sender.Notify(ctx, "initialized", params)
}
func (s *serverDispatcher) Resolve(ctx context.Context, params *InlayHint) (*InlayHint, error) {
	var result *InlayHint
	if err := s.sender.Call(ctx, "inlayHint/resolve", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}
func (s *serverDispatcher) DidChangeNotebookDocument(ctx context.Context, params *DidChangeNotebookDocumentParams) error {
	return s.sender.Notify(ctx, "notebookDocument/didChange", params)
}
func (s *serverDispatcher) DidCloseNotebookDocument(ctx context.Context, params *DidCloseNotebookDocumentParams) error {
	return s.sender.Notify(ctx, "notebookDocument/didClose", params)
}
func (s *serverDispatcher) DidOpenNotebookDocument(ctx context.Context, params *DidOpenNotebookDocumentParams) error {
	return s.sender.Notify(ctx, "notebookDocument/didOpen", params)
}
func (s *serverDispatcher) DidSaveNotebookDocument(ctx context.Context, params *DidSaveNotebookDocumentParams) error {
	return s.sender.Notify(ctx, "notebookDocument/didSave", params)
}
func (s *serverDispatcher) Shutdown(ctx context.Context) error {
	return s.sender.Call(ctx, "shutdown", nil, nil)
}
func (s *serverDispatcher) CodeAction(ctx context.Context, params *CodeActionParams) ([]CodeAction, error) {
	var result []CodeAction
	if err := s.sender.Call(ctx, "textDocument/codeAction", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}
func (s *serverDispatcher) CodeLens(ctx context.Context, params *CodeLensParams) ([]CodeLens, error) {
	var result []CodeLens
	if err := s.sender.Call(ctx, "textDocument/codeLens", params, &result); err != nil {
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
func (s *serverDispatcher) Completion(ctx context.Context, params *CompletionParams) (*CompletionList, error) {
	var result *CompletionList
	if err := s.sender.Call(ctx, "textDocument/completion", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}
func (s *serverDispatcher) Declaration(ctx context.Context, params *DeclarationParams) (*Or_textDocument_declaration, error) {
	var result *Or_textDocument_declaration
	if err := s.sender.Call(ctx, "textDocument/declaration", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}
func (s *serverDispatcher) Definition(ctx context.Context, params *DefinitionParams) ([]Location, error) {
	var result []Location
	if err := s.sender.Call(ctx, "textDocument/definition", params, &result); err != nil {
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
func (s *serverDispatcher) DidChange(ctx context.Context, params *DidChangeTextDocumentParams) error {
	return s.sender.Notify(ctx, "textDocument/didChange", params)
}
func (s *serverDispatcher) DidClose(ctx context.Context, params *DidCloseTextDocumentParams) error {
	return s.sender.Notify(ctx, "textDocument/didClose", params)
}
func (s *serverDispatcher) DidOpen(ctx context.Context, params *DidOpenTextDocumentParams) error {
	return s.sender.Notify(ctx, "textDocument/didOpen", params)
}
func (s *serverDispatcher) DidSave(ctx context.Context, params *DidSaveTextDocumentParams) error {
	return s.sender.Notify(ctx, "textDocument/didSave", params)
}
func (s *serverDispatcher) DocumentColor(ctx context.Context, params *DocumentColorParams) ([]ColorInformation, error) {
	var result []ColorInformation
	if err := s.sender.Call(ctx, "textDocument/documentColor", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}
func (s *serverDispatcher) DocumentHighlight(ctx context.Context, params *DocumentHighlightParams) ([]DocumentHighlight, error) {
	var result []DocumentHighlight
	if err := s.sender.Call(ctx, "textDocument/documentHighlight", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}
func (s *serverDispatcher) DocumentLink(ctx context.Context, params *DocumentLinkParams) ([]DocumentLink, error) {
	var result []DocumentLink
	if err := s.sender.Call(ctx, "textDocument/documentLink", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}
func (s *serverDispatcher) DocumentSymbol(ctx context.Context, params *DocumentSymbolParams) ([]interface{}, error) {
	var result []interface{}
	if err := s.sender.Call(ctx, "textDocument/documentSymbol", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}
func (s *serverDispatcher) FoldingRange(ctx context.Context, params *FoldingRangeParams) ([]FoldingRange, error) {
	var result []FoldingRange
	if err := s.sender.Call(ctx, "textDocument/foldingRange", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}
func (s *serverDispatcher) Formatting(ctx context.Context, params *DocumentFormattingParams) ([]TextEdit, error) {
	var result []TextEdit
	if err := s.sender.Call(ctx, "textDocument/formatting", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}
func (s *serverDispatcher) Hover(ctx context.Context, params *HoverParams) (*Hover, error) {
	var result *Hover
	if err := s.sender.Call(ctx, "textDocument/hover", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}
func (s *serverDispatcher) Implementation(ctx context.Context, params *ImplementationParams) ([]Location, error) {
	var result []Location
	if err := s.sender.Call(ctx, "textDocument/implementation", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}
func (s *serverDispatcher) InlayHint(ctx context.Context, params *InlayHintParams) ([]InlayHint, error) {
	var result []InlayHint
	if err := s.sender.Call(ctx, "textDocument/inlayHint", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}
func (s *serverDispatcher) InlineValue(ctx context.Context, params *InlineValueParams) ([]InlineValue, error) {
	var result []InlineValue
	if err := s.sender.Call(ctx, "textDocument/inlineValue", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}
func (s *serverDispatcher) LinkedEditingRange(ctx context.Context, params *LinkedEditingRangeParams) (*LinkedEditingRanges, error) {
	var result *LinkedEditingRanges
	if err := s.sender.Call(ctx, "textDocument/linkedEditingRange", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}
func (s *serverDispatcher) Moniker(ctx context.Context, params *MonikerParams) ([]Moniker, error) {
	var result []Moniker
	if err := s.sender.Call(ctx, "textDocument/moniker", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}
func (s *serverDispatcher) OnTypeFormatting(ctx context.Context, params *DocumentOnTypeFormattingParams) ([]TextEdit, error) {
	var result []TextEdit
	if err := s.sender.Call(ctx, "textDocument/onTypeFormatting", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}
func (s *serverDispatcher) PrepareCallHierarchy(ctx context.Context, params *CallHierarchyPrepareParams) ([]CallHierarchyItem, error) {
	var result []CallHierarchyItem
	if err := s.sender.Call(ctx, "textDocument/prepareCallHierarchy", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}
func (s *serverDispatcher) PrepareRename(ctx context.Context, params *PrepareRenameParams) (*PrepareRename2Gn, error) {
	var result *PrepareRename2Gn
	if err := s.sender.Call(ctx, "textDocument/prepareRename", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}
func (s *serverDispatcher) PrepareTypeHierarchy(ctx context.Context, params *TypeHierarchyPrepareParams) ([]TypeHierarchyItem, error) {
	var result []TypeHierarchyItem
	if err := s.sender.Call(ctx, "textDocument/prepareTypeHierarchy", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}
func (s *serverDispatcher) RangeFormatting(ctx context.Context, params *DocumentRangeFormattingParams) ([]TextEdit, error) {
	var result []TextEdit
	if err := s.sender.Call(ctx, "textDocument/rangeFormatting", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}
func (s *serverDispatcher) References(ctx context.Context, params *ReferenceParams) ([]Location, error) {
	var result []Location
	if err := s.sender.Call(ctx, "textDocument/references", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}
func (s *serverDispatcher) Rename(ctx context.Context, params *RenameParams) (*WorkspaceEdit, error) {
	var result *WorkspaceEdit
	if err := s.sender.Call(ctx, "textDocument/rename", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}
func (s *serverDispatcher) SelectionRange(ctx context.Context, params *SelectionRangeParams) ([]SelectionRange, error) {
	var result []SelectionRange
	if err := s.sender.Call(ctx, "textDocument/selectionRange", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}
func (s *serverDispatcher) SemanticTokensFull(ctx context.Context, params *SemanticTokensParams) (*SemanticTokens, error) {
	var result *SemanticTokens
	if err := s.sender.Call(ctx, "textDocument/semanticTokens/full", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}
func (s *serverDispatcher) SemanticTokensFullDelta(ctx context.Context, params *SemanticTokensDeltaParams) (interface{}, error) {
	var result interface{}
	if err := s.sender.Call(ctx, "textDocument/semanticTokens/full/delta", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}
func (s *serverDispatcher) SemanticTokensRange(ctx context.Context, params *SemanticTokensRangeParams) (*SemanticTokens, error) {
	var result *SemanticTokens
	if err := s.sender.Call(ctx, "textDocument/semanticTokens/range", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}
func (s *serverDispatcher) SignatureHelp(ctx context.Context, params *SignatureHelpParams) (*SignatureHelp, error) {
	var result *SignatureHelp
	if err := s.sender.Call(ctx, "textDocument/signatureHelp", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}
func (s *serverDispatcher) TypeDefinition(ctx context.Context, params *TypeDefinitionParams) ([]Location, error) {
	var result []Location
	if err := s.sender.Call(ctx, "textDocument/typeDefinition", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}
func (s *serverDispatcher) WillSave(ctx context.Context, params *WillSaveTextDocumentParams) error {
	return s.sender.Notify(ctx, "textDocument/willSave", params)
}
func (s *serverDispatcher) WillSaveWaitUntil(ctx context.Context, params *WillSaveTextDocumentParams) ([]TextEdit, error) {
	var result []TextEdit
	if err := s.sender.Call(ctx, "textDocument/willSaveWaitUntil", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}
func (s *serverDispatcher) Subtypes(ctx context.Context, params *TypeHierarchySubtypesParams) ([]TypeHierarchyItem, error) {
	var result []TypeHierarchyItem
	if err := s.sender.Call(ctx, "typeHierarchy/subtypes", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}
func (s *serverDispatcher) Supertypes(ctx context.Context, params *TypeHierarchySupertypesParams) ([]TypeHierarchyItem, error) {
	var result []TypeHierarchyItem
	if err := s.sender.Call(ctx, "typeHierarchy/supertypes", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}
func (s *serverDispatcher) WorkDoneProgressCancel(ctx context.Context, params *WorkDoneProgressCancelParams) error {
	return s.sender.Notify(ctx, "window/workDoneProgress/cancel", params)
}
func (s *serverDispatcher) DiagnosticWorkspace(ctx context.Context, params *WorkspaceDiagnosticParams) (*WorkspaceDiagnosticReport, error) {
	var result *WorkspaceDiagnosticReport
	if err := s.sender.Call(ctx, "workspace/diagnostic", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}
func (s *serverDispatcher) DidChangeConfiguration(ctx context.Context, params *DidChangeConfigurationParams) error {
	return s.sender.Notify(ctx, "workspace/didChangeConfiguration", params)
}
func (s *serverDispatcher) DidChangeWatchedFiles(ctx context.Context, params *DidChangeWatchedFilesParams) error {
	return s.sender.Notify(ctx, "workspace/didChangeWatchedFiles", params)
}
func (s *serverDispatcher) DidChangeWorkspaceFolders(ctx context.Context, params *DidChangeWorkspaceFoldersParams) error {
	return s.sender.Notify(ctx, "workspace/didChangeWorkspaceFolders", params)
}
func (s *serverDispatcher) DidCreateFiles(ctx context.Context, params *CreateFilesParams) error {
	return s.sender.Notify(ctx, "workspace/didCreateFiles", params)
}
func (s *serverDispatcher) DidDeleteFiles(ctx context.Context, params *DeleteFilesParams) error {
	return s.sender.Notify(ctx, "workspace/didDeleteFiles", params)
}
func (s *serverDispatcher) DidRenameFiles(ctx context.Context, params *RenameFilesParams) error {
	return s.sender.Notify(ctx, "workspace/didRenameFiles", params)
}
func (s *serverDispatcher) ExecuteCommand(ctx context.Context, params *ExecuteCommandParams) (interface{}, error) {
	var result interface{}
	if err := s.sender.Call(ctx, "workspace/executeCommand", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}
func (s *serverDispatcher) Symbol(ctx context.Context, params *WorkspaceSymbolParams) ([]SymbolInformation, error) {
	var result []SymbolInformation
	if err := s.sender.Call(ctx, "workspace/symbol", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}
func (s *serverDispatcher) WillCreateFiles(ctx context.Context, params *CreateFilesParams) (*WorkspaceEdit, error) {
	var result *WorkspaceEdit
	if err := s.sender.Call(ctx, "workspace/willCreateFiles", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}
func (s *serverDispatcher) WillDeleteFiles(ctx context.Context, params *DeleteFilesParams) (*WorkspaceEdit, error) {
	var result *WorkspaceEdit
	if err := s.sender.Call(ctx, "workspace/willDeleteFiles", params, &result); err != nil {
		return nil, err
	}
	return result, nil
}
func (s *serverDispatcher) WillRenameFiles(ctx context.Context, params *RenameFilesParams) (*WorkspaceEdit, error) {
	var result *WorkspaceEdit
	if err := s.sender.Call(ctx, "workspace/willRenameFiles", params, &result); err != nil {
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
func (s *serverDispatcher) NonstandardRequest(ctx context.Context, method string, params interface{}) (interface{}, error) {
	var result interface{}
	if err := s.sender.Call(ctx, method, params, &result); err != nil {
		return nil, err
	}
	return result, nil
}
