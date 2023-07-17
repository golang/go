// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.19
// +build go1.19

package main

import "log"

// prop combines the name of a property with the name of the structure it is in.
type prop [2]string

const (
	nothing = iota
	wantStar
	wantOpt
	wantOptStar
)

// goplsStar records the optionality of each field in the protocol.
// The comments are vague hints as to why removing the line is not trivial.
// A.B.C.D means that one of B or C would change to a pointer
// so a test or initialization would be needed
var goplsStar = map[prop]int{
	{"ClientCapabilities", "textDocument"}: wantOpt, // A.B.C.D at fake/editor.go:255
	{"ClientCapabilities", "window"}:       wantOpt, // regtest failures
	{"ClientCapabilities", "workspace"}:    wantOpt, // regtest failures
	{"CodeAction", "kind"}:                 wantOpt, // A.B.C.D

	{"CodeActionClientCapabilities", "codeActionLiteralSupport"}: wantOpt, // regtest failures

	{"CompletionClientCapabilities", "completionItem"}: wantOpt, // A.B.C.D
	{"CompletionClientCapabilities", "insertTextMode"}: wantOpt, // A.B.C.D
	{"CompletionItem", "kind"}:                         wantOpt, // need temporary variables
	{"CompletionParams", "context"}:                    wantOpt, // needs nil checks

	{"Diagnostic", "severity"}:            wantOpt,     // nil checks or more careful thought
	{"DidSaveTextDocumentParams", "text"}: wantOptStar, // capabilities_test.go:112 logic
	{"DocumentHighlight", "kind"}:         wantOpt,     // need temporary variables
	{"Hover", "range"}:                    wantOpt,     // complex expressions
	{"InlayHint", "kind"}:                 wantOpt,     // temporary variables

	{"Lit_CompletionClientCapabilities_completionItem", "tagSupport"}:     nothing, // A.B.C.
	{"Lit_SemanticTokensClientCapabilities_requests", "full"}:             nothing, // A.B.C.D
	{"Lit_SemanticTokensClientCapabilities_requests", "range"}:            nothing, // A.B.C.D
	{"Lit_SemanticTokensClientCapabilities_requests_full_Item1", "delta"}: nothing, // A.B.C.D
	{"Lit_SemanticTokensOptions_full_Item1", "delta"}:                     nothing, // A.B.C.

	{"Lit_TextDocumentContentChangeEvent_Item0", "range"}: wantStar, // == nil test

	{"TextDocumentClientCapabilities", "codeAction"}:          wantOpt, // A.B.C.D
	{"TextDocumentClientCapabilities", "completion"}:          wantOpt, // A.B.C.D
	{"TextDocumentClientCapabilities", "documentSymbol"}:      wantOpt, // A.B.C.D
	{"TextDocumentClientCapabilities", "publishDiagnostics"}:  wantOpt, //A.B.C.D
	{"TextDocumentClientCapabilities", "semanticTokens"}:      wantOpt, // A.B.C.D
	{"TextDocumentSyncOptions", "change"}:                     wantOpt, // &constant
	{"WorkDoneProgressParams", "workDoneToken"}:               wantOpt, // regtest
	{"WorkspaceClientCapabilities", "didChangeConfiguration"}: wantOpt, // A.B.C.D
	{"WorkspaceClientCapabilities", "didChangeWatchedFiles"}:  wantOpt, // A.B.C.D
}

// keep track of which entries in goplsStar are used
var usedGoplsStar = make(map[prop]bool)

// For gopls compatibility, use a different, typically more restrictive, type for some fields.
var renameProp = map[prop]string{
	{"CancelParams", "id"}:         "interface{}",
	{"Command", "arguments"}:       "[]json.RawMessage",
	{"CompletionItem", "textEdit"}: "TextEdit",
	{"Diagnostic", "code"}:         "interface{}",
	{"Diagnostic", "data"}:         "json.RawMessage", // delay unmarshalling quickfixes

	{"DocumentDiagnosticReportPartialResult", "relatedDocuments"}: "map[DocumentURI]interface{}",

	{"ExecuteCommandParams", "arguments"}: "[]json.RawMessage",
	{"FoldingRange", "kind"}:              "string",
	{"Hover", "contents"}:                 "MarkupContent",
	{"InlayHint", "label"}:                "[]InlayHintLabelPart",

	{"RelatedFullDocumentDiagnosticReport", "relatedDocuments"}:      "map[DocumentURI]interface{}",
	{"RelatedUnchangedDocumentDiagnosticReport", "relatedDocuments"}: "map[DocumentURI]interface{}",

	// PJW: this one is tricky.
	{"ServerCapabilities", "codeActionProvider"}: "interface{}",

	{"ServerCapabilities", "inlayHintProvider"}: "interface{}",
	// slightly tricky
	{"ServerCapabilities", "renameProvider"}: "interface{}",
	// slightly tricky
	{"ServerCapabilities", "semanticTokensProvider"}: "interface{}",
	// slightly tricky
	{"ServerCapabilities", "textDocumentSync"}: "interface{}",
	{"TextDocumentEdit", "edits"}:              "[]TextEdit",
	{"TextDocumentSyncOptions", "save"}:        "SaveOptions",
	{"WorkspaceEdit", "documentChanges"}:       "[]DocumentChanges",
}

// which entries of renameProp were used
var usedRenameProp = make(map[prop]bool)

type adjust struct {
	prefix, suffix string
}

// disambiguate specifies prefixes or suffixes to add to all values of
// some enum types to avoid name conflicts
var disambiguate = map[string]adjust{
	"CodeActionTriggerKind":        {"CodeAction", ""},
	"CompletionItemKind":           {"", "Completion"},
	"CompletionItemTag":            {"Compl", ""},
	"DiagnosticSeverity":           {"Severity", ""},
	"DocumentDiagnosticReportKind": {"Diagnostic", ""},
	"FileOperationPatternKind":     {"", "Pattern"},
	"InlineCompletionTriggerKind":  {"Inline", ""},
	"InsertTextFormat":             {"", "TextFormat"},
	"SemanticTokenModifiers":       {"Mod", ""},
	"SemanticTokenTypes":           {"", "Type"},
	"SignatureHelpTriggerKind":     {"Sig", ""},
	"SymbolTag":                    {"", "Symbol"},
	"WatchKind":                    {"Watch", ""},
}

// which entries of disambiguate got used
var usedDisambiguate = make(map[string]bool)

// for gopls compatibility, replace generated type names with existing ones
var goplsType = map[string]string{
	"And_RegOpt_textDocument_colorPresentation": "WorkDoneProgressOptionsAndTextDocumentRegistrationOptions",
	"ConfigurationParams":                       "ParamConfiguration",
	"DocumentDiagnosticParams":                  "string",
	"DocumentDiagnosticReport":                  "string",
	"DocumentUri":                               "DocumentURI",
	"InitializeParams":                          "ParamInitialize",
	"LSPAny":                                    "interface{}",

	"Lit_CodeActionClientCapabilities_codeActionLiteralSupport":                "PCodeActionLiteralSupportPCodeAction",
	"Lit_CodeActionClientCapabilities_codeActionLiteralSupport_codeActionKind": "FCodeActionKindPCodeActionLiteralSupport",

	"Lit_CodeActionClientCapabilities_resolveSupport":     "PResolveSupportPCodeAction",
	"Lit_CodeAction_disabled":                             "PDisabledMsg_textDocument_codeAction",
	"Lit_CompletionClientCapabilities_completionItem":     "PCompletionItemPCompletion",
	"Lit_CompletionClientCapabilities_completionItemKind": "PCompletionItemKindPCompletion",

	"Lit_CompletionClientCapabilities_completionItem_insertTextModeSupport": "FInsertTextModeSupportPCompletionItem",

	"Lit_CompletionClientCapabilities_completionItem_resolveSupport": "FResolveSupportPCompletionItem",
	"Lit_CompletionClientCapabilities_completionItem_tagSupport":     "FTagSupportPCompletionItem",

	"Lit_CompletionClientCapabilities_completionList":     "PCompletionListPCompletion",
	"Lit_CompletionList_itemDefaults":                     "PItemDefaultsMsg_textDocument_completion",
	"Lit_CompletionList_itemDefaults_editRange_Item1":     "FEditRangePItemDefaults",
	"Lit_CompletionOptions_completionItem":                "PCompletionItemPCompletionProvider",
	"Lit_DocumentSymbolClientCapabilities_symbolKind":     "PSymbolKindPDocumentSymbol",
	"Lit_DocumentSymbolClientCapabilities_tagSupport":     "PTagSupportPDocumentSymbol",
	"Lit_FoldingRangeClientCapabilities_foldingRange":     "PFoldingRangePFoldingRange",
	"Lit_FoldingRangeClientCapabilities_foldingRangeKind": "PFoldingRangeKindPFoldingRange",
	"Lit_GeneralClientCapabilities_staleRequestSupport":   "PStaleRequestSupportPGeneral",
	"Lit_InitializeResult_serverInfo":                     "PServerInfoMsg_initialize",
	"Lit_InlayHintClientCapabilities_resolveSupport":      "PResolveSupportPInlayHint",
	"Lit_MarkedString_Item1":                              "Msg_MarkedString",
	"Lit_NotebookDocumentChangeEvent_cells":               "PCellsPChange",
	"Lit_NotebookDocumentChangeEvent_cells_structure":     "FStructurePCells",
	"Lit_NotebookDocumentFilter_Item0":                    "Msg_NotebookDocumentFilter",

	"Lit_NotebookDocumentSyncOptions_notebookSelector_Elem_Item0": "PNotebookSelectorPNotebookDocumentSync",

	"Lit_PrepareRenameResult_Item1": "Msg_PrepareRename2Gn",

	"Lit_PublishDiagnosticsClientCapabilities_tagSupport":       "PTagSupportPPublishDiagnostics",
	"Lit_SemanticTokensClientCapabilities_requests":             "PRequestsPSemanticTokens",
	"Lit_SemanticTokensClientCapabilities_requests_full_Item1":  "FFullPRequests",
	"Lit_SemanticTokensClientCapabilities_requests_range_Item1": "FRangePRequests",

	"Lit_SemanticTokensOptions_full_Item1":  "PFullESemanticTokensOptions",
	"Lit_SemanticTokensOptions_range_Item1": "PRangeESemanticTokensOptions",
	"Lit_ServerCapabilities_workspace":      "Workspace6Gn",

	"Lit_ShowMessageRequestClientCapabilities_messageActionItem": "PMessageActionItemPShowMessage",
	"Lit_SignatureHelpClientCapabilities_signatureInformation":   "PSignatureInformationPSignatureHelp",

	"Lit_SignatureHelpClientCapabilities_signatureInformation_parameterInformation": "FParameterInformationPSignatureInformation",

	"Lit_TextDocumentContentChangeEvent_Item0":                    "Msg_TextDocumentContentChangeEvent",
	"Lit_TextDocumentFilter_Item0":                                "Msg_TextDocumentFilter",
	"Lit_TextDocumentFilter_Item1":                                "Msg_TextDocumentFilter",
	"Lit_WorkspaceEditClientCapabilities_changeAnnotationSupport": "PChangeAnnotationSupportPWorkspaceEdit",
	"Lit_WorkspaceSymbolClientCapabilities_resolveSupport":        "PResolveSupportPSymbol",
	"Lit_WorkspaceSymbolClientCapabilities_symbolKind":            "PSymbolKindPSymbol",
	"Lit_WorkspaceSymbolClientCapabilities_tagSupport":            "PTagSupportPSymbol",
	"Lit_WorkspaceSymbol_location_Item1":                          "PLocationMsg_workspace_symbol",
	"Lit__InitializeParams_clientInfo":                            "Msg_XInitializeParams_clientInfo",
	"Or_CompletionList_itemDefaults_editRange":                    "OrFEditRangePItemDefaults",
	"Or_Declaration": "[]Location",
	"Or_DidChangeConfigurationRegistrationOptions_section": "OrPSection_workspace_didChangeConfiguration",
	"Or_GlobPattern":                "string",
	"Or_InlayHintLabelPart_tooltip": "OrPTooltipPLabel",
	"Or_InlayHint_tooltip":          "OrPTooltip_textDocument_inlayHint",
	"Or_LSPAny":                     "interface{}",
	"Or_NotebookDocumentFilter":     "Msg_NotebookDocumentFilter",
	"Or_NotebookDocumentSyncOptions_notebookSelector_Elem": "PNotebookSelectorPNotebookDocumentSync",

	"Or_NotebookDocumentSyncOptions_notebookSelector_Elem_Item0_notebook": "OrFNotebookPNotebookSelector",

	"Or_ParameterInformation_documentation":                     "string",
	"Or_ParameterInformation_label":                             "string",
	"Or_PrepareRenameResult":                                    "Msg_PrepareRename2Gn",
	"Or_ProgressToken":                                          "interface{}",
	"Or_Result_textDocument_completion":                         "CompletionList",
	"Or_Result_textDocument_declaration":                        "Or_textDocument_declaration",
	"Or_Result_textDocument_definition":                         "[]Location",
	"Or_Result_textDocument_documentSymbol":                     "[]interface{}",
	"Or_Result_textDocument_implementation":                     "[]Location",
	"Or_Result_textDocument_semanticTokens_full_delta":          "interface{}",
	"Or_Result_textDocument_typeDefinition":                     "[]Location",
	"Or_Result_workspace_symbol":                                "[]SymbolInformation",
	"Or_TextDocumentContentChangeEvent":                         "Msg_TextDocumentContentChangeEvent",
	"Or_TextDocumentFilter":                                     "Msg_TextDocumentFilter",
	"Or_WorkspaceFoldersServerCapabilities_changeNotifications": "string",
	"Or_WorkspaceSymbol_location":                               "OrPLocation_workspace_symbol",
	"PrepareRenameResult":                                       "PrepareRename2Gn",
	"Tuple_ParameterInformation_label_Item1":                    "UIntCommaUInt",
	"WorkspaceFoldersServerCapabilities":                        "WorkspaceFolders5Gn",
	"[]LSPAny":                                                  "[]interface{}",
	"[]Or_NotebookDocumentSyncOptions_notebookSelector_Elem":    "[]PNotebookSelectorPNotebookDocumentSync",
	"[]Or_Result_textDocument_codeAction_Item0_Elem":            "[]CodeAction",
	"[]PreviousResultId":                                        "[]PreviousResultID",
	"[]uinteger":                                                "[]uint32",
	"boolean":                                                   "bool",
	"decimal":                                                   "float64",
	"integer":                                                   "int32",
	"map[DocumentUri][]TextEdit":                                "map[DocumentURI][]TextEdit",
	"uinteger":                                                  "uint32",
}

var usedGoplsType = make(map[string]bool)

// methodNames is a map from the method to the name of the function that handles it
var methodNames = map[string]string{
	"$/cancelRequest":                        "CancelRequest",
	"$/logTrace":                             "LogTrace",
	"$/progress":                             "Progress",
	"$/setTrace":                             "SetTrace",
	"callHierarchy/incomingCalls":            "IncomingCalls",
	"callHierarchy/outgoingCalls":            "OutgoingCalls",
	"client/registerCapability":              "RegisterCapability",
	"client/unregisterCapability":            "UnregisterCapability",
	"codeAction/resolve":                     "ResolveCodeAction",
	"codeLens/resolve":                       "ResolveCodeLens",
	"completionItem/resolve":                 "ResolveCompletionItem",
	"documentLink/resolve":                   "ResolveDocumentLink",
	"exit":                                   "Exit",
	"initialize":                             "Initialize",
	"initialized":                            "Initialized",
	"inlayHint/resolve":                      "Resolve",
	"notebookDocument/didChange":             "DidChangeNotebookDocument",
	"notebookDocument/didClose":              "DidCloseNotebookDocument",
	"notebookDocument/didOpen":               "DidOpenNotebookDocument",
	"notebookDocument/didSave":               "DidSaveNotebookDocument",
	"shutdown":                               "Shutdown",
	"telemetry/event":                        "Event",
	"textDocument/codeAction":                "CodeAction",
	"textDocument/codeLens":                  "CodeLens",
	"textDocument/colorPresentation":         "ColorPresentation",
	"textDocument/completion":                "Completion",
	"textDocument/declaration":               "Declaration",
	"textDocument/definition":                "Definition",
	"textDocument/diagnostic":                "Diagnostic",
	"textDocument/didChange":                 "DidChange",
	"textDocument/didClose":                  "DidClose",
	"textDocument/didOpen":                   "DidOpen",
	"textDocument/didSave":                   "DidSave",
	"textDocument/documentColor":             "DocumentColor",
	"textDocument/documentHighlight":         "DocumentHighlight",
	"textDocument/documentLink":              "DocumentLink",
	"textDocument/documentSymbol":            "DocumentSymbol",
	"textDocument/foldingRange":              "FoldingRange",
	"textDocument/formatting":                "Formatting",
	"textDocument/hover":                     "Hover",
	"textDocument/implementation":            "Implementation",
	"textDocument/inlayHint":                 "InlayHint",
	"textDocument/inlineCompletion":          "InlineCompletion",
	"textDocument/inlineValue":               "InlineValue",
	"textDocument/linkedEditingRange":        "LinkedEditingRange",
	"textDocument/moniker":                   "Moniker",
	"textDocument/onTypeFormatting":          "OnTypeFormatting",
	"textDocument/prepareCallHierarchy":      "PrepareCallHierarchy",
	"textDocument/prepareRename":             "PrepareRename",
	"textDocument/prepareTypeHierarchy":      "PrepareTypeHierarchy",
	"textDocument/publishDiagnostics":        "PublishDiagnostics",
	"textDocument/rangeFormatting":           "RangeFormatting",
	"textDocument/rangesFormatting":          "RangesFormatting",
	"textDocument/references":                "References",
	"textDocument/rename":                    "Rename",
	"textDocument/selectionRange":            "SelectionRange",
	"textDocument/semanticTokens/full":       "SemanticTokensFull",
	"textDocument/semanticTokens/full/delta": "SemanticTokensFullDelta",
	"textDocument/semanticTokens/range":      "SemanticTokensRange",
	"textDocument/signatureHelp":             "SignatureHelp",
	"textDocument/typeDefinition":            "TypeDefinition",
	"textDocument/willSave":                  "WillSave",
	"textDocument/willSaveWaitUntil":         "WillSaveWaitUntil",
	"typeHierarchy/subtypes":                 "Subtypes",
	"typeHierarchy/supertypes":               "Supertypes",
	"window/logMessage":                      "LogMessage",
	"window/showDocument":                    "ShowDocument",
	"window/showMessage":                     "ShowMessage",
	"window/showMessageRequest":              "ShowMessageRequest",
	"window/workDoneProgress/cancel":         "WorkDoneProgressCancel",
	"window/workDoneProgress/create":         "WorkDoneProgressCreate",
	"workspace/applyEdit":                    "ApplyEdit",
	"workspace/codeLens/refresh":             "CodeLensRefresh",
	"workspace/configuration":                "Configuration",
	"workspace/diagnostic":                   "DiagnosticWorkspace",
	"workspace/diagnostic/refresh":           "DiagnosticRefresh",
	"workspace/didChangeConfiguration":       "DidChangeConfiguration",
	"workspace/didChangeWatchedFiles":        "DidChangeWatchedFiles",
	"workspace/didChangeWorkspaceFolders":    "DidChangeWorkspaceFolders",
	"workspace/didCreateFiles":               "DidCreateFiles",
	"workspace/didDeleteFiles":               "DidDeleteFiles",
	"workspace/didRenameFiles":               "DidRenameFiles",
	"workspace/executeCommand":               "ExecuteCommand",
	"workspace/inlayHint/refresh":            "InlayHintRefresh",
	"workspace/inlineValue/refresh":          "InlineValueRefresh",
	"workspace/semanticTokens/refresh":       "SemanticTokensRefresh",
	"workspace/symbol":                       "Symbol",
	"workspace/willCreateFiles":              "WillCreateFiles",
	"workspace/willDeleteFiles":              "WillDeleteFiles",
	"workspace/willRenameFiles":              "WillRenameFiles",
	"workspace/workspaceFolders":             "WorkspaceFolders",
	"workspaceSymbol/resolve":                "ResolveWorkspaceSymbol",
}

func methodName(method string) string {
	ans := methodNames[method]
	if ans == "" {
		log.Fatalf("unknown method %q", method)
	}
	return ans
}
