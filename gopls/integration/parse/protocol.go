// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package parse

import (
	"log"

	p "golang.org/x/tools/internal/lsp/protocol"
)

// Requests and notifications are fixed types
// Responses may be one of several types

// Requests returns a pointer to a type suitable for Unmarshal
func Requests(m string) interface{} {
	// these are in the documentation's order
	switch m {
	case "initialize":
		return new(p.InitializeParams)
	case "shutdown":
		return new(struct{})
	case "window/showMessgeRequest":
		return new(p.ShowMessageRequestParams)
	case "client/registerCapability":
		return new(p.RegistrationParams)
	case "client/unregisterCapability":
		return new(p.UnregistrationParams)
	case "workspace/workspaceFolders":
		return nil
	case "workspace/configuration":
		return new(p.ConfigurationParams)
	case "workspace/symbol":
		return new(p.WorkspaceSymbolParams)
	case "workspace/executeCommand":
		return new(p.ExecuteCommandParams)
	case "workspace/applyEdit":
		return new(p.ApplyWorkspaceEditParams)
	case "textDocument/willSaveWaitUntil":
		return new(p.WillSaveTextDocumentParams)
	case "textDocument/completion":
		return new(p.CompletionParams)
	case "completionItem/resolve":
		return new(p.CompletionItem)
	case "textDocument/hover":
		return new(p.TextDocumentPositionParams)
	case "textDocument/signatureHelp":
		return new(p.TextDocumentPositionParams)
	case "textDocument/declaration":
		return new(p.TextDocumentPositionParams)
	case "textDocument/definition":
		return new(p.TextDocumentPositionParams)
	case "textDocument/typeDefinition":
		return new(p.TextDocumentPositionParams)
	case "textDocument/implementation":
		return new(p.TextDocumentPositionParams)
	case "textDocument/references":
		return new(p.ReferenceParams)
	case "textDocument/documentHighlight":
		return new(p.TextDocumentPositionParams)
	case "textDocument/documentSymbol":
		return new(p.DocumentSymbolParams)
	case "textDocument/codeAction":
		return new(p.CodeActionParams)
	case "textDocument/codeLens":
		return new(p.CodeLensParams)
	case "codeLens/resolve":
		return new(p.CodeLens)
	case "textDocument/documentLink":
		return new(p.DocumentLinkParams)
	case "documentLink/resolve":
		return new(p.DocumentLink)
	case "textDocument/documentColor":
		return new(p.DocumentColorParams)
	case "textDocument/colorPressentation":
		return new(p.ColorPresentationParams)
	case "textDocument/formatting":
		return new(p.DocumentFormattingParams)
	case "textDocument/rangeFormatting":
		return new(p.DocumentRangeFormattingParams)
	case "textDocument/typeFormatting":
		return new(p.DocumentOnTypeFormattingParams)
	case "textDocument/rename":
		return new(p.RenameParams)
	case "textDocument/prepareRename":
		return new(p.TextDocumentPositionParams)
	case "textDocument/foldingRange":
		return new(p.FoldingRangeParams)
	}
	log.Fatalf("request(%s) undefined", m)
	return ""
}

// Notifs returns a pointer to a type suitable for Unmarshal
func Notifs(m string) interface{} {
	switch m {
	case "$/cancelRequest":
		return new(p.CancelParams)
	case "$/setTraceNotification":
		return new(struct{ Value string })
	case "client/registerCapability": // why is this a notification? (serer->client rpc)
		return new(p.RegistrationParams)
	case "initialized":
		return new(p.InitializedParams)
	case "exit":
		return nil
	case "window/showMessage":
		return new(p.ShowMessageParams)
	case "window/logMessage":
		return new(p.LogMessageParams)
	case "telemetry/event":
		return new(interface{}) // any
	case "workspace/didChangeWorkspaceFolders":
		return new(p.DidChangeWorkspaceFoldersParams)
	case "workspace/didChangeConfiguration":
		return new(p.DidChangeConfigurationParams)
	case "workspace/didChangeWatchedFiles":
		return new(p.DidChangeWatchedFilesParams)
	case "textDocument/didOpen":
		return new(p.DidOpenTextDocumentParams)
	case "textDocument/didChange":
		return new(p.DidChangeTextDocumentParams)
	case "textDocument/willSave":
		return new(p.WillSaveTextDocumentParams)
	case "textDocument/didSave":
		return new(p.DidSaveTextDocumentParams)
	case "textDocument/didClose":
		return new(p.DidCloseTextDocumentParams)
	case "textDocument/willClose":
		return new(p.DidCloseTextDocumentParams)
	case "textDocument/publishDiagnostics":
		return new(p.PublishDiagnosticsParams)
	}
	log.Fatalf("notif(%s) undefined", m)
	return ""
}

// Responses returns a slice of types, one of which should be
// suitable for Unmarshal
func Responses(m string) []interface{} {
	switch m {
	case "initialize":
		return []interface{}{new(p.InitializeResult)}
	case "shutdown":
		return []interface{}{nil}
	case "window/showMessageRequest":
		return []interface{}{new(p.MessageActionItem), nil}
	case "client/registerCapability":
		return []interface{}{nil}
	case "client/unregisterCapability":
		return []interface{}{nil}
	case "workspace/workspaceFolder":
		return []interface{}{new([]p.WorkspaceFolder), nil}
	case "workspace/configuration":
		return []interface{}{new([]interface{}), new(interface{})}
	case "workspace/symbol":
		return []interface{}{new([]p.SymbolInformation), nil}
	case "workspace/executeCommand":
		return []interface{}{new(interface{}), nil}
	case "workspace/applyEdit":
		return []interface{}{new(p.ApplyWorkspaceEditResponse)}
	case "textDocument/willSaveWaitUntil":
		return []interface{}{new([]p.TextEdit), nil}
	case "textDocument/completion":
		return []interface{}{new(p.CompletionList), new([]p.CompletionItem), nil}
	case "completionItem/resolve":
		return []interface{}{new(p.CompletionItem)}
	case "textDocument/hover":
		return []interface{}{new(p.Hover), nil}
	case "textDocument/signatureHelp":
		return []interface{}{new(p.SignatureHelp), nil}
	case "textDocument/declaration":
		return []interface{}{new(p.Location), new([]p.Location), new([]p.LocationLink), nil}
	case "textDocument/definition":
		return []interface{}{new([]p.Location), new([]p.Location), new([]p.LocationLink), nil}
	case "textDocument/typeDefinition":
		return []interface{}{new([]p.Location), new([]p.LocationLink), new(p.Location), nil}
	case "textDocument/implementation":
		return []interface{}{new(p.Location), new([]p.Location), new([]p.LocationLink), nil}
	case "textDocument/references":
		return []interface{}{new([]p.Location), nil}
	case "textDocument/documentHighlight":
		return []interface{}{new([]p.DocumentHighlight), nil}
	case "textDocument/documentSymbol":
		return []interface{}{new([]p.DocumentSymbol), new([]p.SymbolInformation), nil}
	case "textDocument/codeAction":
		return []interface{}{new([]p.CodeAction), new(p.Command), nil}
	case "textDocument/codeLens":
		return []interface{}{new([]p.CodeLens), nil}
	case "codelens/resolve":
		return []interface{}{new(p.CodeLens)}
	case "textDocument/documentLink":
		return []interface{}{new([]p.DocumentLink), nil}
	case "documentLink/resolve":
		return []interface{}{new(p.DocumentLink)}
	case "textDocument/documentColor":
		return []interface{}{new([]p.ColorInformation)}
	case "textDocument/colorPresentation":
		return []interface{}{new([]p.ColorPresentation)}
	case "textDocument/formatting":
		return []interface{}{new([]p.TextEdit), nil}
	case "textDocument/rangeFormatting":
		return []interface{}{new([]p.TextEdit), nil}
	case "textDocument/onTypeFormatting":
		return []interface{}{new([]p.TextEdit), nil}
	case "textDocument/rename":
		return []interface{}{new(p.WorkspaceEdit), nil}
	case "textDocument/prepareRename":
		return []interface{}{new(p.Range), nil}
	case "textDocument/foldingRange":
		return []interface{}{new([]p.FoldingRange), nil}
	}
	log.Fatalf("responses(%q) undefined", m)
	return nil
}

// Msgtype given method names. Note that mSrv|mCl is possible
type Msgtype int

const (
	// Mnot for notifications
	Mnot Msgtype = 1
	// Mreq for requests
	Mreq Msgtype = 2
	// Msrv for messages from the server
	Msrv Msgtype = 4
	// Mcl for messages from the client
	Mcl Msgtype = 8
)

// IsNotify says if the message is a notification
func IsNotify(msg string) bool {
	m, ok := fromMethod[msg]
	if !ok {
		log.Fatalf("%q", msg)
	}
	return m&Mnot != 0
}

// FromServer says if the message is from the server
func FromServer(msg string) bool {
	m, ok := fromMethod[msg]
	if !ok {
		log.Fatalf("%q", msg)
	}
	return m&Msrv != 0
}

// FromClient says if the message is from the client
func FromClient(msg string) bool {
	m, ok := fromMethod[msg]
	if !ok {
		log.Fatalf("%q", msg)
	}
	return m&Mcl != 0
}

// rpc name to message type
var fromMethod = map[string]Msgtype{
	"$/cancelRequest":             Mnot | Msrv | Mcl,
	"initialize":                  Mreq | Msrv,
	"initialized":                 Mnot | Mcl,
	"shutdown":                    Mreq | Mcl,
	"exit":                        Mnot | Mcl,
	"window/showMessage":          Mreq | Msrv,
	"window/logMessage":           Mnot | Msrv,
	"telemetry'event":             Mnot | Msrv,
	"client/registerCapability":   Mreq | Msrv,
	"client/unregisterCapability": Mreq | Msrv,
	"workspace/workspaceFolders":  Mreq | Msrv,
	"workspace/workspaceDidChangeWorkspaceFolders": Mnot | Mcl,
	"workspace/didChangeConfiguration":             Mnot | Mcl,
	"workspace/configuration":                      Mreq | Msrv,
	"workspace/didChangeWatchedFiles":              Mnot | Mcl,
	"workspace/symbol":                             Mreq | Mcl,
	"workspace/executeCommand":                     Mreq | Mcl,
	"workspace/applyEdit":                          Mreq | Msrv,
	"textDocument/didOpen":                         Mnot | Mcl,
	"textDocument/didChange":                       Mnot | Mcl,
	"textDocument/willSave":                        Mnot | Mcl,
	"textDocument/willSaveWaitUntil":               Mreq | Mcl,
	"textDocument/didSave":                         Mnot | Mcl,
	"textDocument/didClose":                        Mnot | Mcl,
	"textDocument/publishDiagnostics":              Mnot | Msrv,
	"textDocument/completion":                      Mreq | Mcl,
	"completionItem/resolve":                       Mreq | Mcl,
	"textDocument/hover":                           Mreq | Mcl,
	"textDocument/signatureHelp":                   Mreq | Mcl,
	"textDocument/declaration":                     Mreq | Mcl,
	"textDocument/definition":                      Mreq | Mcl,
	"textDocument/typeDefinition":                  Mreq | Mcl,
	"textDocument/implementation":                  Mreq | Mcl,
	"textDocument/references":                      Mreq | Mcl,
	"textDocument/documentHighlight":               Mreq | Mcl,
	"textDocument/documentSymbol":                  Mreq | Mcl,
	"textDocument/codeAction":                      Mreq | Mcl,
	"textDocument/codeLens":                        Mreq | Mcl,
	"codeLens/resolve":                             Mreq | Mcl,
	"textDocument/documentLink":                    Mreq | Mcl,
	"documentLink/resolve":                         Mreq | Mcl,
	"textDocument/documentColor":                   Mreq | Mcl,
	"textDocument/colorPresentation":               Mreq | Mcl,
	"textDocument/formatting":                      Mreq | Mcl,
	"textDocument/rangeFormatting":                 Mreq | Mcl,
	"textDocument/onTypeFormatting":                Mreq | Mcl,
	"textDocument/rename":                          Mreq | Mcl,
	"textDocument/prepareRename":                   Mreq | Mcl,
	"textDocument/foldingRange":                    Mreq | Mcl,
}
