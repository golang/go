// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains the corresponding structures to the
// "General" messages part of the LSP specification.

package protocol

import "golang.org/x/tools/internal/jsonrpc2"

type CancelParams struct {
	/**
	 * The request id to cancel.
	 */
	ID jsonrpc2.ID `json:"id"`
}

type InitializeParams struct {
	/**
	 * The process Id of the parent process that started
	 * the server. Is null if the process has not been started by another process.
	 * If the parent process is not alive then the server should exit (see exit notification) its process.
	 */
	ProcessID *float64 `json:"processId"`

	/**
	 * The rootPath of the workspace. Is null
	 * if no folder is open.
	 *
	 * @deprecated in favour of rootURI.
	 */
	RootPath *string `json:"rootPath"`

	/**
	 * The rootURI of the workspace. Is null if no
	 * folder is open. If both `rootPath` and `rootURI` are set
	 * `rootURI` wins.
	 */
	RootURI *DocumentURI `json:"rootURI"`

	/**
	 * User provided initialization options.
	 */
	InitializationOptions interface{} `json:"initializationOptions"`

	/**
	 * The capabilities provided by the client (editor or tool)
	 */
	Capabilities ClientCapabilities `json:"capabilities"`

	/**
	 * The initial trace setting. If omitted trace is disabled ('off').
	 */
	Trace string `json:"trace"` // 'off' | 'messages' | 'verbose'

	/**
	 * The workspace folders configured in the client when the server starts.
	 * This property is only available if the client supports workspace folders.
	 * It can be `null` if the client supports workspace folders but none are
	 * configured.
	 *
	 * Since 3.6.0
	 */
	WorkspaceFolders []WorkspaceFolder `json:"workspaceFolders,omitempty"`
}

/**
 * Workspace specific client capabilities.
 */
type WorkspaceClientCapabilities struct {
	/**
	 * The client supports applying batch edits to the workspace by supporting
	 * the request 'workspace/applyEdit'
	 */
	ApplyEdit bool `json:"applyEdit,omitempty"`

	/**
	 * Capabilities specific to `WorkspaceEdit`s
	 */
	WorkspaceEdit struct {
		/**
		 * The client supports versioned document changes in `WorkspaceEdit`s
		 */
		DocumentChanges bool `json:"documentChanges,omitempty"`
	} `json:"workspaceEdit,omitempty"`

	/**
	 * Capabilities specific to the `workspace/didChangeConfiguration` notification.
	 */
	DidChangeConfiguration struct {
		/**
		 * Did change configuration notification supports dynamic registration.
		 */
		DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
	} `json:"didChangeConfiguration,omitempty"`

	/**
	 * Capabilities specific to the `workspace/didChangeWatchedFiles` notification.
	 */
	DidChangeWatchedFiles struct {
		/**
		 * Did change watched files notification supports dynamic registration.
		 */
		DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
	} `json:"didChangeWatchedFiles,omitempty"`

	/**
	 * Capabilities specific to the `workspace/symbol` request.
	 */
	Symbol struct {
		/**
		 * Symbol request supports dynamic registration.
		 */
		DynamicRegistration bool `json:"dynamicRegistration,omitempty"`

		/**
		 * Specific capabilities for the `SymbolKind` in the `workspace/symbol` request.
		 */
		SymbolKind struct {
			/**
			 * The symbol kind values the client supports. When this
			 * property exists the client also guarantees that it will
			 * handle values outside its set gracefully and falls back
			 * to a default value when unknown.
			 *
			 * If this property is not present the client only supports
			 * the symbol kinds from `File` to `Array` as defined in
			 * the initial version of the protocol.
			 */
			ValueSet []SymbolKind `json:"valueSet,omitempty"`
		} `json:"symbolKind,omitempty"`
	} `json:"symbol,omitempty"`

	/**
	 * Capabilities specific to the `workspace/executeCommand` request.
	 */
	ExecuteCommand struct {
		/**
		 * Execute command supports dynamic registration.
		 */
		DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
	} `json:"executeCommand,omitempty"`

	/**
	 * The client has support for workspace folders.
	 *
	 * Since 3.6.0
	 */
	WorkspaceFolders bool `json:"workspaceFolders,omitempty"`

	/**
	 * The client supports `workspace/configuration` requests.
	 *
	 * Since 3.6.0
	 */
	Configuration bool `json:"configuration,omitempty"`
}

/**
 * Text document specific client capabilities.
 */
type TextDocumentClientCapabilities struct {
	Synchronization struct {
		/**
		 * Whether text document synchronization supports dynamic registration.
		 */
		DynamicRegistration bool `json:"dynamicRegistration,omitempty"`

		/**
		 * The client supports sending will save notifications.
		 */
		WillSave bool `json:"willSave,omitempty"`

		/**
		 * The client supports sending a will save request and
		 * waits for a response providing text edits which will
		 * be applied to the document before it is saved.
		 */
		WillSaveWaitUntil bool `json:"willSaveWaitUntil,omitempty"`

		/**
		 * The client supports did save notifications.
		 */
		DidSave bool `json:"didSave,omitempty"`
	} `json:"synchronization,omitempty"`

	/**
	 * Capabilities specific to the `textDocument/completion`
	 */
	Completion struct {
		/**
		 * Whether completion supports dynamic registration.
		 */
		DynamicRegistration bool `json:"dynamicRegistration,omitempty"`

		/**
		 * The client supports the following `CompletionItem` specific
		 * capabilities.
		 */
		CompletionItem struct {
			/**
			 * Client supports snippets as insert text.
			 *
			 * A snippet can define tab stops and placeholders with `$1`, `$2`
			 * and `${3:foo}`. `$0` defines the final tab stop, it defaults to
			 * the end of the snippet. Placeholders with equal identifiers are linked,
			 * that is typing in one will update others too.
			 */
			SnippetSupport bool `json:"snippetSupport,omitempty"`

			/**
			 * Client supports commit characters on a completion item.
			 */
			CommitCharactersSupport bool `json:"commitCharactersSupport,omitempty"`

			/**
			 * Client supports the follow content formats for the documentation
			 * property. The order describes the preferred format of the client.
			 */
			DocumentationFormat []MarkupKind `json:"documentationFormat,omitempty"`

			/**
			 * Client supports the deprecated property on a completion item.
			 */
			DeprecatedSupport bool `json:"deprecatedSupport,omitempty"`

			/**
			 * Client supports the preselect property on a completion item.
			 */
			PreselectSupport bool `json:"preselectSupport,omitempty"`
		} `json:"completionItem,omitempty"`

		CompletionItemKind struct {
			/**
			 * The completion item kind values the client supports. When this
			 * property exists the client also guarantees that it will
			 * handle values outside its set gracefully and falls back
			 * to a default value when unknown.
			 *
			 * If this property is not present the client only supports
			 * the completion items kinds from `Text` to `Reference` as defined in
			 * the initial version of the protocol.
			 */
			ValueSet []CompletionItemKind `json:"valueSet,omitempty"`
		} `json:"completionItemKind,omitempty"`

		/**
		 * The client supports to send additional context information for a
		 * `textDocument/completion` request.
		 */
		ContextSupport bool `json:"contextSupport,omitempty"`
	} `json:"completion"`

	/**
	 * Capabilities specific to the `textDocument/hover`
	 */
	Hover struct {
		/**
		 * Whether hover supports dynamic registration.
		 */
		DynamicRegistration bool `json:"dynamicRegistration,omitempty"`

		/**
		 * Client supports the follow content formats for the content
		 * property. The order describes the preferred format of the client.
		 */
		ContentFormat []MarkupKind `json:"contentFormat,omitempty"`
	} `json:"hover,omitempty"`

	/**
	 * Capabilities specific to the `textDocument/signatureHelp`
	 */
	SignatureHelp struct {
		/**
		 * Whether signature help supports dynamic registration.
		 */
		DynamicRegistration bool `json:"dynamicRegistration,omitempty"`

		/**
		 * The client supports the following `SignatureInformation`
		 * specific properties.
		 */
		SignatureInformation struct {
			/**
			 * Client supports the follow content formats for the documentation
			 * property. The order describes the preferred format of the client.
			 */
			DocumentationFormat []MarkupKind `json:"documentationFormat,omitempty"`
		} `json:"signatureInformation,omitempty"`
	} `json:"signatureHelp,omitempty"`

	/**
	 * Capabilities specific to the `textDocument/references`
	 */
	References struct {
		/**
		 * Whether references supports dynamic registration.
		 */
		DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
	} `json:"references,omitempty"`

	/**
	 * Capabilities specific to the `textDocument/documentHighlight`
	 */
	DocumentHighlight struct {
		/**
		 * Whether document highlight supports dynamic registration.
		 */
		DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
	} `json:"documentHighlight,omitempty"`

	/**
	 * Capabilities specific to the `textDocument/documentSymbol`
	 */
	DocumentSymbol struct {
		/**
		 * Whether document symbol supports dynamic registration.
		 */
		DynamicRegistration bool `json:"dynamicRegistration,omitempty"`

		/**
		 * Specific capabilities for the `SymbolKind`.
		 */
		SymbolKind struct {
			/**
			 * The symbol kind values the client supports. When this
			 * property exists the client also guarantees that it will
			 * handle values outside its set gracefully and falls back
			 * to a default value when unknown.
			 *
			 * If this property is not present the client only supports
			 * the symbol kinds from `File` to `Array` as defined in
			 * the initial version of the protocol.
			 */
			ValueSet []SymbolKind `json:"valueSet,omitempty"`
		} `json:"symbolKind,omitempty"`

		/**
		 * The client support hierarchical document symbols.
		 */
		HierarchicalDocumentSymbolSupport bool `json:"hierarchicalDocumentSymbolSupport,omitempty"`
	} `json:"documentSymbol,omitempty"`

	/**
	 * Capabilities specific to the `textDocument/formatting`
	 */
	Formatting struct {
		/**
		 * Whether formatting supports dynamic registration.
		 */
		DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
	} `json:"formatting,omitempty"`

	/**
	 * Capabilities specific to the `textDocument/rangeFormatting`
	 */
	RangeFormatting struct {
		/**
		 * Whether range formatting supports dynamic registration.
		 */
		DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
	} `json:"rangeFormatting,omitempty"`

	/**
	 * Capabilities specific to the `textDocument/onTypeFormatting`
	 */
	OnTypeFormatting struct {
		/**
		 * Whether on type formatting supports dynamic registration.
		 */
		DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
	} `json:"onTypeFormatting,omitempty"`

	/**
	 * Capabilities specific to the `textDocument/definition`
	 */
	Definition struct {
		/**
		 * Whether definition supports dynamic registration.
		 */
		DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
	} `json:"definition,omitempty"`

	/**
	 * Capabilities specific to the `textDocument/typeDefinition`
	 *
	 * Since 3.6.0
	 */
	TypeDefinition struct {
		/**
		 * Whether typeDefinition supports dynamic registration. If this is set to `true`
		 * the client supports the new `(TextDocumentRegistrationOptions & StaticRegistrationOptions)`
		 * return value for the corresponding server capability as well.
		 */
		DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
	} `json:"typeDefinition,omitempty"`

	/**
	 * Capabilities specific to the `textDocument/implementation`.
	 *
	 * Since 3.6.0
	 */
	Implementation struct {
		/**
		 * Whether implementation supports dynamic registration. If this is set to `true`
		 * the client supports the new `(TextDocumentRegistrationOptions & StaticRegistrationOptions)`
		 * return value for the corresponding server capability as well.
		 */
		DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
	} `json:"implementation,omitempty"`

	/**
	 * Capabilities specific to the `textDocument/codeAction`
	 */
	CodeAction struct {
		/**
		 * Whether code action supports dynamic registration.
		 */
		DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
		/**
		 * The client support code action literals as a valid
		 * response of the `textDocument/codeAction` request.
		 *
		 * Since 3.8.0
		 */
		CodeActionLiteralSupport struct {
			/**
			 * The code action kind is support with the following value
			 * set.
			 */
			CodeActionKind struct {

				/**
				 * The code action kind values the client supports. When this
				 * property exists the client also guarantees that it will
				 * handle values outside its set gracefully and falls back
				 * to a default value when unknown.
				 */
				ValueSet []CodeActionKind `json:"valueSet"`
			} `json:"codeActionKind"`
		} `json:"codeActionLiteralSupport,omitempty"`
	} `json:"codeAction,omitempty"`

	/**
	 * Capabilities specific to the `textDocument/codeLens`
	 */
	CodeLens struct {
		/**
		 * Whether code lens supports dynamic registration.
		 */
		DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
	} `json:"codeLens,omitempty"`

	/**
	 * Capabilities specific to the `textDocument/documentLink`
	 */
	DocumentLink struct {
		/**
		 * Whether document link supports dynamic registration.
		 */
		DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
	} `json:"documentLink,omitempty"`

	/**
	 * Capabilities specific to the `textDocument/documentColor` and the
	 * `textDocument/colorPresentation` request.
	 *
	 * Since 3.6.0
	 */
	ColorProvider struct {
		/**
		 * Whether colorProvider supports dynamic registration. If this is set to `true`
		 * the client supports the new `(ColorProviderOptions & TextDocumentRegistrationOptions & StaticRegistrationOptions)`
		 * return value for the corresponding server capability as well.
		 */
		DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
	} `json:"colorProvider,omitempty"`

	/**
	 * Capabilities specific to the `textDocument/rename`
	 */
	Rename struct {
		/**
		 * Whether rename supports dynamic registration.
		 */
		DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
	} `json:"rename,omitempty"`

	/**
	 * Capabilities specific to `textDocument/publishDiagnostics`.
	 */
	PublishDiagnostics struct {
		/**
		 * Whether the clients accepts diagnostics with related information.
		 */
		RelatedInformation bool `json:"relatedInformation,omitempty"`
	} `json:"publishDiagnostics,omitempty"`

	/**
	 * Capabilities specific to `textDocument/foldingRange` requests.
	 *
	 * Since 3.10.0
	 */
	FoldingRange struct {
		/**
		 * Whether implementation supports dynamic registration for folding range providers. If this is set to `true`
		 * the client supports the new `(FoldingRangeProviderOptions & TextDocumentRegistrationOptions & StaticRegistrationOptions)`
		 * return value for the corresponding server capability as well.
		 */
		DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
		/**
		 * The maximum number of folding ranges that the client prefers to receive per document. The value serves as a
		 * hint, servers are free to follow the limit.
		 */
		RangeLimit float64 `json:"rangeLimit,omitempty"`
		/**
		 * If set, the client signals that it only supports folding complete lines. If set, client will
		 * ignore specified `startCharacter` and `endCharacter` properties in a FoldingRange.
		 */
		LineFoldingOnly bool `json:"lineFoldingOnly,omitempty"`
	}
}

// ClientCapabilities now define capabilities for dynamic registration, workspace
// and text document features the client supports. The experimental can be used to
// pass experimental capabilities under development. For future compatibility a
// ClientCapabilities object literal can have more properties set than currently
// defined. Servers receiving a ClientCapabilities object literal with unknown
// properties should ignore these properties. A missing property should be
// interpreted as an absence of the capability. If a property is missing that
// defines sub properties all sub properties should be interpreted as an absence
// of the capability.
//
// Client capabilities got introduced with version 3.0 of the protocol. They
// therefore only describe capabilities that got introduced in 3.x or later.
// Capabilities that existed in the 2.x version of the protocol are still
// mandatory for clients. Clients cannot opt out of providing them. So even if a
// client omits the ClientCapabilities.textDocument.synchronization it is still
// required that the client provides text document synchronization (e.g. open,
// changed and close notifications).
type ClientCapabilities struct {
	/**
	 * Workspace specific client capabilities.
	 */
	Workspace WorkspaceClientCapabilities `json:"workspace,omitempty"`

	/**
	 * Text document specific client capabilities.
	 */
	TextDocument TextDocumentClientCapabilities `json:"textDocument,omitempty"`

	/**
	 * Experimental client capabilities.
	 */
	Experimental interface{} `json:"experimental,omitempty"`
}

type InitializeResult struct {
	/**
	 * The capabilities the language server provides.
	 */
	Capabilities ServerCapabilities `json:"capabilities"`
}

/**
 * Defines how the host (editor) should sync document changes to the language server.
 */
type TextDocumentSyncKind float64

const (
	/**
	 * Documents should not be synced at all.
	 */
	None TextDocumentSyncKind = 0

	/**
	 * Documents are synced by always sending the full content
	 * of the document.
	 */
	Full TextDocumentSyncKind = 1

	/**
	 * Documents are synced by sending the full content on open.
	 * After that only incremental updates to the document are
	 * send.
	 */
	Incremental TextDocumentSyncKind = 2
)

/**
 * Completion options.
 */
type CompletionOptions struct {
	/**
	 * The server provides support to resolve additional
	 * information for a completion item.
	 */
	ResolveProvider bool `json:"resolveProvider,omitempty"`

	/**
	 * The characters that trigger completion automatically.
	 */
	TriggerCharacters []string `json:"triggerCharacters,omitempty"`
}

/**
 * Signature help options.
 */
type SignatureHelpOptions struct {
	/**
	 * The characters that trigger signature help
	 * automatically.
	 */
	TriggerCharacters []string `json:"triggerCharacters,omitempty"`
}

/**
 * Code Lens options.
 */
type CodeLensOptions struct {
	/**
	 * Code lens has a resolve provider as well.
	 */
	ResolveProvider bool `json:"resolveProvider,omitempty"`
}

/**
 * Format document on type options.
 */
type DocumentOnTypeFormattingOptions struct {
	/**
	 * A character on which formatting should be triggered, like `}`.
	 */
	FirstTriggerCharacter string `json:"firstTriggerCharacter"`

	/**
	 * More trigger characters.
	 */
	MoreTriggerCharacter []string `json:"moreTriggerCharacter,omitempty"`
}

/**
 * Document link options.
 */
type DocumentLinkOptions struct {
	/**
	 * Document links have a resolve provider as well.
	 */
	ResolveProvider bool `json:"resolveProvider,omitempty"`
}

/**
 * Execute command options.
 */
type ExecuteCommandOptions struct {
	/**
	 * The commands to be executed on the server
	 */
	Commands []string `json:"commands"`
}

/**
 * Save options.
 */
type SaveOptions struct {
	/**
	 * The client is supposed to include the content on save.
	 */
	IncludeText bool `json:"includeText,omitempty"`
}

/**
 * Color provider options.
 */
type ColorProviderOptions struct {
}

/**
 * Folding range provider options.
 */
type FoldingRangeProviderOptions struct {
}

type TextDocumentSyncOptions struct {
	/**
	 * Open and close notifications are sent to the server.
	 */
	OpenClose bool `json:"openClose,omitempty"`
	/**
	 * Change notifications are sent to the server. See TextDocumentSyncKind.None, TextDocumentSyncKind.Full
	 * and TextDocumentSyncKind.Incremental. If omitted it defaults to TextDocumentSyncKind.None.
	 */
	Change float64 `json:"change,omitempty"`
	/**
	 * Will save notifications are sent to the server.
	 */
	WillSave bool `json:"willSave,omitempty"`
	/**
	 * Will save wait until requests are sent to the server.
	 */
	WillSaveWaitUntil bool `json:"willSaveWaitUntil,omitempty"`
	/**
	 * Save notifications are sent to the server.
	 */
	Save SaveOptions `json:"save,omitempty"`
}

/**
 * Static registration options to be returned in the initialize request.
 */
type StaticRegistrationOptions struct {
	/**
	 * The id used to register the request. The id can be used to deregister
	 * the request again. See also Registration#id.
	 */
	ID string `json:"id,omitempty"`
}

type ServerCapabilities struct {
	/**
	 * Defines how text documents are synced. Is either a detailed structure defining each notification or
	 * for backwards compatibility the TextDocumentSyncKind number. If omitted it defaults to `TextDocumentSyncKind.None`.
	 */
	TextDocumentSync interface{} `json:"textDocumentSync,omitempty"` // TextDocumentSyncOptions | number
	/**
	 * The server provides hover support.
	 */
	HoverProvider bool `json:"hoverProvider,omitempty"`
	/**
	 * The server provides completion support.
	 */
	CompletionProvider CompletionOptions `json:"completionProvider,omitempty"`
	/**
	 * The server provides signature help support.
	 */
	SignatureHelpProvider SignatureHelpOptions `json:"signatureHelpProvider,omitempty"`
	/**
	 * The server provides goto definition support.
	 */
	DefinitionProvider bool `json:"definitionProvider,omitempty"`
	/**
	 * The server provides Goto Type Definition support.
	 *
	 * Since 3.6.0
	 */
	TypeDefinitionProvider interface{} `json:"typeDefinitionProvider,omitempty"` // boolean | (TextDocumentRegistrationOptions & StaticRegistrationOptions)
	/**
	 * The server provides Goto Implementation support.
	 *
	 * Since 3.6.0
	 */
	ImplementationProvider interface{} `json:"implementationProvider,omitempty"` // boolean | (TextDocumentRegistrationOptions & StaticRegistrationOptions)
	/**
	 * The server provides find references support.
	 */
	ReferencesProvider bool `json:"referencesProvider,omitempty"`
	/**
	 * The server provides document highlight support.
	 */
	DocumentHighlightProvider bool `json:"documentHighlightProvider,omitempty"`
	/**
	 * The server provides document symbol support.
	 */
	DocumentSymbolProvider bool `json:"documentSymbolProvider,omitempty"`
	/**
	 * The server provides workspace symbol support.
	 */
	WorkspaceSymbolProvider bool `json:"workspaceSymbolProvider,omitempty"`
	/**
	 * The server provides code actions.
	 */
	CodeActionProvider bool `json:"codeActionProvider,omitempty"`
	/**
	 * The server provides code lens.
	 */
	CodeLensProvider CodeLensOptions `json:"codeLensProvider,omitempty"`
	/**
	 * The server provides document formatting.
	 */
	DocumentFormattingProvider bool `json:"documentFormattingProvider,omitempty"`
	/**
	 * The server provides document range formatting.
	 */
	DocumentRangeFormattingProvider bool `json:"documentRangeFormattingProvider,omitempty"`
	/**
	 * The server provides document formatting on typing.
	 */
	DocumentOnTypeFormattingProvider DocumentOnTypeFormattingOptions `json:"documentOnTypeFormattingProvider,omitempty"`
	/**
	 * The server provides rename support.
	 */
	RenameProvider bool `json:"renameProvider,omitempty"`
	/**
	 * The server provides document link support.
	 */
	DocumentLinkProvider DocumentLinkOptions `json:"documentLinkProvider,omitempty"`
	/**
	 * The server provides color provider support.
	 *
	 * Since 3.6.0
	 */
	//TODO: complex union type to decode here
	ColorProvider interface{} `json:"colorProvider,omitempty"` // boolean | ColorProviderOptions | (ColorProviderOptions & TextDocumentRegistrationOptions & StaticRegistrationOptions)
	/**
	 * The server provides folding provider support.
	 *
	 * Since 3.10.0
	 */
	//TODO: complex union type to decode here
	FoldingRangeProvider interface{} `json:"foldingRangeProvider,omitempty"` // boolean | FoldingRangeProviderOptions | (FoldingRangeProviderOptions & TextDocumentRegistrationOptions & StaticRegistrationOptions)
	/**
	 * The server provides execute command support.
	 */
	ExecuteCommandProvider ExecuteCommandOptions `json:"executeCommandProvider,omitempty"`
	/**
	 * Workspace specific server capabilities
	 */
	Workspace struct {
		/**
		 * The server supports workspace folder.
		 *
		 * Since 3.6.0
		 */
		WorkspaceFolders struct {
			/**
			* The server has support for workspace folders
			 */
			Supported bool `json:"supported,omitempty"`
			/**
			* Whether the server wants to receive workspace folder
			* change notifications.
			*
			* If a strings is provided the string is treated as a ID
			* under which the notification is registered on the client
			* side. The ID can be used to unregister for these events
			* using the `client/unregisterCapability` request.
			 */
			ChangeNotifications interface{} `json:"changeNotifications,omitempty"` // string | boolean
		} `json:"workspaceFolders,omitempty"`
	} `json:"workspace,omitempty"`
	/**
	 * Experimental server capabilities.
	 */
	Experimental interface{} `json:"experimental,omitempty"`
}

type InitializedParams struct {
}
