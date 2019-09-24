// Package protocol contains data types and code for LSP jsonrpcs
// generated automatically from vscode-languageserver-node
// commit: 36ac51f057215e6e2e0408384e07ecf564a938da
// last fetched Tue Sep 24 2019 17:44:28 GMT-0400 (Eastern Daylight Time)
package protocol

// Code generated (see typescript/README.md) DO NOT EDIT.

/*ImplementationClientCapabilities defined:
 * Since 3.6.0
 */
type ImplementationClientCapabilities struct {

	/*DynamicRegistration defined:
	 * Whether implementation supports dynamic registration. If this is set to `true`
	 * the client supports the new `ImplementationRegistrationOptions` return value
	 * for the corresponding server capability as well.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`

	/*LinkSupport defined:
	 * The client supports additional metadata in the form of definition links.
	 *
	 * Since 3.14.0
	 */
	LinkSupport bool `json:"linkSupport,omitempty"`
}

// ImplementationOptions is
type ImplementationOptions struct {
	WorkDoneProgressOptions
}

// ImplementationRegistrationOptions is
type ImplementationRegistrationOptions struct {
	TextDocumentRegistrationOptions
	ImplementationOptions
	StaticRegistrationOptions
}

// ImplementationParams is
type ImplementationParams struct {
	TextDocumentPositionParams
	WorkDoneProgressParams
	PartialResultParams
}

/*TypeDefinitionClientCapabilities defined:
 * Since 3.6.0
 */
type TypeDefinitionClientCapabilities struct {

	/*DynamicRegistration defined:
	 * Whether implementation supports dynamic registration. If this is set to `true`
	 * the client supports the new `TypeDefinitionRegistrationOptions` return value
	 * for the corresponding server capability as well.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`

	/*LinkSupport defined:
	 * The client supports additional metadata in the form of definition links.
	 *
	 * Since 3.14.0
	 */
	LinkSupport bool `json:"linkSupport,omitempty"`
}

// TypeDefinitionOptions is
type TypeDefinitionOptions struct {
	WorkDoneProgressOptions
}

// TypeDefinitionRegistrationOptions is
type TypeDefinitionRegistrationOptions struct {
	TextDocumentRegistrationOptions
	TypeDefinitionOptions
	StaticRegistrationOptions
}

// TypeDefinitionParams is
type TypeDefinitionParams struct {
	TextDocumentPositionParams
	WorkDoneProgressParams
	PartialResultParams
}

// WorkspaceFoldersInitializeParams is
type WorkspaceFoldersInitializeParams struct {

	/*WorkspaceFolders defined:
	 * The actual configured workspace folders.
	 */
	WorkspaceFolders []WorkspaceFolder `json:"workspaceFolders"`
}

// WorkspaceFoldersClientCapabilities is
type WorkspaceFoldersClientCapabilities struct {

	/*Workspace defined:
	 * The workspace client capabilities
	 */
	Workspace *struct {

		/*WorkspaceFolders defined:
		 * The client has support for workspace folders
		 */
		WorkspaceFolders bool `json:"workspaceFolders,omitempty"`
	} `json:"workspace,omitempty"`
}

// WorkspaceFoldersServerCapabilities is
type WorkspaceFoldersServerCapabilities struct {

	/*Workspace defined:
	 * The workspace server capabilities
	 */
	Workspace *struct {

		// WorkspaceFolders is
		WorkspaceFolders *struct {

			/*Supported defined:
			 * The Server has support for workspace folders
			 */
			Supported bool `json:"supported,omitempty"`

			/*ChangeNotifications defined:
			 * Whether the server wants to receive workspace folder
			 * change notifications.
			 *
			 * If a strings is provided the string is treated as a ID
			 * under which the notification is registed on the client
			 * side. The ID can be used to unregister for these events
			 * using the `client/unregisterCapability` request.
			 */
			ChangeNotifications string `json:"changeNotifications,omitempty"` // string | boolean
		} `json:"workspaceFolders,omitempty"`
	} `json:"workspace,omitempty"`
}

// WorkspaceFolder is
type WorkspaceFolder struct {

	/*URI defined:
	 * The associated URI for this workspace folder.
	 */
	URI string `json:"uri"`

	/*Name defined:
	 * The name of the workspace folder. Used to refer to this
	 * workspace folder in thge user interface.
	 */
	Name string `json:"name"`
}

/*DidChangeWorkspaceFoldersParams defined:
 * The parameters of a `workspace/didChangeWorkspaceFolders` notification.
 */
type DidChangeWorkspaceFoldersParams struct {

	/*Event defined:
	 * The actual workspace folder change event.
	 */
	Event WorkspaceFoldersChangeEvent `json:"event"`
}

/*WorkspaceFoldersChangeEvent defined:
 * The workspace folder change event.
 */
type WorkspaceFoldersChangeEvent struct {

	/*Added defined:
	 * The array of added workspace folders
	 */
	Added []WorkspaceFolder `json:"added"`

	/*Removed defined:
	 * The array of the removed workspace folders
	 */
	Removed []WorkspaceFolder `json:"removed"`
}

// ConfigurationClientCapabilities is
type ConfigurationClientCapabilities struct {

	/*Workspace defined:
	 * The workspace client capabilities
	 */
	Workspace *struct {

		/*Configuration defined:
		* The client supports `workspace/configuration` requests.
		 */
		Configuration bool `json:"configuration,omitempty"`
	} `json:"workspace,omitempty"`
}

// ConfigurationItem is
type ConfigurationItem struct {

	/*ScopeURI defined:
	 * The scope to get the configuration section for.
	 */
	ScopeURI string `json:"scopeUri,omitempty"`

	/*Section defined:
	 * The configuration section asked for.
	 */
	Section string `json:"section,omitempty"`
}

/*ConfigurationParams defined:
 * The parameters of a configuration request.
 */
type ConfigurationParams struct {

	// Items is
	Items []ConfigurationItem `json:"items"`
}

// DocumentColorClientCapabilities is
type DocumentColorClientCapabilities struct {

	/*DynamicRegistration defined:
	 * Whether implementation supports dynamic registration. If this is set to `true`
	 * the client supports the new `DocumentColorRegistrationOptions` return value
	 * for the corresponding server capability as well.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
}

// DocumentColorOptions is
type DocumentColorOptions struct {

	/*ResolveProvider defined:
	 * Code lens has a resolve provider as well.
	 */
	ResolveProvider bool `json:"resolveProvider,omitempty"`
	WorkDoneProgressOptions
}

// DocumentColorRegistrationOptions is
type DocumentColorRegistrationOptions struct {
	TextDocumentRegistrationOptions
	StaticRegistrationOptions
	DocumentColorOptions
}

/*DocumentColorParams defined:
 * Parameters for a [DocumentColorRequest](#DocumentColorRequest).
 */
type DocumentColorParams struct {

	/*TextDocument defined:
	 * The text document.
	 */
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	WorkDoneProgressParams
	PartialResultParams
}

/*ColorPresentationParams defined:
 * Parameters for a [ColorPresentationRequest](#ColorPresentationRequest).
 */
type ColorPresentationParams struct {

	/*TextDocument defined:
	 * The text document.
	 */
	TextDocument TextDocumentIdentifier `json:"textDocument"`

	/*Color defined:
	 * The color to request presentations for.
	 */
	Color Color `json:"color"`

	/*Range defined:
	 * The range where the color would be inserted. Serves as a context.
	 */
	Range Range `json:"range"`
	WorkDoneProgressParams
	PartialResultParams
}

// FoldingRangeClientCapabilities is
type FoldingRangeClientCapabilities struct {

	/*DynamicRegistration defined:
	 * Whether implementation supports dynamic registration for folding range providers. If this is set to `true`
	 * the client supports the new `FoldingRangeRegistrationOptions` return value for the corresponding server
	 * capability as well.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`

	/*RangeLimit defined:
	 * The maximum number of folding ranges that the client prefers to receive per document. The value serves as a
	 * hint, servers are free to follow the limit.
	 */
	RangeLimit float64 `json:"rangeLimit,omitempty"`

	/*LineFoldingOnly defined:
	 * If set, the client signals that it only supports folding complete lines. If set, client will
	 * ignore specified `startCharacter` and `endCharacter` properties in a FoldingRange.
	 */
	LineFoldingOnly bool `json:"lineFoldingOnly,omitempty"`
}

// FoldingRangeOptions is
type FoldingRangeOptions struct {
	WorkDoneProgressOptions
}

// FoldingRangeRegistrationOptions is
type FoldingRangeRegistrationOptions struct {
	TextDocumentRegistrationOptions
	FoldingRangeOptions
	StaticRegistrationOptions
}

/*FoldingRange defined:
 * Represents a folding range.
 */
type FoldingRange struct {

	/*StartLine defined:
	 * The zero-based line number from where the folded range starts.
	 */
	StartLine float64 `json:"startLine"`

	/*StartCharacter defined:
	 * The zero-based character offset from where the folded range starts. If not defined, defaults to the length of the start line.
	 */
	StartCharacter float64 `json:"startCharacter,omitempty"`

	/*EndLine defined:
	 * The zero-based line number where the folded range ends.
	 */
	EndLine float64 `json:"endLine"`

	/*EndCharacter defined:
	 * The zero-based character offset before the folded range ends. If not defined, defaults to the length of the end line.
	 */
	EndCharacter float64 `json:"endCharacter,omitempty"`

	/*Kind defined:
	 * Describes the kind of the folding range such as `comment' or 'region'. The kind
	 * is used to categorize folding ranges and used by commands like 'Fold all comments'. See
	 * [FoldingRangeKind](#FoldingRangeKind) for an enumeration of standardized kinds.
	 */
	Kind string `json:"kind,omitempty"`
}

/*FoldingRangeParams defined:
 * Parameters for a [FoldingRangeRequest](#FoldingRangeRequest).
 */
type FoldingRangeParams struct {

	/*TextDocument defined:
	 * The text document.
	 */
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	WorkDoneProgressParams
	PartialResultParams
}

/*DeclarationClientCapabilities defined:
 * Since 3.14.0
 */
type DeclarationClientCapabilities struct {

	/*DynamicRegistration defined:
	 * Whether declaration supports dynamic registration. If this is set to `true`
	 * the client supports the new `DeclarationRegistrationOptions` return value
	 * for the corresponding server capability as well.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`

	/*LinkSupport defined:
	 * The client supports additional metadata in the form of declaration links.
	 */
	LinkSupport bool `json:"linkSupport,omitempty"`
}

// DeclarationOptions is
type DeclarationOptions struct {
	WorkDoneProgressOptions
}

// DeclarationRegistrationOptions is
type DeclarationRegistrationOptions struct {
	DeclarationOptions
	TextDocumentRegistrationOptions
	StaticRegistrationOptions
}

// DeclarationParams is
type DeclarationParams struct {
	TextDocumentPositionParams
	WorkDoneProgressParams
	PartialResultParams
}

// SelectionRangeClientCapabilities is
type SelectionRangeClientCapabilities struct {

	/*DynamicRegistration defined:
	 * Whether implementation supports dynamic registration for selection range providers. If this is set to `true`
	 * the client supports the new `SelectionRangeRegistrationOptions` return value for the corresponding server
	 * capability as well.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
}

// SelectionRangeOptions is
type SelectionRangeOptions struct {
	WorkDoneProgressOptions
}

// SelectionRangeRegistrationOptions is
type SelectionRangeRegistrationOptions struct {
	SelectionRangeOptions
	TextDocumentRegistrationOptions
	StaticRegistrationOptions
}

/*SelectionRangeParams defined:
 * A parameter literal used in selection range requests.
 */
type SelectionRangeParams struct {

	/*TextDocument defined:
	 * The text document.
	 */
	TextDocument TextDocumentIdentifier `json:"textDocument"`

	/*Positions defined:
	 * The positions inside the text document.
	 */
	Positions []Position `json:"positions"`
	WorkDoneProgressParams
	PartialResultParams
}

/*Registration defined:
 * General parameters to to register for an notification or to register a provider.
 */
type Registration struct {

	/*ID defined:
	 * The id used to register the request. The id can be used to deregister
	 * the request again.
	 */
	ID string `json:"id"`

	/*Method defined:
	 * The method to register for.
	 */
	Method string `json:"method"`

	/*RegisterOptions defined:
	 * Options necessary for the registration.
	 */
	RegisterOptions interface{} `json:"registerOptions,omitempty"`
}

// RegistrationParams is
type RegistrationParams struct {

	// Registrations is
	Registrations []Registration `json:"registrations"`
}

/*Unregistration defined:
 * General parameters to unregister a request or notification.
 */
type Unregistration struct {

	/*ID defined:
	 * The id used to unregister the request or notification. Usually an id
	 * provided during the register request.
	 */
	ID string `json:"id"`

	/*Method defined:
	 * The method to unregister for.
	 */
	Method string `json:"method"`
}

// UnregistrationParams is
type UnregistrationParams struct {

	// Unregisterations is
	Unregisterations []Unregistration `json:"unregisterations"`
}

// WorkDoneProgressParams is
type WorkDoneProgressParams struct {

	/*WorkDoneToken defined:
	 * An optional token that a server can use to report work done progress.
	 */
	WorkDoneToken *ProgressToken `json:"workDoneToken,omitempty"`
}

// PartialResultParams is
type PartialResultParams struct {

	/*PartialResultToken defined:
	 * An optional token that a server can use to report partial results (e.g. streaming) to
	 * the client.
	 */
	PartialResultToken *ProgressToken `json:"partialResultToken,omitempty"`
}

/*TextDocumentPositionParams defined:
 * A parameter literal used in requests to pass a text document and a position inside that
 * document.
 */
type TextDocumentPositionParams struct {

	/*TextDocument defined:
	 * The text document.
	 */
	TextDocument TextDocumentIdentifier `json:"textDocument"`

	/*Position defined:
	 * The position inside the text document.
	 */
	Position Position `json:"position"`
}

/*WorkspaceClientCapabilities defined:
 * Workspace specific client capabilities.
 */
type WorkspaceClientCapabilities struct {

	/*ApplyEdit defined:
	 * The client supports applying batch edits
	 * to the workspace by supporting the request
	 * 'workspace/applyEdit'
	 */
	ApplyEdit bool `json:"applyEdit,omitempty"`

	/*WorkspaceEdit defined:
	 * Capabilities specific to `WorkspaceEdit`s
	 */
	WorkspaceEdit *WorkspaceEditClientCapabilities `json:"workspaceEdit,omitempty"`

	/*DidChangeConfiguration defined:
	 * Capabilities specific to the `workspace/didChangeConfiguration` notification.
	 */
	DidChangeConfiguration *DidChangeConfigurationClientCapabilities `json:"didChangeConfiguration,omitempty"`

	/*DidChangeWatchedFiles defined:
	 * Capabilities specific to the `workspace/didChangeWatchedFiles` notification.
	 */
	DidChangeWatchedFiles *DidChangeWatchedFilesClientCapabilities `json:"didChangeWatchedFiles,omitempty"`

	/*Symbol defined:
	 * Capabilities specific to the `workspace/symbol` request.
	 */
	Symbol *WorkspaceSymbolClientCapabilities `json:"symbol,omitempty"`

	/*ExecuteCommand defined:
	 * Capabilities specific to the `workspace/executeCommand` request.
	 */
	ExecuteCommand *ExecuteCommandClientCapabilities `json:"executeCommand,omitempty"`
}

/*TextDocumentClientCapabilities defined:
 * Text document specific client capabilities.
 */
type TextDocumentClientCapabilities struct {

	/*Synchronization defined:
	 * Defines which synchronization capabilities the client supports.
	 */
	Synchronization *TextDocumentSyncClientCapabilities `json:"synchronization,omitempty"`

	/*Completion defined:
	 * Capabilities specific to the `textDocument/completion`
	 */
	Completion *CompletionClientCapabilities `json:"completion,omitempty"`

	/*Hover defined:
	 * Capabilities specific to the `textDocument/hover`
	 */
	Hover *HoverClientCapabilities `json:"hover,omitempty"`

	/*SignatureHelp defined:
	 * Capabilities specific to the `textDocument/signatureHelp`
	 */
	SignatureHelp *SignatureHelpClientCapabilities `json:"signatureHelp,omitempty"`

	/*Declaration defined:
	 * Capabilities specific to the `textDocument/declaration`
	 *
	 * @since 3.14.0
	 */
	Declaration *DeclarationClientCapabilities `json:"declaration,omitempty"`

	/*Definition defined:
	 * Capabilities specific to the `textDocument/definition`
	 */
	Definition *DefinitionClientCapabilities `json:"definition,omitempty"`

	/*TypeDefinition defined:
	 * Capabilities specific to the `textDocument/typeDefinition`
	 *
	 * @since 3.6.0
	 */
	TypeDefinition *TypeDefinitionClientCapabilities `json:"typeDefinition,omitempty"`

	/*Implementation defined:
	 * Capabilities specific to the `textDocument/implementation`
	 *
	 * @since 3.6.0
	 */
	Implementation *ImplementationClientCapabilities `json:"implementation,omitempty"`

	/*References defined:
	 * Capabilities specific to the `textDocument/references`
	 */
	References *ReferenceClientCapabilities `json:"references,omitempty"`

	/*DocumentHighlight defined:
	 * Capabilities specific to the `textDocument/documentHighlight`
	 */
	DocumentHighlight *DocumentHighlightClientCapabilities `json:"documentHighlight,omitempty"`

	/*DocumentSymbol defined:
	 * Capabilities specific to the `textDocument/documentSymbol`
	 */
	DocumentSymbol *DocumentSymbolClientCapabilities `json:"documentSymbol,omitempty"`

	/*CodeAction defined:
	 * Capabilities specific to the `textDocument/codeAction`
	 */
	CodeAction *CodeActionClientCapabilities `json:"codeAction,omitempty"`

	/*CodeLens defined:
	 * Capabilities specific to the `textDocument/codeLens`
	 */
	CodeLens *CodeLensClientCapabilities `json:"codeLens,omitempty"`

	/*DocumentLink defined:
	 * Capabilities specific to the `textDocument/documentLink`
	 */
	DocumentLink *DocumentLinkClientCapabilities `json:"documentLink,omitempty"`

	/*ColorProvider defined:
	 * Capabilities specific to the `textDocument/documentColor`
	 */
	ColorProvider *DocumentColorClientCapabilities `json:"colorProvider,omitempty"`

	/*Formatting defined:
	 * Capabilities specific to the `textDocument/formatting`
	 */
	Formatting *DocumentFormattingClientCapabilities `json:"formatting,omitempty"`

	/*RangeFormatting defined:
	 * Capabilities specific to the `textDocument/rangeFormatting`
	 */
	RangeFormatting *DocumentRangeFormattingClientCapabilities `json:"rangeFormatting,omitempty"`

	/*OnTypeFormatting defined:
	 * Capabilities specific to the `textDocument/onTypeFormatting`
	 */
	OnTypeFormatting *DocumentOnTypeFormattingClientCapabilities `json:"onTypeFormatting,omitempty"`

	/*Rename defined:
	 * Capabilities specific to the `textDocument/rename`
	 */
	Rename *RenameClientCapabilities `json:"rename,omitempty"`

	/*FoldingRange defined:
	 * Capabilities specific to `textDocument/foldingRange` requests.
	 *
	 * @since 3.10.0
	 */
	FoldingRange *FoldingRangeClientCapabilities `json:"foldingRange,omitempty"`

	/*SelectionRange defined:
	 * Capabilities specific to `textDocument/selectionRange` requests
	 *
	 * @since 3.15.0
	 */
	SelectionRange *SelectionRangeClientCapabilities `json:"selectionRange,omitempty"`

	/*PublishDiagnostics defined:
	 * Capabilities specific to `textDocument/publishDiagnostics`.
	 */
	PublishDiagnostics *PublishDiagnosticsClientCapabilities `json:"publishDiagnostics,omitempty"`
}

/*InnerClientCapabilities defined:
 * Defines the capabilities provided by the client.
 */
type InnerClientCapabilities struct {

	/*Workspace defined:
	 * Workspace specific client capabilities.
	 */
	Workspace *WorkspaceClientCapabilities `json:"workspace,omitempty"`

	/*TextDocument defined:
	 * Text document specific client capabilities.
	 */
	TextDocument *TextDocumentClientCapabilities `json:"textDocument,omitempty"`

	/*Window defined:
	 * Window specific client capabilities.
	 */
	Window interface{} `json:"window,omitempty"`

	/*Experimental defined:
	 * Experimental client capabilities.
	 */
	Experimental interface{} `json:"experimental,omitempty"`
}

// ClientCapabilities is
type ClientCapabilities struct {

	/*Workspace defined:
	 * Workspace specific client capabilities.
	 */
	Workspace struct {

		/*ApplyEdit defined:
		 * The client supports applying batch edits
		 * to the workspace by supporting the request
		 * 'workspace/applyEdit'
		 */
		ApplyEdit bool `json:"applyEdit,omitempty"`

		/*WorkspaceEdit defined:
		 * Capabilities specific to `WorkspaceEdit`s
		 */
		WorkspaceEdit WorkspaceEditClientCapabilities `json:"workspaceEdit,omitempty"`

		/*DidChangeConfiguration defined:
		 * Capabilities specific to the `workspace/didChangeConfiguration` notification.
		 */
		DidChangeConfiguration DidChangeConfigurationClientCapabilities `json:"didChangeConfiguration,omitempty"`

		/*DidChangeWatchedFiles defined:
		 * Capabilities specific to the `workspace/didChangeWatchedFiles` notification.
		 */
		DidChangeWatchedFiles DidChangeWatchedFilesClientCapabilities `json:"didChangeWatchedFiles,omitempty"`

		/*Symbol defined:
		 * Capabilities specific to the `workspace/symbol` request.
		 */
		Symbol WorkspaceSymbolClientCapabilities `json:"symbol,omitempty"`

		/*ExecuteCommand defined:
		 * Capabilities specific to the `workspace/executeCommand` request.
		 */
		ExecuteCommand ExecuteCommandClientCapabilities `json:"executeCommand,omitempty"`

		/*WorkspaceFolders defined:
		 * The client has support for workspace folders
		 */
		WorkspaceFolders bool `json:"workspaceFolders,omitempty"`

		/*Configuration defined:
		* The client supports `workspace/configuration` requests.
		 */
		Configuration bool `json:"configuration,omitempty"`
	} `json:"workspace,omitempty"`

	/*TextDocument defined:
	 * Text document specific client capabilities.
	 */
	TextDocument TextDocumentClientCapabilities `json:"textDocument,omitempty"`

	/*Window defined:
	 * Window specific client capabilities.
	 */
	Window interface{} `json:"window,omitempty"`

	/*Experimental defined:
	 * Experimental client capabilities.
	 */
	Experimental interface{} `json:"experimental,omitempty"`

	/*DynamicRegistration defined:
	 * Whether implementation supports dynamic registration for selection range providers. If this is set to `true`
	 * the client supports the new `SelectionRangeRegistrationOptions` return value for the corresponding server
	 * capability as well.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
}

/*StaticRegistrationOptions defined:
 * Static registration options to be returned in the initialize
 * request.
 */
type StaticRegistrationOptions struct {

	/*ID defined:
	 * The id used to register the request. The id can be used to deregister
	 * the request again. See also Registration#id.
	 */
	ID string `json:"id,omitempty"`
}

/*TextDocumentRegistrationOptions defined:
 * General text document registration options.
 */
type TextDocumentRegistrationOptions struct {

	/*DocumentSelector defined:
	 * A document selector to identify the scope of the registration. If set to null
	 * the document selector provided on the client side will be used.
	 */
	DocumentSelector DocumentSelector `json:"documentSelector"`
}

/*SaveOptions defined:
 * Save options.
 */
type SaveOptions struct {

	/*IncludeText defined:
	 * The client is supposed to include the content on save.
	 */
	IncludeText bool `json:"includeText,omitempty"`
}

// WorkDoneProgressOptions is
type WorkDoneProgressOptions struct {

	// WorkDoneProgress is
	WorkDoneProgress bool `json:"workDoneProgress,omitempty"`
}

/*InnerServerCapabilities defined:
 * Defines the capabilities provided by a language
 * server.
 */
type InnerServerCapabilities struct {

	/*TextDocumentSync defined:
	 * Defines how text documents are synced. Is either a detailed structure defining each notification or
	 * for backwards compatibility the TextDocumentSyncKind number.
	 */
	TextDocumentSync interface{} `json:"textDocumentSync,omitempty"` // TextDocumentSyncOptions | TextDocumentSyncKind

	/*CompletionProvider defined:
	 * The server provides completion support.
	 */
	CompletionProvider *CompletionOptions `json:"completionProvider,omitempty"`

	/*HoverProvider defined:
	 * The server provides hover support.
	 */
	HoverProvider bool `json:"hoverProvider,omitempty"` // boolean | HoverOptions

	/*SignatureHelpProvider defined:
	 * The server provides signature help support.
	 */
	SignatureHelpProvider *SignatureHelpOptions `json:"signatureHelpProvider,omitempty"`

	/*DeclarationProvider defined:
	 * The server provides Goto Declaration support.
	 */
	DeclarationProvider bool `json:"declarationProvider,omitempty"` // boolean | DeclarationOptions | DeclarationRegistrationOptions

	/*DefinitionProvider defined:
	 * The server provides goto definition support.
	 */
	DefinitionProvider bool `json:"definitionProvider,omitempty"` // boolean | DefinitionOptions

	/*TypeDefinitionProvider defined:
	 * The server provides Goto Type Definition support.
	 */
	TypeDefinitionProvider bool `json:"typeDefinitionProvider,omitempty"` // boolean | TypeDefinitionOptions | TypeDefinitionRegistrationOptions

	/*ImplementationProvider defined:
	 * The server provides Goto Implementation support.
	 */
	ImplementationProvider bool `json:"implementationProvider,omitempty"` // boolean | ImplementationOptions | ImplementationRegistrationOptions

	/*ReferencesProvider defined:
	 * The server provides find references support.
	 */
	ReferencesProvider bool `json:"referencesProvider,omitempty"` // boolean | ReferenceOptions

	/*DocumentHighlightProvider defined:
	 * The server provides document highlight support.
	 */
	DocumentHighlightProvider bool `json:"documentHighlightProvider,omitempty"` // boolean | DocumentHighlightOptions

	/*DocumentSymbolProvider defined:
	 * The server provides document symbol support.
	 */
	DocumentSymbolProvider bool `json:"documentSymbolProvider,omitempty"` // boolean | DocumentSymbolOptions

	/*CodeActionProvider defined:
	 * The server provides code actions. CodeActionOptions may only be
	 * specified if the client states that it supports
	 * `codeActionLiteralSupport` in its initial `initialize` request.
	 */
	CodeActionProvider interface{} `json:"codeActionProvider,omitempty"` // boolean | CodeActionOptions

	/*CodeLensProvider defined:
	 * The server provides code lens.
	 */
	CodeLensProvider *CodeLensOptions `json:"codeLensProvider,omitempty"`

	/*DocumentLinkProvider defined:
	 * The server provides document link support.
	 */
	DocumentLinkProvider *DocumentLinkOptions `json:"documentLinkProvider,omitempty"`

	/*ColorProvider defined:
	 * The server provides color provider support.
	 */
	ColorProvider bool `json:"colorProvider,omitempty"` // boolean | DocumentColorOptions | DocumentColorRegistrationOptions

	/*WorkspaceSymbolProvider defined:
	 * The server provides workspace symbol support.
	 */
	WorkspaceSymbolProvider bool `json:"workspaceSymbolProvider,omitempty"` // boolean | WorkspaceSymbolOptions

	/*DocumentFormattingProvider defined:
	 * The server provides document formatting.
	 */
	DocumentFormattingProvider bool `json:"documentFormattingProvider,omitempty"` // boolean | DocumentFormattingOptions

	/*DocumentRangeFormattingProvider defined:
	 * The server provides document range formatting.
	 */
	DocumentRangeFormattingProvider bool `json:"documentRangeFormattingProvider,omitempty"` // boolean | DocumentRangeFormattingOptions

	/*DocumentOnTypeFormattingProvider defined:
	 * The server provides document formatting on typing.
	 */
	DocumentOnTypeFormattingProvider *DocumentOnTypeFormattingOptions `json:"documentOnTypeFormattingProvider,omitempty"`

	/*RenameProvider defined:
	 * The server provides rename support. RenameOptions may only be
	 * specified if the client states that it supports
	 * `prepareSupport` in its initial `initialize` request.
	 */
	RenameProvider interface{} `json:"renameProvider,omitempty"` // boolean | RenameOptions

	/*FoldingRangeProvider defined:
	 * The server provides folding provider support.
	 */
	FoldingRangeProvider bool `json:"foldingRangeProvider,omitempty"` // boolean | FoldingRangeOptions | FoldingRangeRegistrationOptions

	/*SelectionRangeProvider defined:
	 * The server provides selection range support.
	 */
	SelectionRangeProvider bool `json:"selectionRangeProvider,omitempty"` // boolean | SelectionRangeOptions | SelectionRangeRegistrationOptions

	/*ExecuteCommandProvider defined:
	 * The server provides execute command support.
	 */
	ExecuteCommandProvider *ExecuteCommandOptions `json:"executeCommandProvider,omitempty"`

	/*Experimental defined:
	 * Experimental server capabilities.
	 */
	Experimental interface{} `json:"experimental,omitempty"`
}

// ServerCapabilities is
type ServerCapabilities struct {

	/*TextDocumentSync defined:
	 * Defines how text documents are synced. Is either a detailed structure defining each notification or
	 * for backwards compatibility the TextDocumentSyncKind number.
	 */
	TextDocumentSync interface{} `json:"textDocumentSync,omitempty"` // TextDocumentSyncOptions | TextDocumentSyncKind

	/*CompletionProvider defined:
	 * The server provides completion support.
	 */
	CompletionProvider *CompletionOptions `json:"completionProvider,omitempty"`

	/*HoverProvider defined:
	 * The server provides hover support.
	 */
	HoverProvider bool `json:"hoverProvider,omitempty"` // boolean | HoverOptions

	/*SignatureHelpProvider defined:
	 * The server provides signature help support.
	 */
	SignatureHelpProvider *SignatureHelpOptions `json:"signatureHelpProvider,omitempty"`

	/*DeclarationProvider defined:
	 * The server provides Goto Declaration support.
	 */
	DeclarationProvider bool `json:"declarationProvider,omitempty"` // boolean | DeclarationOptions | DeclarationRegistrationOptions

	/*DefinitionProvider defined:
	 * The server provides goto definition support.
	 */
	DefinitionProvider bool `json:"definitionProvider,omitempty"` // boolean | DefinitionOptions

	/*TypeDefinitionProvider defined:
	 * The server provides Goto Type Definition support.
	 */
	TypeDefinitionProvider bool `json:"typeDefinitionProvider,omitempty"` // boolean | TypeDefinitionOptions | TypeDefinitionRegistrationOptions

	/*ImplementationProvider defined:
	 * The server provides Goto Implementation support.
	 */
	ImplementationProvider bool `json:"implementationProvider,omitempty"` // boolean | ImplementationOptions | ImplementationRegistrationOptions

	/*ReferencesProvider defined:
	 * The server provides find references support.
	 */
	ReferencesProvider bool `json:"referencesProvider,omitempty"` // boolean | ReferenceOptions

	/*DocumentHighlightProvider defined:
	 * The server provides document highlight support.
	 */
	DocumentHighlightProvider bool `json:"documentHighlightProvider,omitempty"` // boolean | DocumentHighlightOptions

	/*DocumentSymbolProvider defined:
	 * The server provides document symbol support.
	 */
	DocumentSymbolProvider bool `json:"documentSymbolProvider,omitempty"` // boolean | DocumentSymbolOptions

	/*CodeActionProvider defined:
	 * The server provides code actions. CodeActionOptions may only be
	 * specified if the client states that it supports
	 * `codeActionLiteralSupport` in its initial `initialize` request.
	 */
	CodeActionProvider interface{} `json:"codeActionProvider,omitempty"` // boolean | CodeActionOptions

	/*CodeLensProvider defined:
	 * The server provides code lens.
	 */
	CodeLensProvider *CodeLensOptions `json:"codeLensProvider,omitempty"`

	/*DocumentLinkProvider defined:
	 * The server provides document link support.
	 */
	DocumentLinkProvider *DocumentLinkOptions `json:"documentLinkProvider,omitempty"`

	/*ColorProvider defined:
	 * The server provides color provider support.
	 */
	ColorProvider bool `json:"colorProvider,omitempty"` // boolean | DocumentColorOptions | DocumentColorRegistrationOptions

	/*WorkspaceSymbolProvider defined:
	 * The server provides workspace symbol support.
	 */
	WorkspaceSymbolProvider bool `json:"workspaceSymbolProvider,omitempty"` // boolean | WorkspaceSymbolOptions

	/*DocumentFormattingProvider defined:
	 * The server provides document formatting.
	 */
	DocumentFormattingProvider bool `json:"documentFormattingProvider,omitempty"` // boolean | DocumentFormattingOptions

	/*DocumentRangeFormattingProvider defined:
	 * The server provides document range formatting.
	 */
	DocumentRangeFormattingProvider bool `json:"documentRangeFormattingProvider,omitempty"` // boolean | DocumentRangeFormattingOptions

	/*DocumentOnTypeFormattingProvider defined:
	 * The server provides document formatting on typing.
	 */
	DocumentOnTypeFormattingProvider *DocumentOnTypeFormattingOptions `json:"documentOnTypeFormattingProvider,omitempty"`

	/*RenameProvider defined:
	 * The server provides rename support. RenameOptions may only be
	 * specified if the client states that it supports
	 * `prepareSupport` in its initial `initialize` request.
	 */
	RenameProvider interface{} `json:"renameProvider,omitempty"` // boolean | RenameOptions

	/*FoldingRangeProvider defined:
	 * The server provides folding provider support.
	 */
	FoldingRangeProvider bool `json:"foldingRangeProvider,omitempty"` // boolean | FoldingRangeOptions | FoldingRangeRegistrationOptions

	/*SelectionRangeProvider defined:
	 * The server provides selection range support.
	 */
	SelectionRangeProvider bool `json:"selectionRangeProvider,omitempty"` // boolean | SelectionRangeOptions | SelectionRangeRegistrationOptions

	/*ExecuteCommandProvider defined:
	 * The server provides execute command support.
	 */
	ExecuteCommandProvider *ExecuteCommandOptions `json:"executeCommandProvider,omitempty"`

	/*Experimental defined:
	 * Experimental server capabilities.
	 */
	Experimental interface{} `json:"experimental,omitempty"`

	/*Workspace defined:
	 * The workspace server capabilities
	 */
	Workspace *struct {

		// WorkspaceFolders is
		WorkspaceFolders *struct {

			/*Supported defined:
			 * The Server has support for workspace folders
			 */
			Supported bool `json:"supported,omitempty"`

			/*ChangeNotifications defined:
			 * Whether the server wants to receive workspace folder
			 * change notifications.
			 *
			 * If a strings is provided the string is treated as a ID
			 * under which the notification is registed on the client
			 * side. The ID can be used to unregister for these events
			 * using the `client/unregisterCapability` request.
			 */
			ChangeNotifications string `json:"changeNotifications,omitempty"` // string | boolean
		} `json:"workspaceFolders,omitempty"`
	} `json:"workspace,omitempty"`
}

/*InnerInitializeParams defined:
 * The initialize parameters
 */
type InnerInitializeParams struct {

	/*ProcessID defined:
	 * The process Id of the parent process that started
	 * the server.
	 */
	ProcessID float64 `json:"processId"`

	/*ClientInfo defined:
	 * Information about the client
	 *
	 * @since 3.15.0
	 */
	ClientInfo *struct {

		/*Name defined:
		 * The name of the client as defined by the client.
		 */
		Name string `json:"name"`

		/*Version defined:
		 * The client's version as defined by the client.
		 */
		Version string `json:"version,omitempty"`
	} `json:"clientInfo,omitempty"`

	/*RootPath defined:
	 * The rootPath of the workspace. Is null
	 * if no folder is open.
	 *
	 * @deprecated in favour of rootUri.
	 */
	RootPath string `json:"rootPath,omitempty"`

	/*RootURI defined:
	 * The rootUri of the workspace. Is null if no
	 * folder is open. If both `rootPath` and `rootUri` are set
	 * `rootUri` wins.
	 *
	 * @deprecated in favour of workspaceFolders.
	 */
	RootURI DocumentURI `json:"rootUri"`

	/*Capabilities defined:
	 * The capabilities provided by the client (editor or tool)
	 */
	Capabilities ClientCapabilities `json:"capabilities"`

	/*InitializationOptions defined:
	 * User provided initialization options.
	 */
	InitializationOptions interface{} `json:"initializationOptions,omitempty"`

	/*Trace defined:
	 * The initial trace setting. If omitted trace is disabled ('off').
	 */
	Trace string `json:"trace,omitempty"` // 'off' | 'messages' | 'verbose'
	WorkDoneProgressParams
}

// InitializeParams is
type InitializeParams struct {

	/*ProcessID defined:
	 * The process Id of the parent process that started
	 * the server.
	 */
	ProcessID float64 `json:"processId"`

	/*ClientInfo defined:
	 * Information about the client
	 *
	 * @since 3.15.0
	 */
	ClientInfo *struct {

		/*Name defined:
		 * The name of the client as defined by the client.
		 */
		Name string `json:"name"`

		/*Version defined:
		 * The client's version as defined by the client.
		 */
		Version string `json:"version,omitempty"`
	} `json:"clientInfo,omitempty"`

	/*RootPath defined:
	 * The rootPath of the workspace. Is null
	 * if no folder is open.
	 *
	 * @deprecated in favour of rootUri.
	 */
	RootPath string `json:"rootPath,omitempty"`

	/*RootURI defined:
	 * The rootUri of the workspace. Is null if no
	 * folder is open. If both `rootPath` and `rootUri` are set
	 * `rootUri` wins.
	 *
	 * @deprecated in favour of workspaceFolders.
	 */
	RootURI DocumentURI `json:"rootUri"`

	/*Capabilities defined:
	 * The capabilities provided by the client (editor or tool)
	 */
	Capabilities ClientCapabilities `json:"capabilities"`

	/*InitializationOptions defined:
	 * User provided initialization options.
	 */
	InitializationOptions interface{} `json:"initializationOptions,omitempty"`

	/*Trace defined:
	 * The initial trace setting. If omitted trace is disabled ('off').
	 */
	Trace string `json:"trace,omitempty"` // 'off' | 'messages' | 'verbose'

	/*WorkspaceFolders defined:
	 * The actual configured workspace folders.
	 */
	WorkspaceFolders []WorkspaceFolder `json:"workspaceFolders"`
}

/*InitializeResult defined:
 * The result returned from an initialize request.
 */
type InitializeResult struct {

	/*Capabilities defined:
	 * The capabilities the language server provides.
	 */
	Capabilities ServerCapabilities `json:"capabilities"`

	/*ServerInfo defined:
	 * Information about the server.
	 *
	 * @since 3.15.0
	 */
	ServerInfo *struct {

		/*Name defined:
		 * The name of the server as defined by the server.
		 */
		Name string `json:"name"`

		/*Version defined:
		 * The servers's version as defined by the server.
		 */
		Version string `json:"version,omitempty"`
	} `json:"serverInfo,omitempty"`

	/*Custom defined:
	 * Custom initialization results.
	 */
	Custom map[string]interface{} `json:"custom"` // [custom: string]: any;
}

// InitializedParams is
type InitializedParams struct {
}

// DidChangeConfigurationClientCapabilities is
type DidChangeConfigurationClientCapabilities struct {

	/*DynamicRegistration defined:
	 * Did change configuration notification supports dynamic registration.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
}

// DidChangeConfigurationRegistrationOptions is
type DidChangeConfigurationRegistrationOptions struct {

	// Section is
	Section string `json:"section,omitempty"` // string | string[]
}

/*DidChangeConfigurationParams defined:
 * The parameters of a change configuration notification.
 */
type DidChangeConfigurationParams struct {

	/*Settings defined:
	 * The actual changed settings
	 */
	Settings interface{} `json:"settings"`
}

/*ShowMessageParams defined:
 * The parameters of a notification message.
 */
type ShowMessageParams struct {

	/*Type defined:
	 * The message type. See {@link MessageType}
	 */
	Type MessageType `json:"type"`

	/*Message defined:
	 * The actual message
	 */
	Message string `json:"message"`
}

// MessageActionItem is
type MessageActionItem struct {

	/*Title defined:
	 * A short title like 'Retry', 'Open Log' etc.
	 */
	Title string `json:"title"`
}

// ShowMessageRequestParams is
type ShowMessageRequestParams struct {

	/*Type defined:
	 * The message type. See {@link MessageType}
	 */
	Type MessageType `json:"type"`

	/*Message defined:
	 * The actual message
	 */
	Message string `json:"message"`

	/*Actions defined:
	 * The message action items to present.
	 */
	Actions []MessageActionItem `json:"actions,omitempty"`
}

/*LogMessageParams defined:
 * The log message parameters.
 */
type LogMessageParams struct {

	/*Type defined:
	 * The message type. See {@link MessageType}
	 */
	Type MessageType `json:"type"`

	/*Message defined:
	 * The actual message
	 */
	Message string `json:"message"`
}

// TextDocumentSyncClientCapabilities is
type TextDocumentSyncClientCapabilities struct {

	/*DynamicRegistration defined:
	 * Whether text document synchronization supports dynamic registration.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`

	/*WillSave defined:
	 * The client supports sending will save notifications.
	 */
	WillSave bool `json:"willSave,omitempty"`

	/*WillSaveWaitUntil defined:
	 * The client supports sending a will save request and
	 * waits for a response providing text edits which will
	 * be applied to the document before it is saved.
	 */
	WillSaveWaitUntil bool `json:"willSaveWaitUntil,omitempty"`

	/*DidSave defined:
	 * The client supports did save notifications.
	 */
	DidSave bool `json:"didSave,omitempty"`
}

// TextDocumentSyncOptions is
type TextDocumentSyncOptions struct {

	/*OpenClose defined:
	 * Open and close notifications are sent to the server. If omitted open close notification should not
	 * be sent.
	 */
	OpenClose bool `json:"openClose,omitempty"`

	/*Change defined:
	 * Change notifications are sent to the server. See TextDocumentSyncKind.None, TextDocumentSyncKind.Full
	 * and TextDocumentSyncKind.Incremental. If omitted it defaults to TextDocumentSyncKind.None.
	 */
	Change TextDocumentSyncKind `json:"change,omitempty"`

	/*WillSave defined:
	 * If present will save notifications are sent to the server. If omitted the notification should not be
	 * sent.
	 */
	WillSave bool `json:"willSave,omitempty"`

	/*WillSaveWaitUntil defined:
	 * If present will save wait until requests are sent to the server. If omitted the request should not be
	 * sent.
	 */
	WillSaveWaitUntil bool `json:"willSaveWaitUntil,omitempty"`

	/*Save defined:
	 * If present save notifications are sent to the server. If omitted the notification should not be
	 * sent.
	 */
	Save *SaveOptions `json:"save,omitempty"`
}

/*DidOpenTextDocumentParams defined:
 * The parameters send in a open text document notification
 */
type DidOpenTextDocumentParams struct {

	/*TextDocument defined:
	 * The document that was opened.
	 */
	TextDocument TextDocumentItem `json:"textDocument"`
}

/*DidChangeTextDocumentParams defined:
 * The change text document notification's parameters.
 */
type DidChangeTextDocumentParams struct {

	/*TextDocument defined:
	 * The document that did change. The version number points
	 * to the version after all provided content changes have
	 * been applied.
	 */
	TextDocument VersionedTextDocumentIdentifier `json:"textDocument"`

	/*ContentChanges defined:
	 * The actual content changes. The content changes describe single state changes
	 * to the document. So if there are two content changes c1 and c2 for a document
	 * in state S then c1 move the document to S' and c2 to S''.
	 */
	ContentChanges []TextDocumentContentChangeEvent `json:"contentChanges"`
}

/*TextDocumentChangeRegistrationOptions defined:
 * Describe options to be used when registered for text document change events.
 */
type TextDocumentChangeRegistrationOptions struct {

	/*SyncKind defined:
	 * How documents are synced to the server.
	 */
	SyncKind TextDocumentSyncKind `json:"syncKind"`
	TextDocumentRegistrationOptions
}

/*DidCloseTextDocumentParams defined:
 * The parameters send in a close text document notification
 */
type DidCloseTextDocumentParams struct {

	/*TextDocument defined:
	 * The document that was closed.
	 */
	TextDocument TextDocumentIdentifier `json:"textDocument"`
}

/*DidSaveTextDocumentParams defined:
 * The parameters send in a save text document notification
 */
type DidSaveTextDocumentParams struct {

	/*TextDocument defined:
	 * The document that was closed.
	 */
	TextDocument VersionedTextDocumentIdentifier `json:"textDocument"`

	/*Text defined:
	 * Optional the content when saved. Depends on the includeText value
	 * when the save notification was requested.
	 */
	Text string `json:"text,omitempty"`
}

/*TextDocumentSaveRegistrationOptions defined:
 * Save registration options.
 */
type TextDocumentSaveRegistrationOptions struct {
	TextDocumentRegistrationOptions
	SaveOptions
}

/*WillSaveTextDocumentParams defined:
 * The parameters send in a will save text document notification.
 */
type WillSaveTextDocumentParams struct {

	/*TextDocument defined:
	 * The document that will be saved.
	 */
	TextDocument TextDocumentIdentifier `json:"textDocument"`

	/*Reason defined:
	 * The 'TextDocumentSaveReason'.
	 */
	Reason TextDocumentSaveReason `json:"reason"`
}

// DidChangeWatchedFilesClientCapabilities is
type DidChangeWatchedFilesClientCapabilities struct {

	/*DynamicRegistration defined:
	 * Did change watched files notification supports dynamic registration. Please note
	 * that the current protocol doesn't support static configuration for file changes
	 * from the server side.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
}

/*DidChangeWatchedFilesParams defined:
 * The watched files change notification's parameters.
 */
type DidChangeWatchedFilesParams struct {

	/*Changes defined:
	 * The actual file events.
	 */
	Changes []FileEvent `json:"changes"`
}

/*FileEvent defined:
 * An event describing a file change.
 */
type FileEvent struct {

	/*URI defined:
	 * The file's uri.
	 */
	URI DocumentURI `json:"uri"`

	/*Type defined:
	 * The change type.
	 */
	Type FileChangeType `json:"type"`
}

/*DidChangeWatchedFilesRegistrationOptions defined:
 * Describe options to be used when registered for text document change events.
 */
type DidChangeWatchedFilesRegistrationOptions struct {

	/*Watchers defined:
	 * The watchers to register.
	 */
	Watchers []FileSystemWatcher `json:"watchers"`
}

// FileSystemWatcher is
type FileSystemWatcher struct {

	/*GlobPattern defined:
	 * The  glob pattern to watch. Glob patterns can have the following syntax:
	 * - `*` to match one or more characters in a path segment
	 * - `?` to match on one character in a path segment
	 * - `**` to match any number of path segments, including none
	 * - `{}` to group conditions (e.g. `**​/*.{ts,js}` matches all TypeScript and JavaScript files)
	 * - `[]` to declare a range of characters to match in a path segment (e.g., `example.[0-9]` to match on `example.0`, `example.1`, …)
	 * - `[!...]` to negate a range of characters to match in a path segment (e.g., `example.[!0-9]` to match on `example.a`, `example.b`, but not `example.0`)
	 */
	GlobPattern string `json:"globPattern"`

	/*Kind defined:
	 * The kind of events of interest. If omitted it defaults
	 * to WatchKind.Create | WatchKind.Change | WatchKind.Delete
	 * which is 7.
	 */
	Kind float64 `json:"kind,omitempty"`
}

/*PublishDiagnosticsClientCapabilities defined:
 * The publish diagnostic client capabilities.
 */
type PublishDiagnosticsClientCapabilities struct {

	/*RelatedInformation defined:
	 * Whether the clients accepts diagnostics with related information.
	 */
	RelatedInformation bool `json:"relatedInformation,omitempty"`

	/*TagSupport defined:
	 * Client supports the tag property to provide meta data about a diagnostic.
	 * Clients supporting tags have to handle unknown tags gracefully.
	 *
	 * @since 3.15.0
	 */
	TagSupport *struct {

		/*ValueSet defined:
		 * The tags supported by the client.
		 */
		ValueSet []DiagnosticTag `json:"valueSet"`
	} `json:"tagSupport,omitempty"`
}

/*PublishDiagnosticsParams defined:
 * The publish diagnostic notification's parameters.
 */
type PublishDiagnosticsParams struct {

	/*URI defined:
	 * The URI for which diagnostic information is reported.
	 */
	URI DocumentURI `json:"uri"`

	/*Version defined:
	 * Optional the version number of the document the diagnostics are published for.
	 *
	 * @since 3.15.0
	 */
	Version float64 `json:"version,omitempty"`

	/*Diagnostics defined:
	 * An array of diagnostic information items.
	 */
	Diagnostics []Diagnostic `json:"diagnostics"`
}

/*CompletionClientCapabilities defined:
 * Completion client capabilities
 */
type CompletionClientCapabilities struct {

	/*DynamicRegistration defined:
	 * Whether completion supports dynamic registration.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`

	/*CompletionItem defined:
	 * The client supports the following `CompletionItem` specific
	 * capabilities.
	 */
	CompletionItem *struct {

		/*SnippetSupport defined:
		 * Client supports snippets as insert text.
		 *
		 * A snippet can define tab stops and placeholders with `$1`, `$2`
		 * and `${3:foo}`. `$0` defines the final tab stop, it defaults to
		 * the end of the snippet. Placeholders with equal identifiers are linked,
		 * that is typing in one will update others too.
		 */
		SnippetSupport bool `json:"snippetSupport,omitempty"`

		/*CommitCharactersSupport defined:
		 * Client supports commit characters on a completion item.
		 */
		CommitCharactersSupport bool `json:"commitCharactersSupport,omitempty"`

		/*DocumentationFormat defined:
		 * Client supports the follow content formats for the documentation
		 * property. The order describes the preferred format of the client.
		 */
		DocumentationFormat []MarkupKind `json:"documentationFormat,omitempty"`

		/*DeprecatedSupport defined:
		 * Client supports the deprecated property on a completion item.
		 */
		DeprecatedSupport bool `json:"deprecatedSupport,omitempty"`

		/*PreselectSupport defined:
		 * Client supports the preselect property on a completion item.
		 */
		PreselectSupport bool `json:"preselectSupport,omitempty"`

		/*TagSupport defined:
		 * Client supports the tag property on a completion item. Clients supporting
		 * tags have to handle unknown tags gracefully. Clients especially need to
		 * preserve unknown tags when sending a completion item back to the server in
		 * a resolve call.
		 *
		 * @since 3.15.0
		 */
		TagSupport *struct {

			/*ValueSet defined:
			 * The tags supported by the client.
			 */
			ValueSet []CompletionItemTag `json:"valueSet"`
		} `json:"tagSupport,omitempty"`
	} `json:"completionItem,omitempty"`

	// CompletionItemKind is
	CompletionItemKind *struct {

		/*ValueSet defined:
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

	/*ContextSupport defined:
	 * The client supports to send additional context information for a
	 * `textDocument/completion` requestion.
	 */
	ContextSupport bool `json:"contextSupport,omitempty"`
}

/*CompletionContext defined:
 * Contains additional information about the context in which a completion request is triggered.
 */
type CompletionContext struct {

	/*TriggerKind defined:
	 * How the completion was triggered.
	 */
	TriggerKind CompletionTriggerKind `json:"triggerKind"`

	/*TriggerCharacter defined:
	 * The trigger character (a single character) that has trigger code complete.
	 * Is undefined if `triggerKind !== CompletionTriggerKind.TriggerCharacter`
	 */
	TriggerCharacter string `json:"triggerCharacter,omitempty"`
}

/*CompletionParams defined:
 * Completion parameters
 */
type CompletionParams struct {

	/*Context defined:
	 * The completion context. This is only available it the client specifies
	 * to send this using the client capability `textDocument.completion.contextSupport === true`
	 */
	Context *CompletionContext `json:"context,omitempty"`
	TextDocumentPositionParams
	WorkDoneProgressParams
	PartialResultParams
}

/*CompletionOptions defined:
 * Completion options.
 */
type CompletionOptions struct {

	/*TriggerCharacters defined:
	 * Most tools trigger completion request automatically without explicitly requesting
	 * it using a keyboard shortcut (e.g. Ctrl+Space). Typically they do so when the user
	 * starts to type an identifier. For example if the user types `c` in a JavaScript file
	 * code complete will automatically pop up present `console` besides others as a
	 * completion item. Characters that make up identifiers don't need to be listed here.
	 *
	 * If code complete should automatically be trigger on characters not being valid inside
	 * an identifier (for example `.` in JavaScript) list them in `triggerCharacters`.
	 */
	TriggerCharacters []string `json:"triggerCharacters,omitempty"`

	/*AllCommitCharacters defined:
	 * The list of all possible characters that commit a completion. This field can be used
	 * if clients don't support individual commmit characters per completion item. See
	 * `ClientCapabilities.textDocument.completion.completionItem.commitCharactersSupport`
	 *
	 * @since 3.2.0
	 */
	AllCommitCharacters []string `json:"allCommitCharacters,omitempty"`

	/*ResolveProvider defined:
	 * The server provides support to resolve additional
	 * information for a completion item.
	 */
	ResolveProvider bool `json:"resolveProvider,omitempty"`
	WorkDoneProgressOptions
}

/*CompletionRegistrationOptions defined:
 * Registration options for a [CompletionRequest](#CompletionRequest).
 */
type CompletionRegistrationOptions struct {
	TextDocumentRegistrationOptions
	CompletionOptions
}

// HoverClientCapabilities is
type HoverClientCapabilities struct {

	/*DynamicRegistration defined:
	 * Whether hover supports dynamic registration.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`

	/*ContentFormat defined:
	 * Client supports the follow content formats for the content
	 * property. The order describes the preferred format of the client.
	 */
	ContentFormat []MarkupKind `json:"contentFormat,omitempty"`
}

/*HoverOptions defined:
 * Hover options.
 */
type HoverOptions struct {
	WorkDoneProgressOptions
}

/*HoverParams defined:
 * Parameters for a [HoverRequest](#HoverRequest).
 */
type HoverParams struct {
	TextDocumentPositionParams
	WorkDoneProgressParams
}

/*HoverRegistrationOptions defined:
 * Registration options for a [HoverRequest](#HoverRequest).
 */
type HoverRegistrationOptions struct {
	TextDocumentRegistrationOptions
	HoverOptions
}

/*SignatureHelpClientCapabilities defined:
 * Client Capabilities for a [SignatureHelpRequest](#SignatureHelpRequest).
 */
type SignatureHelpClientCapabilities struct {

	/*DynamicRegistration defined:
	 * Whether signature help supports dynamic registration.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`

	/*SignatureInformation defined:
	 * The client supports the following `SignatureInformation`
	 * specific properties.
	 */
	SignatureInformation *struct {

		/*DocumentationFormat defined:
		 * Client supports the follow content formats for the documentation
		 * property. The order describes the preferred format of the client.
		 */
		DocumentationFormat []MarkupKind `json:"documentationFormat,omitempty"`

		/*ParameterInformation defined:
		 * Client capabilities specific to parameter information.
		 */
		ParameterInformation *struct {

			/*LabelOffsetSupport defined:
			 * The client supports processing label offsets instead of a
			 * simple label string.
			 *
			 * @since 3.14.0
			 */
			LabelOffsetSupport bool `json:"labelOffsetSupport,omitempty"`
		} `json:"parameterInformation,omitempty"`
	} `json:"signatureInformation,omitempty"`

	/*ContextSupport defined:
	 * The client supports to send additional context information for a
	 * `textDocument/signatureHelp` request. A client that opts into
	 * contextSupport will also support the `retriggerCharacters` on
	 * `SignatureHelpOptions`.
	 *
	 * @since 3.15.0
	 */
	ContextSupport bool `json:"contextSupport,omitempty"`
}

/*SignatureHelpOptions defined:
 * Server Capabilities for a [SignatureHelpRequest](#SignatureHelpRequest).
 */
type SignatureHelpOptions struct {

	/*TriggerCharacters defined:
	 * List of characters that trigger signature help.
	 */
	TriggerCharacters []string `json:"triggerCharacters,omitempty"`

	/*RetriggerCharacters defined:
	 * List of characters that re-trigger signature help.
	 *
	 * These trigger characters are only active when signature help is already showing. All trigger characters
	 * are also counted as re-trigger characters.
	 *
	 * @since 3.15.0
	 */
	RetriggerCharacters []string `json:"retriggerCharacters,omitempty"`
	WorkDoneProgressOptions
}

/*SignatureHelpContext defined:
 * Additional information about the context in which a signature help request was triggered.
 *
 * @since 3.15.0
 */
type SignatureHelpContext struct {

	/*TriggerKind defined:
	 * Action that caused signature help to be triggered.
	 */
	TriggerKind SignatureHelpTriggerKind `json:"triggerKind"`

	/*TriggerCharacter defined:
	 * Character that caused signature help to be triggered.
	 *
	 * This is undefined when `triggerKind !== SignatureHelpTriggerKind.TriggerCharacter`
	 */
	TriggerCharacter string `json:"triggerCharacter,omitempty"`

	/*IsRetrigger defined:
	 * `true` if signature help was already showing when it was triggered.
	 *
	 * Retriggers occur when the signature help is already active and can be caused by actions such as
	 * typing a trigger character, a cursor move, or document content changes.
	 */
	IsRetrigger bool `json:"isRetrigger"`

	/*ActiveSignatureHelp defined:
	 * The currently active `SignatureHelp`.
	 *
	 * The `activeSignatureHelp` has its `SignatureHelp.activeSignature` field updated based on
	 * the user navigating through available signatures.
	 */
	ActiveSignatureHelp *SignatureHelp `json:"activeSignatureHelp,omitempty"`
}

/*SignatureHelpParams defined:
 * Parameters for a [SignatureHelpRequest](#SignatureHelpRequest).
 */
type SignatureHelpParams struct {

	/*Context defined:
	 * The signature help context. This is only available if the client specifies
	 * to send this using the client capability `textDocument.signatureHelp.contextSupport === true`
	 *
	 * @since 3.15.0
	 */
	Context *SignatureHelpContext `json:"context,omitempty"`
	TextDocumentPositionParams
	WorkDoneProgressParams
}

/*SignatureHelpRegistrationOptions defined:
 * Registration options for a [SignatureHelpRequest](#SignatureHelpRequest).
 */
type SignatureHelpRegistrationOptions struct {
	TextDocumentRegistrationOptions
	SignatureHelpOptions
}

/*DefinitionClientCapabilities defined:
 * Client Capabilities for a [DefinitionRequest](#DefinitionRequest).
 */
type DefinitionClientCapabilities struct {

	/*DynamicRegistration defined:
	 * Whether definition supports dynamic registration.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`

	/*LinkSupport defined:
	 * The client supports additional metadata in the form of definition links.
	 *
	 * @since 3.14.0
	 */
	LinkSupport bool `json:"linkSupport,omitempty"`
}

/*DefinitionOptions defined:
 * Server Capabilities for a [DefinitionRequest](#DefinitionRequest).
 */
type DefinitionOptions struct {
	WorkDoneProgressOptions
}

/*DefinitionParams defined:
 * Parameters for a [DefinitionRequest](#DefinitionRequest).
 */
type DefinitionParams struct {
	TextDocumentPositionParams
	WorkDoneProgressParams
	PartialResultParams
}

/*DefinitionRegistrationOptions defined:
 * Registration options for a [DefinitionRequest](#DefinitionRequest).
 */
type DefinitionRegistrationOptions struct {
	TextDocumentRegistrationOptions
	DefinitionOptions
}

/*ReferenceClientCapabilities defined:
 * Client Capabilities for a [ReferencesRequest](#ReferencesRequest).
 */
type ReferenceClientCapabilities struct {

	/*DynamicRegistration defined:
	 * Whether references supports dynamic registration.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
}

/*ReferenceParams defined:
 * Parameters for a [ReferencesRequest](#ReferencesRequest).
 */
type ReferenceParams struct {

	// Context is
	Context ReferenceContext `json:"context"`
	TextDocumentPositionParams
	WorkDoneProgressParams
	PartialResultParams
}

/*ReferenceOptions defined:
 * Reference options.
 */
type ReferenceOptions struct {
	WorkDoneProgressOptions
}

/*ReferenceRegistrationOptions defined:
 * Registration options for a [ReferencesRequest](#ReferencesRequest).
 */
type ReferenceRegistrationOptions struct {
	TextDocumentRegistrationOptions
	ReferenceOptions
}

/*DocumentHighlightClientCapabilities defined:
 * Client Capabilities for a [DocumentHighlightRequest](#DocumentHighlightRequest).
 */
type DocumentHighlightClientCapabilities struct {

	/*DynamicRegistration defined:
	 * Whether document highlight supports dynamic registration.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
}

/*DocumentHighlightParams defined:
 * Parameters for a [DocumentHighlightRequest](#DocumentHighlightRequest).
 */
type DocumentHighlightParams struct {
	TextDocumentPositionParams
	WorkDoneProgressParams
	PartialResultParams
}

/*DocumentHighlightOptions defined:
 * Provider options for a [DocumentHighlightRequest](#DocumentHighlightRequest).
 */
type DocumentHighlightOptions struct {
	WorkDoneProgressOptions
}

/*DocumentHighlightRegistrationOptions defined:
 * Registration options for a [DocumentHighlightRequest](#DocumentHighlightRequest).
 */
type DocumentHighlightRegistrationOptions struct {
	TextDocumentRegistrationOptions
	DocumentHighlightOptions
}

/*DocumentSymbolClientCapabilities defined:
 * Client Capabilities for a [DocumentSymbolRequest](#DocumentSymbolRequest).
 */
type DocumentSymbolClientCapabilities struct {

	/*DynamicRegistration defined:
	 * Whether document symbol supports dynamic registration.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`

	/*SymbolKind defined:
	 * Specific capabilities for the `SymbolKind`.
	 */
	SymbolKind *struct {

		/*ValueSet defined:
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

	/*HierarchicalDocumentSymbolSupport defined:
	 * The client support hierarchical document symbols.
	 */
	HierarchicalDocumentSymbolSupport bool `json:"hierarchicalDocumentSymbolSupport,omitempty"`
}

/*DocumentSymbolParams defined:
 * Parameters for a [DocumentSymbolRequest](#DocumentSymbolRequest).
 */
type DocumentSymbolParams struct {

	/*TextDocument defined:
	 * The text document.
	 */
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	WorkDoneProgressParams
	PartialResultParams
}

/*DocumentSymbolOptions defined:
 * Provider options for a [DocumentSymbolRequest](#DocumentSymbolRequest).
 */
type DocumentSymbolOptions struct {
	WorkDoneProgressOptions
}

/*DocumentSymbolRegistrationOptions defined:
 * Registration options for a [DocumentSymbolRequest](#DocumentSymbolRequest).
 */
type DocumentSymbolRegistrationOptions struct {
	TextDocumentRegistrationOptions
	DocumentSymbolOptions
}

/*CodeActionClientCapabilities defined:
 * The Client Capabilities of a [CodeActionRequest](#CodeActionRequest).
 */
type CodeActionClientCapabilities struct {

	/*DynamicRegistration defined:
	 * Whether code action supports dynamic registration.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`

	/*CodeActionLiteralSupport defined:
	 * The client support code action literals as a valid
	 * response of the `textDocument/codeAction` request.
	 *
	 * @since 3.8.0
	 */
	CodeActionLiteralSupport *struct {

		/*CodeActionKind defined:
		 * The code action kind is support with the following value
		 * set.
		 */
		CodeActionKind struct {

			/*ValueSet defined:
			 * The code action kind values the client supports. When this
			 * property exists the client also guarantees that it will
			 * handle values outside its set gracefully and falls back
			 * to a default value when unknown.
			 */
			ValueSet []CodeActionKind `json:"valueSet"`
		} `json:"codeActionKind"`
	} `json:"codeActionLiteralSupport,omitempty"`

	/*IsPreferredSupport defined:
	 * Whether code action supports the `isPreferred` property.
	 * @since 3.15.0
	 */
	IsPreferredSupport bool `json:"isPreferredSupport,omitempty"`
}

/*CodeActionParams defined:
 * The parameters of a [CodeActionRequest](#CodeActionRequest).
 */
type CodeActionParams struct {

	/*TextDocument defined:
	 * The document in which the command was invoked.
	 */
	TextDocument TextDocumentIdentifier `json:"textDocument"`

	/*Range defined:
	 * The range for which the command was invoked.
	 */
	Range Range `json:"range"`

	/*Context defined:
	 * Context carrying additional information.
	 */
	Context CodeActionContext `json:"context"`
	WorkDoneProgressParams
	PartialResultParams
}

/*CodeActionOptions defined:
 * Provider options for a [CodeActionRequest](#CodeActionRequest).
 */
type CodeActionOptions struct {

	/*CodeActionKinds defined:
	 * CodeActionKinds that this server may return.
	 *
	 * The list of kinds may be generic, such as `CodeActionKind.Refactor`, or the server
	 * may list out every specific kind they provide.
	 */
	CodeActionKinds []CodeActionKind `json:"codeActionKinds,omitempty"`
	WorkDoneProgressOptions
}

/*CodeActionRegistrationOptions defined:
 * Registration options for a [CodeActionRequest](#CodeActionRequest).
 */
type CodeActionRegistrationOptions struct {
	TextDocumentRegistrationOptions
	CodeActionOptions
}

/*WorkspaceSymbolClientCapabilities defined:
 * Client capabilities for a [WorkspaceSymbolRequest](#WorkspaceSymbolRequest).
 */
type WorkspaceSymbolClientCapabilities struct {

	/*DynamicRegistration defined:
	 * Symbol request supports dynamic registration.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`

	/*SymbolKind defined:
	 * Specific capabilities for the `SymbolKind` in the `workspace/symbol` request.
	 */
	SymbolKind *struct {

		/*ValueSet defined:
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
}

/*WorkspaceSymbolParams defined:
 * The parameters of a [WorkspaceSymbolRequest](#WorkspaceSymbolRequest).
 */
type WorkspaceSymbolParams struct {

	/*Query defined:
	 * A query string to filter symbols by. Clients may send an empty
	 * string here to request all symbols.
	 */
	Query string `json:"query"`
	WorkDoneProgressParams
	PartialResultParams
}

/*WorkspaceSymbolOptions defined:
 * Server capabilities for a [WorkspaceSymbolRequest](#WorkspaceSymbolRequest).
 */
type WorkspaceSymbolOptions struct {
	WorkDoneProgressOptions
}

/*WorkspaceSymbolRegistrationOptions defined:
 * Registration options for a [WorkspaceSymbolRequest](#WorkspaceSymbolRequest).
 */
type WorkspaceSymbolRegistrationOptions struct {
	WorkspaceSymbolOptions
}

/*CodeLensClientCapabilities defined:
 * The client capabilities  of a [CodeLensRequest](#CodeLensRequest).
 */
type CodeLensClientCapabilities struct {

	/*DynamicRegistration defined:
	 * Whether code lens supports dynamic registration.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
}

/*CodeLensParams defined:
 * The parameters of a [CodeLensRequest](#CodeLensRequest).
 */
type CodeLensParams struct {

	/*TextDocument defined:
	 * The document to request code lens for.
	 */
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	WorkDoneProgressParams
	PartialResultParams
}

/*CodeLensOptions defined:
 * Code Lens provider options of a [CodeLensRequest](#CodeLensRequest).
 */
type CodeLensOptions struct {

	/*ResolveProvider defined:
	 * Code lens has a resolve provider as well.
	 */
	ResolveProvider bool `json:"resolveProvider,omitempty"`
	WorkDoneProgressOptions
}

/*CodeLensRegistrationOptions defined:
 * Registration options for a [CodeLensRequest](#CodeLensRequest).
 */
type CodeLensRegistrationOptions struct {
	TextDocumentRegistrationOptions
	CodeLensOptions
}

/*DocumentLinkClientCapabilities defined:
 * The client capabilities of a [DocumentLinkRequest](#DocumentLinkRequest).
 */
type DocumentLinkClientCapabilities struct {

	/*DynamicRegistration defined:
	 * Whether document link supports dynamic registration.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`

	/*TooltipSupport defined:
	 * Whether the client support the `tooltip` property on `DocumentLink`.
	 *
	 * @since 3.15.0
	 */
	TooltipSupport bool `json:"tooltipSupport,omitempty"`
}

/*DocumentLinkParams defined:
 * The parameters of a [DocumentLinkRequest](#DocumentLinkRequest).
 */
type DocumentLinkParams struct {

	/*TextDocument defined:
	 * The document to provide document links for.
	 */
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	WorkDoneProgressParams
	PartialResultParams
}

/*DocumentLinkOptions defined:
 * Provider options for a [DocumentLinkRequest](#DocumentLinkRequest).
 */
type DocumentLinkOptions struct {

	/*ResolveProvider defined:
	 * Document links have a resolve provider as well.
	 */
	ResolveProvider bool `json:"resolveProvider,omitempty"`
	WorkDoneProgressOptions
}

/*DocumentLinkRegistrationOptions defined:
 * Registration options for a [DocumentLinkRequest](#DocumentLinkRequest).
 */
type DocumentLinkRegistrationOptions struct {
	TextDocumentRegistrationOptions
	DocumentLinkOptions
}

/*DocumentFormattingClientCapabilities defined:
 * Client capabilities of a [DocumentFormattingRequest](#DocumentFormattingRequest).
 */
type DocumentFormattingClientCapabilities struct {

	/*DynamicRegistration defined:
	 * Whether formatting supports dynamic registration.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
}

/*DocumentFormattingParams defined:
 * The parameters of a [DocumentFormattingRequest](#DocumentFormattingRequest).
 */
type DocumentFormattingParams struct {

	/*TextDocument defined:
	 * The document to format.
	 */
	TextDocument TextDocumentIdentifier `json:"textDocument"`

	/*Options defined:
	 * The format options
	 */
	Options FormattingOptions `json:"options"`
	WorkDoneProgressParams
}

/*DocumentFormattingOptions defined:
 * Provider options for a [DocumentFormattingRequest](#DocumentFormattingRequest).
 */
type DocumentFormattingOptions struct {
	WorkDoneProgressOptions
}

/*DocumentFormattingRegistrationOptions defined:
 * Registration options for a [DocumentFormattingRequest](#DocumentFormattingRequest).
 */
type DocumentFormattingRegistrationOptions struct {
	TextDocumentRegistrationOptions
	DocumentFormattingOptions
}

/*DocumentRangeFormattingClientCapabilities defined:
 * Client capabilities of a [DocumentRangeFormattingRequest](#DocumentRangeFormattingRequest).
 */
type DocumentRangeFormattingClientCapabilities struct {

	/*DynamicRegistration defined:
	 * Whether range formatting supports dynamic registration.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
}

/*DocumentRangeFormattingParams defined:
 * The parameters of a [DocumentRangeFormattingRequest](#DocumentRangeFormattingRequest).
 */
type DocumentRangeFormattingParams struct {

	/*TextDocument defined:
	 * The document to format.
	 */
	TextDocument TextDocumentIdentifier `json:"textDocument"`

	/*Range defined:
	 * The range to format
	 */
	Range Range `json:"range"`

	/*Options defined:
	 * The format options
	 */
	Options FormattingOptions `json:"options"`
	WorkDoneProgressParams
}

/*DocumentRangeFormattingOptions defined:
 * Provider options for a [DocumentRangeFormattingRequest](#DocumentRangeFormattingRequest).
 */
type DocumentRangeFormattingOptions struct {
	WorkDoneProgressOptions
}

/*DocumentRangeFormattingRegistrationOptions defined:
 * Registration options for a [DocumentRangeFormattingRequest](#DocumentRangeFormattingRequest).
 */
type DocumentRangeFormattingRegistrationOptions struct {
	TextDocumentRegistrationOptions
	DocumentRangeFormattingOptions
}

/*DocumentOnTypeFormattingClientCapabilities defined:
 * Client capabilities of a [DocumentOnTypeFormattingRequest](#DocumentOnTypeFormattingRequest).
 */
type DocumentOnTypeFormattingClientCapabilities struct {

	/*DynamicRegistration defined:
	 * Whether on type formatting supports dynamic registration.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
}

/*DocumentOnTypeFormattingParams defined:
 * The parameters of a [DocumentOnTypeFormattingRequest](#DocumentOnTypeFormattingRequest).
 */
type DocumentOnTypeFormattingParams struct {

	/*TextDocument defined:
	 * The document to format.
	 */
	TextDocument TextDocumentIdentifier `json:"textDocument"`

	/*Position defined:
	 * The position at which this request was send.
	 */
	Position Position `json:"position"`

	/*Ch defined:
	 * The character that has been typed.
	 */
	Ch string `json:"ch"`

	/*Options defined:
	 * The format options.
	 */
	Options FormattingOptions `json:"options"`
}

/*DocumentOnTypeFormattingOptions defined:
 * Provider options for a [DocumentOnTypeFormattingRequest](#DocumentOnTypeFormattingRequest).
 */
type DocumentOnTypeFormattingOptions struct {

	/*FirstTriggerCharacter defined:
	 * A character on which formatting should be triggered, like `}`.
	 */
	FirstTriggerCharacter string `json:"firstTriggerCharacter"`

	/*MoreTriggerCharacter defined:
	 * More trigger characters.
	 */
	MoreTriggerCharacter []string `json:"moreTriggerCharacter,omitempty"`
}

/*DocumentOnTypeFormattingRegistrationOptions defined:
 * Registration options for a [DocumentOnTypeFormattingRequest](#DocumentOnTypeFormattingRequest).
 */
type DocumentOnTypeFormattingRegistrationOptions struct {
	TextDocumentRegistrationOptions
	DocumentOnTypeFormattingOptions
}

// RenameClientCapabilities is
type RenameClientCapabilities struct {

	/*DynamicRegistration defined:
	 * Whether rename supports dynamic registration.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`

	/*PrepareSupport defined:
	 * Client supports testing for validity of rename operations
	 * before execution.
	 *
	 * @since version 3.12.0
	 */
	PrepareSupport bool `json:"prepareSupport,omitempty"`
}

/*RenameParams defined:
 * The parameters of a [RenameRequest](#RenameRequest).
 */
type RenameParams struct {

	/*TextDocument defined:
	 * The document to rename.
	 */
	TextDocument TextDocumentIdentifier `json:"textDocument"`

	/*Position defined:
	 * The position at which this request was sent.
	 */
	Position Position `json:"position"`

	/*NewName defined:
	 * The new name of the symbol. If the given name is not valid the
	 * request must return a [ResponseError](#ResponseError) with an
	 * appropriate message set.
	 */
	NewName string `json:"newName"`
	WorkDoneProgressParams
}

/*RenameOptions defined:
 * Provider options for a [RenameRequest](#RenameRequest).
 */
type RenameOptions struct {

	/*PrepareProvider defined:
	 * Renames should be checked and tested before being executed.
	 *
	 * @since version 3.12.0
	 */
	PrepareProvider bool `json:"prepareProvider,omitempty"`
	WorkDoneProgressOptions
}

/*RenameRegistrationOptions defined:
 * Registration options for a [RenameRequest](#RenameRequest).
 */
type RenameRegistrationOptions struct {
	TextDocumentRegistrationOptions
	RenameOptions
}

// PrepareRenameParams is
type PrepareRenameParams struct {
	TextDocumentPositionParams
	WorkDoneProgressParams
}

/*ExecuteCommandClientCapabilities defined:
 * The client capabilities of a [ExecuteCommandRequest](#ExecuteCommandRequest).
 */
type ExecuteCommandClientCapabilities struct {

	/*DynamicRegistration defined:
	 * Execute command supports dynamic registration.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
}

/*ExecuteCommandParams defined:
 * The parameters of a [ExecuteCommandRequest](#ExecuteCommandRequest).
 */
type ExecuteCommandParams struct {

	/*Command defined:
	 * The identifier of the actual command handler.
	 */
	Command string `json:"command"`

	/*Arguments defined:
	 * Arguments that the command should be invoked with.
	 */
	Arguments []interface{} `json:"arguments,omitempty"`
	WorkDoneProgressParams
}

/*ExecuteCommandOptions defined:
 * The server capabilities of a [ExecuteCommandRequest](#ExecuteCommandRequest).
 */
type ExecuteCommandOptions struct {

	/*Commands defined:
	 * The commands to be executed on the server
	 */
	Commands []string `json:"commands"`
	WorkDoneProgressOptions
}

/*ExecuteCommandRegistrationOptions defined:
 * Registration options for a [ExecuteCommandRequest](#ExecuteCommandRequest).
 */
type ExecuteCommandRegistrationOptions struct {
	ExecuteCommandOptions
}

// WorkspaceEditClientCapabilities is
type WorkspaceEditClientCapabilities struct {

	/*DocumentChanges defined:
	 * The client supports versioned document changes in `WorkspaceEdit`s
	 */
	DocumentChanges bool `json:"documentChanges,omitempty"`

	/*ResourceOperations defined:
	 * The resource operations the client supports. Clients should at least
	 * support 'create', 'rename' and 'delete' files and folders.
	 *
	 * @since 3.13.0
	 */
	ResourceOperations []ResourceOperationKind `json:"resourceOperations,omitempty"`

	/*FailureHandling defined:
	 * The failure handling strategy of a client if applying the workspace edit
	 * fails.
	 *
	 * @since 3.13.0
	 */
	FailureHandling FailureHandlingKind `json:"failureHandling,omitempty"`
}

/*ApplyWorkspaceEditParams defined:
 * The parameters passed via a apply workspace edit request.
 */
type ApplyWorkspaceEditParams struct {

	/*Label defined:
	 * An optional label of the workspace edit. This label is
	 * presented in the user interface for example on an undo
	 * stack to undo the workspace edit.
	 */
	Label string `json:"label,omitempty"`

	/*Edit defined:
	 * The edits to apply.
	 */
	Edit WorkspaceEdit `json:"edit"`
}

/*ApplyWorkspaceEditResponse defined:
 * A response returned from the apply workspace edit request.
 */
type ApplyWorkspaceEditResponse struct {

	/*Applied defined:
	 * Indicates whether the edit was applied or not.
	 */
	Applied bool `json:"applied"`

	/*FailureReason defined:
	 * An optional textual description for why the edit was not applied.
	 * This may be used by the server for diagnostic logging or to provide
	 * a suitable error for a request that triggered the edit.
	 */
	FailureReason string `json:"failureReason,omitempty"`

	/*FailedChange defined:
	 * Depending on the client's failure handling strategy `failedChange` might
	 * contain the index of the change that failed. This property is only available
	 * if the client signals a `failureHandlingStrategy` in its client capabilities.
	 */
	FailedChange float64 `json:"failedChange,omitempty"`
}

/*Position defined:
 * Position in a text document expressed as zero-based line and character offset.
 * The offsets are based on a UTF-16 string representation. So a string of the form
 * `a𐐀b` the character offset of the character `a` is 0, the character offset of `𐐀`
 * is 1 and the character offset of b is 3 since `𐐀` is represented using two code
 * units in UTF-16.
 *
 * Positions are line end character agnostic. So you can not specify a position that
 * denotes `\r|\n` or `\n|` where `|` represents the character offset.
 */
type Position struct {

	/*Line defined:
	 * Line position in a document (zero-based).
	 * If a line number is greater than the number of lines in a document, it defaults back to the number of lines in the document.
	 * If a line number is negative, it defaults to 0.
	 */
	Line float64 `json:"line"`

	/*Character defined:
	 * Character offset on a line in a document (zero-based). Assuming that the line is
	 * represented as a string, the `character` value represents the gap between the
	 * `character` and `character + 1`.
	 *
	 * If the character value is greater than the line length it defaults back to the
	 * line length.
	 * If a line number is negative, it defaults to 0.
	 */
	Character float64 `json:"character"`
}

/*Range defined:
 * A range in a text document expressed as (zero-based) start and end positions.
 *
 * If you want to specify a range that contains a line including the line ending
 * character(s) then use an end position denoting the start of the next line.
 * For example:
 * ```ts
 * {
 *     start: { line: 5, character: 23 }
 *     end : { line 6, character : 0 }
 * }
 * ```
 */
type Range struct {

	/*Start defined:
	 * The range's start position
	 */
	Start Position `json:"start"`

	/*End defined:
	 * The range's end position.
	 */
	End Position `json:"end"`
}

/*Location defined:
 * Represents a location inside a resource, such as a line
 * inside a text file.
 */
type Location struct {

	// URI is
	URI DocumentURI `json:"uri"`

	// Range is
	Range Range `json:"range"`
}

/*LocationLink defined:
 * Represents the connection of two locations. Provides additional metadata over normal [locations](#Location),
 * including an origin range.
 */
type LocationLink struct {

	/*OriginSelectionRange defined:
	 * Span of the origin of this link.
	 *
	 * Used as the underlined span for mouse definition hover. Defaults to the word range at
	 * the definition position.
	 */
	OriginSelectionRange *Range `json:"originSelectionRange,omitempty"`

	/*TargetURI defined:
	 * The target resource identifier of this link.
	 */
	TargetURI DocumentURI `json:"targetUri"`

	/*TargetRange defined:
	 * The full target range of this link. If the target for example is a symbol then target range is the
	 * range enclosing this symbol not including leading/trailing whitespace but everything else
	 * like comments. This information is typically used to highlight the range in the editor.
	 */
	TargetRange Range `json:"targetRange"`

	/*TargetSelectionRange defined:
	 * The range that should be selected and revealed when this link is being followed, e.g the name of a function.
	 * Must be contained by the the `targetRange`. See also `DocumentSymbol#range`
	 */
	TargetSelectionRange Range `json:"targetSelectionRange"`
}

/*Color defined:
 * Represents a color in RGBA space.
 */
type Color struct {

	/*Red defined:
	 * The red component of this color in the range [0-1].
	 */
	Red float64 `json:"red"`

	/*Green defined:
	 * The green component of this color in the range [0-1].
	 */
	Green float64 `json:"green"`

	/*Blue defined:
	 * The blue component of this color in the range [0-1].
	 */
	Blue float64 `json:"blue"`

	/*Alpha defined:
	 * The alpha component of this color in the range [0-1].
	 */
	Alpha float64 `json:"alpha"`
}

/*ColorInformation defined:
 * Represents a color range from a document.
 */
type ColorInformation struct {

	/*Range defined:
	 * The range in the document where this color appers.
	 */
	Range Range `json:"range"`

	/*Color defined:
	 * The actual color value for this color range.
	 */
	Color Color `json:"color"`
}

// ColorPresentation is
type ColorPresentation struct {

	/*Label defined:
	 * The label of this color presentation. It will be shown on the color
	 * picker header. By default this is also the text that is inserted when selecting
	 * this color presentation.
	 */
	Label string `json:"label"`

	/*TextEdit defined:
	 * An [edit](#TextEdit) which is applied to a document when selecting
	 * this presentation for the color.  When `falsy` the [label](#ColorPresentation.label)
	 * is used.
	 */
	TextEdit *TextEdit `json:"textEdit,omitempty"`

	/*AdditionalTextEdits defined:
	 * An optional array of additional [text edits](#TextEdit) that are applied when
	 * selecting this color presentation. Edits must not overlap with the main [edit](#ColorPresentation.textEdit) nor with themselves.
	 */
	AdditionalTextEdits []TextEdit `json:"additionalTextEdits,omitempty"`
}

/*DiagnosticRelatedInformation defined:
 * Represents a related message and source code location for a diagnostic. This should be
 * used to point to code locations that cause or related to a diagnostics, e.g when duplicating
 * a symbol in a scope.
 */
type DiagnosticRelatedInformation struct {

	/*Location defined:
	 * The location of this related diagnostic information.
	 */
	Location Location `json:"location"`

	/*Message defined:
	 * The message of this related diagnostic information.
	 */
	Message string `json:"message"`
}

/*Diagnostic defined:
 * Represents a diagnostic, such as a compiler error or warning. Diagnostic objects
 * are only valid in the scope of a resource.
 */
type Diagnostic struct {

	/*Range defined:
	 * The range at which the message applies
	 */
	Range Range `json:"range"`

	/*Severity defined:
	 * The diagnostic's severity. Can be omitted. If omitted it is up to the
	 * client to interpret diagnostics as error, warning, info or hint.
	 */
	Severity DiagnosticSeverity `json:"severity,omitempty"`

	/*Code defined:
	 * The diagnostic's code, which usually appear in the user interface.
	 */
	Code interface{} `json:"code,omitempty"` // number | string

	/*Source defined:
	 * A human-readable string describing the source of this
	 * diagnostic, e.g. 'typescript' or 'super lint'. It usually
	 * appears in the user interface.
	 */
	Source string `json:"source,omitempty"`

	/*Message defined:
	 * The diagnostic's message. It usually appears in the user interface
	 */
	Message string `json:"message"`

	/*Tags defined:
	 * Additional metadata about the diagnostic.
	 */
	Tags []DiagnosticTag `json:"tags,omitempty"`

	/*RelatedInformation defined:
	 * An array of related diagnostic information, e.g. when symbol-names within
	 * a scope collide all definitions can be marked via this property.
	 */
	RelatedInformation []DiagnosticRelatedInformation `json:"relatedInformation,omitempty"`
}

/*Command defined:
 * Represents a reference to a command. Provides a title which
 * will be used to represent a command in the UI and, optionally,
 * an array of arguments which will be passed to the command handler
 * function when invoked.
 */
type Command struct {

	/*Title defined:
	 * Title of the command, like `save`.
	 */
	Title string `json:"title"`

	/*Command defined:
	 * The identifier of the actual command handler.
	 */
	Command string `json:"command"`

	/*Arguments defined:
	 * Arguments that the command handler should be
	 * invoked with.
	 */
	Arguments []interface{} `json:"arguments,omitempty"`
}

/*TextEdit defined:
 * A text edit applicable to a text document.
 */
type TextEdit struct {

	/*Range defined:
	 * The range of the text document to be manipulated. To insert
	 * text into a document create a range where start === end.
	 */
	Range Range `json:"range"`

	/*NewText defined:
	 * The string to be inserted. For delete operations use an
	 * empty string.
	 */
	NewText string `json:"newText"`
}

/*TextDocumentEdit defined:
 * Describes textual changes on a text document.
 */
type TextDocumentEdit struct {

	/*TextDocument defined:
	 * The text document to change.
	 */
	TextDocument VersionedTextDocumentIdentifier `json:"textDocument"`

	/*Edits defined:
	 * The edits to be applied.
	 */
	Edits []TextEdit `json:"edits"`
}

// ResourceOperation is
type ResourceOperation struct {

	// Kind is
	Kind string `json:"kind"`
}

/*CreateFileOptions defined:
 * Options to create a file.
 */
type CreateFileOptions struct {

	/*Overwrite defined:
	 * Overwrite existing file. Overwrite wins over `ignoreIfExists`
	 */
	Overwrite bool `json:"overwrite,omitempty"`

	/*IgnoreIfExists defined:
	 * Ignore if exists.
	 */
	IgnoreIfExists bool `json:"ignoreIfExists,omitempty"`
}

/*CreateFile defined:
 * Create file operation.
 */
type CreateFile struct {

	/*Kind defined:
	 * A create
	 */
	Kind string `json:"kind"` // 'create'

	/*URI defined:
	 * The resource to create.
	 */
	URI DocumentURI `json:"uri"`

	/*Options defined:
	 * Additional options
	 */
	Options *CreateFileOptions `json:"options,omitempty"`
}

/*RenameFileOptions defined:
 * Rename file options
 */
type RenameFileOptions struct {

	/*Overwrite defined:
	 * Overwrite target if existing. Overwrite wins over `ignoreIfExists`
	 */
	Overwrite bool `json:"overwrite,omitempty"`

	/*IgnoreIfExists defined:
	 * Ignores if target exists.
	 */
	IgnoreIfExists bool `json:"ignoreIfExists,omitempty"`
}

/*RenameFile defined:
 * Rename file operation
 */
type RenameFile struct {

	/*Kind defined:
	 * A rename
	 */
	Kind string `json:"kind"` // 'rename'

	/*OldURI defined:
	 * The old (existing) location.
	 */
	OldURI DocumentURI `json:"oldUri"`

	/*NewURI defined:
	 * The new location.
	 */
	NewURI DocumentURI `json:"newUri"`

	/*Options defined:
	 * Rename options.
	 */
	Options *RenameFileOptions `json:"options,omitempty"`
}

/*DeleteFileOptions defined:
 * Delete file options
 */
type DeleteFileOptions struct {

	/*Recursive defined:
	 * Delete the content recursively if a folder is denoted.
	 */
	Recursive bool `json:"recursive,omitempty"`

	/*IgnoreIfNotExists defined:
	 * Ignore the operation if the file doesn't exist.
	 */
	IgnoreIfNotExists bool `json:"ignoreIfNotExists,omitempty"`
}

/*DeleteFile defined:
 * Delete file operation
 */
type DeleteFile struct {

	/*Kind defined:
	 * A delete
	 */
	Kind string `json:"kind"` // 'delete'

	/*URI defined:
	 * The file to delete.
	 */
	URI DocumentURI `json:"uri"`

	/*Options defined:
	 * Delete options.
	 */
	Options *DeleteFileOptions `json:"options,omitempty"`
}

/*WorkspaceEdit defined:
 * A workspace edit represents changes to many resources managed in the workspace. The edit
 * should either provide `changes` or `documentChanges`. If documentChanges are present
 * they are preferred over `changes` if the client can handle versioned document edits.
 */
type WorkspaceEdit struct {

	/*Changes defined:
	 * Holds changes to existing resources.
	 */
	Changes *map[string][]TextEdit `json:"changes,omitempty"` // [uri: string]: TextEdit[];

	/*DocumentChanges defined:
	 * Depending on the client capability `workspace.workspaceEdit.resourceOperations` document changes
	 * are either an array of `TextDocumentEdit`s to express changes to n different text documents
	 * where each text document edit addresses a specific version of a text document. Or it can contain
	 * above `TextDocumentEdit`s mixed with create, rename and delete file / folder operations.
	 *
	 * Whether a client supports versioned document edits is expressed via
	 * `workspace.workspaceEdit.documentChanges` client capability.
	 *
	 * If a client neither supports `documentChanges` nor `workspace.workspaceEdit.resourceOperations` then
	 * only plain `TextEdit`s using the `changes` property are supported.
	 */
	DocumentChanges []TextDocumentEdit `json:"documentChanges,omitempty"` // (TextDocumentEdit | CreateFile | RenameFile | DeleteFile)
}

/*TextEditChange defined:
 * A change to capture text edits for existing resources.
 */
type TextEditChange struct {
}

/*TextDocumentIdentifier defined:
 * A literal to identify a text document in the client.
 */
type TextDocumentIdentifier struct {

	/*URI defined:
	 * The text document's uri.
	 */
	URI DocumentURI `json:"uri"`
}

/*VersionedTextDocumentIdentifier defined:
 * An identifier to denote a specific version of a text document.
 */
type VersionedTextDocumentIdentifier struct {

	/*Version defined:
	 * The version number of this document. If a versioned text document identifier
	 * is sent from the server to the client and the file is not open in the editor
	 * (the server has not received an open notification before) the server can send
	 * `null` to indicate that the version is unknown and the content on disk is the
	 * truth (as speced with document content ownership).
	 */
	Version float64 `json:"version"`
	TextDocumentIdentifier
}

/*TextDocumentItem defined:
 * An item to transfer a text document from the client to the
 * server.
 */
type TextDocumentItem struct {

	/*URI defined:
	 * The text document's uri.
	 */
	URI DocumentURI `json:"uri"`

	/*LanguageID defined:
	 * The text document's language identifier
	 */
	LanguageID string `json:"languageId"`

	/*Version defined:
	 * The version number of this document (it will increase after each
	 * change, including undo/redo).
	 */
	Version float64 `json:"version"`

	/*Text defined:
	 * The content of the opened text document.
	 */
	Text string `json:"text"`
}

/*MarkupContent defined:
 * A `MarkupContent` literal represents a string value which content is interpreted base on its
 * kind flag. Currently the protocol supports `plaintext` and `markdown` as markup kinds.
 *
 * If the kind is `markdown` then the value can contain fenced code blocks like in GitHub issues.
 * See https://help.github.com/articles/creating-and-highlighting-code-blocks/#syntax-highlighting
 *
 * Here is an example how such a string can be constructed using JavaScript / TypeScript:
 * ```ts
 * let markdown: MarkdownContent = {
 *  kind: MarkupKind.Markdown,
 *	value: [
 *		'# Header',
 *		'Some text',
 *		'```typescript',
 *		'someCode();',
 *		'```'
 *	].join('\n')
 * };
 * ```
 *
 * *Please Note* that clients might sanitize the return markdown. A client could decide to
 * remove HTML from the markdown to avoid script execution.
 */
type MarkupContent struct {

	/*Kind defined:
	 * The type of the Markup
	 */
	Kind MarkupKind `json:"kind"`

	/*Value defined:
	 * The content itself
	 */
	Value string `json:"value"`
}

/*CompletionItem defined:
 * A completion item represents a text snippet that is
 * proposed to complete text that is being typed.
 */
type CompletionItem struct {

	/*Label defined:
	 * The label of this completion item. By default
	 * also the text that is inserted when selecting
	 * this completion.
	 */
	Label string `json:"label"`

	/*Kind defined:
	 * The kind of this completion item. Based of the kind
	 * an icon is chosen by the editor.
	 */
	Kind CompletionItemKind `json:"kind,omitempty"`

	/*Tags defined:
	 * Tags for this completion item.
	 *
	 * @since 3.15.0
	 */
	Tags []CompletionItemTag `json:"tags,omitempty"`

	/*Detail defined:
	 * A human-readable string with additional information
	 * about this item, like type or symbol information.
	 */
	Detail string `json:"detail,omitempty"`

	/*Documentation defined:
	 * A human-readable string that represents a doc-comment.
	 */
	Documentation string `json:"documentation,omitempty"` // string | MarkupContent

	/*Deprecated defined:
	 * Indicates if this item is deprecated.
	 * @deprecated Use `tags` instead.
	 */
	Deprecated bool `json:"deprecated,omitempty"`

	/*Preselect defined:
	 * Select this item when showing.
	 *
	 * *Note* that only one completion item can be selected and that the
	 * tool / client decides which item that is. The rule is that the *first*
	 * item of those that match best is selected.
	 */
	Preselect bool `json:"preselect,omitempty"`

	/*SortText defined:
	 * A string that should be used when comparing this item
	 * with other items. When `falsy` the [label](#CompletionItem.label)
	 * is used.
	 */
	SortText string `json:"sortText,omitempty"`

	/*FilterText defined:
	 * A string that should be used when filtering a set of
	 * completion items. When `falsy` the [label](#CompletionItem.label)
	 * is used.
	 */
	FilterText string `json:"filterText,omitempty"`

	/*InsertText defined:
	 * A string that should be inserted into a document when selecting
	 * this completion. When `falsy` the [label](#CompletionItem.label)
	 * is used.
	 *
	 * The `insertText` is subject to interpretation by the client side.
	 * Some tools might not take the string literally. For example
	 * VS Code when code complete is requested in this example `con<cursor position>`
	 * and a completion item with an `insertText` of `console` is provided it
	 * will only insert `sole`. Therefore it is recommended to use `textEdit` instead
	 * since it avoids additional client side interpretation.
	 */
	InsertText string `json:"insertText,omitempty"`

	/*InsertTextFormat defined:
	 * The format of the insert text. The format applies to both the `insertText` property
	 * and the `newText` property of a provided `textEdit`.
	 */
	InsertTextFormat InsertTextFormat `json:"insertTextFormat,omitempty"`

	/*TextEdit defined:
	 * An [edit](#TextEdit) which is applied to a document when selecting
	 * this completion. When an edit is provided the value of
	 * [insertText](#CompletionItem.insertText) is ignored.
	 *
	 * *Note:* The text edit's range must be a [single line] and it must contain the position
	 * at which completion has been requested.
	 */
	TextEdit *TextEdit `json:"textEdit,omitempty"`

	/*AdditionalTextEdits defined:
	 * An optional array of additional [text edits](#TextEdit) that are applied when
	 * selecting this completion. Edits must not overlap (including the same insert position)
	 * with the main [edit](#CompletionItem.textEdit) nor with themselves.
	 *
	 * Additional text edits should be used to change text unrelated to the current cursor position
	 * (for example adding an import statement at the top of the file if the completion item will
	 * insert an unqualified type).
	 */
	AdditionalTextEdits []TextEdit `json:"additionalTextEdits,omitempty"`

	/*CommitCharacters defined:
	 * An optional set of characters that when pressed while this completion is active will accept it first and
	 * then type that character. *Note* that all commit characters should have `length=1` and that superfluous
	 * characters will be ignored.
	 */
	CommitCharacters []string `json:"commitCharacters,omitempty"`

	/*Command defined:
	 * An optional [command](#Command) that is executed *after* inserting this completion. *Note* that
	 * additional modifications to the current document should be described with the
	 * [additionalTextEdits](#CompletionItem.additionalTextEdits)-property.
	 */
	Command *Command `json:"command,omitempty"`

	/*Data defined:
	 * An data entry field that is preserved on a completion item between
	 * a [CompletionRequest](#CompletionRequest) and a [CompletionResolveRequest]
	 * (#CompletionResolveRequest)
	 */
	Data interface{} `json:"data,omitempty"`
}

/*CompletionList defined:
 * Represents a collection of [completion items](#CompletionItem) to be presented
 * in the editor.
 */
type CompletionList struct {

	/*IsIncomplete defined:
	 * This list it not complete. Further typing results in recomputing this list.
	 */
	IsIncomplete bool `json:"isIncomplete"`

	/*Items defined:
	 * The completion items.
	 */
	Items []CompletionItem `json:"items"`
}

/*Hover defined:
 * The result of a hover request.
 */
type Hover struct {

	/*Contents defined:
	 * The hover's content
	 */
	Contents MarkupContent `json:"contents"` // MarkupContent | MarkedString | MarkedString[]

	/*Range defined:
	 * An optional range
	 */
	Range *Range `json:"range,omitempty"`
}

/*ParameterInformation defined:
 * Represents a parameter of a callable-signature. A parameter can
 * have a label and a doc-comment.
 */
type ParameterInformation struct {

	/*Label defined:
	 * The label of this parameter information.
	 *
	 * Either a string or an inclusive start and exclusive end offsets within its containing
	 * signature label. (see SignatureInformation.label). The offsets are based on a UTF-16
	 * string representation as `Position` and `Range` does.
	 *
	 * *Note*: a label of type string should be a substring of its containing signature label.
	 * Its intended use case is to highlight the parameter label part in the `SignatureInformation.label`.
	 */
	Label string `json:"label"` // string | [number, number]

	/*Documentation defined:
	 * The human-readable doc-comment of this signature. Will be shown
	 * in the UI but can be omitted.
	 */
	Documentation string `json:"documentation,omitempty"` // string | MarkupContent
}

/*SignatureInformation defined:
 * Represents the signature of something callable. A signature
 * can have a label, like a function-name, a doc-comment, and
 * a set of parameters.
 */
type SignatureInformation struct {

	/*Label defined:
	 * The label of this signature. Will be shown in
	 * the UI.
	 */
	Label string `json:"label"`

	/*Documentation defined:
	 * The human-readable doc-comment of this signature. Will be shown
	 * in the UI but can be omitted.
	 */
	Documentation string `json:"documentation,omitempty"` // string | MarkupContent

	/*Parameters defined:
	 * The parameters of this signature.
	 */
	Parameters []ParameterInformation `json:"parameters,omitempty"`
}

/*SignatureHelp defined:
 * Signature help represents the signature of something
 * callable. There can be multiple signature but only one
 * active and only one active parameter.
 */
type SignatureHelp struct {

	/*Signatures defined:
	 * One or more signatures.
	 */
	Signatures []SignatureInformation `json:"signatures"`

	/*ActiveSignature defined:
	 * The active signature. Set to `null` if no
	 * signatures exist.
	 */
	ActiveSignature float64 `json:"activeSignature"`

	/*ActiveParameter defined:
	 * The active parameter of the active signature. Set to `null`
	 * if the active signature has no parameters.
	 */
	ActiveParameter float64 `json:"activeParameter"`
}

/*ReferenceContext defined:
 * Value-object that contains additional information when
 * requesting references.
 */
type ReferenceContext struct {

	/*IncludeDeclaration defined:
	 * Include the declaration of the current symbol.
	 */
	IncludeDeclaration bool `json:"includeDeclaration"`
}

/*DocumentHighlight defined:
 * A document highlight is a range inside a text document which deserves
 * special attention. Usually a document highlight is visualized by changing
 * the background color of its range.
 */
type DocumentHighlight struct {

	/*Range defined:
	 * The range this highlight applies to.
	 */
	Range Range `json:"range"`

	/*Kind defined:
	 * The highlight kind, default is [text](#DocumentHighlightKind.Text).
	 */
	Kind *DocumentHighlightKind `json:"kind,omitempty"`
}

/*SymbolInformation defined:
 * Represents information about programming constructs like variables, classes,
 * interfaces etc.
 */
type SymbolInformation struct {

	/*Name defined:
	 * The name of this symbol.
	 */
	Name string `json:"name"`

	/*Kind defined:
	 * The kind of this symbol.
	 */
	Kind SymbolKind `json:"kind"`

	/*Deprecated defined:
	 * Indicates if this symbol is deprecated.
	 */
	Deprecated bool `json:"deprecated,omitempty"`

	/*Location defined:
	 * The location of this symbol. The location's range is used by a tool
	 * to reveal the location in the editor. If the symbol is selected in the
	 * tool the range's start information is used to position the cursor. So
	 * the range usually spans more than the actual symbol's name and does
	 * normally include thinks like visibility modifiers.
	 *
	 * The range doesn't have to denote a node range in the sense of a abstract
	 * syntax tree. It can therefore not be used to re-construct a hierarchy of
	 * the symbols.
	 */
	Location Location `json:"location"`

	/*ContainerName defined:
	 * The name of the symbol containing this symbol. This information is for
	 * user interface purposes (e.g. to render a qualifier in the user interface
	 * if necessary). It can't be used to re-infer a hierarchy for the document
	 * symbols.
	 */
	ContainerName string `json:"containerName,omitempty"`
}

/*DocumentSymbol defined:
 * Represents programming constructs like variables, classes, interfaces etc.
 * that appear in a document. Document symbols can be hierarchical and they
 * have two ranges: one that encloses its definition and one that points to
 * its most interesting range, e.g. the range of an identifier.
 */
type DocumentSymbol struct {

	/*Name defined:
	 * The name of this symbol. Will be displayed in the user interface and therefore must not be
	 * an empty string or a string only consisting of white spaces.
	 */
	Name string `json:"name"`

	/*Detail defined:
	 * More detail for this symbol, e.g the signature of a function.
	 */
	Detail string `json:"detail,omitempty"`

	/*Kind defined:
	 * The kind of this symbol.
	 */
	Kind SymbolKind `json:"kind"`

	/*Deprecated defined:
	 * Indicates if this symbol is deprecated.
	 */
	Deprecated bool `json:"deprecated,omitempty"`

	/*Range defined:
	 * The range enclosing this symbol not including leading/trailing whitespace but everything else
	 * like comments. This information is typically used to determine if the the clients cursor is
	 * inside the symbol to reveal in the symbol in the UI.
	 */
	Range Range `json:"range"`

	/*SelectionRange defined:
	 * The range that should be selected and revealed when this symbol is being picked, e.g the name of a function.
	 * Must be contained by the the `range`.
	 */
	SelectionRange Range `json:"selectionRange"`

	/*Children defined:
	 * Children of this symbol, e.g. properties of a class.
	 */
	Children []DocumentSymbol `json:"children,omitempty"`
}

/*CodeActionContext defined:
 * Contains additional diagnostic information about the context in which
 * a [code action](#CodeActionProvider.provideCodeActions) is run.
 */
type CodeActionContext struct {

	/*Diagnostics defined:
	 * An array of diagnostics known on the client side overlapping the range provided to the
	 * `textDocument/codeAction` request. They are provied so that the server knows which
	 * errors are currently presented to the user for the given range. There is no guarantee
	 * that these accurately reflect the error state of the resource. The primary parameter
	 * to compute code actions is the provided range.
	 */
	Diagnostics []Diagnostic `json:"diagnostics"`

	/*Only defined:
	 * Requested kind of actions to return.
	 *
	 * Actions not of this kind are filtered out by the client before being shown. So servers
	 * can omit computing them.
	 */
	Only []CodeActionKind `json:"only,omitempty"`
}

/*CodeAction defined:
 * A code action represents a change that can be performed in code, e.g. to fix a problem or
 * to refactor code.
 *
 * A CodeAction must set either `edit` and/or a `command`. If both are supplied, the `edit` is applied first, then the `command` is executed.
 */
type CodeAction struct {

	/*Title defined:
	 * A short, human-readable, title for this code action.
	 */
	Title string `json:"title"`

	/*Kind defined:
	 * The kind of the code action.
	 *
	 * Used to filter code actions.
	 */
	Kind CodeActionKind `json:"kind,omitempty"`

	/*Diagnostics defined:
	 * The diagnostics that this code action resolves.
	 */
	Diagnostics []Diagnostic `json:"diagnostics,omitempty"`

	/*IsPreferred defined:
	 * Marks this as a preferred action. Preferred actions are used by the `auto fix` command and can be targeted
	 * by keybindings.
	 *
	 * A quick fix should be marked preferred if it properly addresses the underlying error.
	 * A refactoring should be marked preferred if it is the most reasonable choice of actions to take.
	 *
	 * @since 3.15.0
	 */
	IsPreferred bool `json:"isPreferred,omitempty"`

	/*Edit defined:
	 * The workspace edit this code action performs.
	 */
	Edit *WorkspaceEdit `json:"edit,omitempty"`

	/*Command defined:
	 * A command this code action executes. If a code action
	 * provides a edit and a command, first the edit is
	 * executed and then the command.
	 */
	Command *Command `json:"command,omitempty"`
}

/*CodeLens defined:
 * A code lens represents a [command](#Command) that should be shown along with
 * source text, like the number of references, a way to run tests, etc.
 *
 * A code lens is _unresolved_ when no command is associated to it. For performance
 * reasons the creation of a code lens and resolving should be done to two stages.
 */
type CodeLens struct {

	/*Range defined:
	 * The range in which this code lens is valid. Should only span a single line.
	 */
	Range Range `json:"range"`

	/*Command defined:
	 * The command this code lens represents.
	 */
	Command *Command `json:"command,omitempty"`

	/*Data defined:
	 * An data entry field that is preserved on a code lens item between
	 * a [CodeLensRequest](#CodeLensRequest) and a [CodeLensResolveRequest]
	 * (#CodeLensResolveRequest)
	 */
	Data interface{} `json:"data,omitempty"`
}

/*FormattingOptions defined:
 * Value-object describing what options formatting should use.
 */
type FormattingOptions struct {

	/*TabSize defined:
	 * Size of a tab in spaces.
	 */
	TabSize float64 `json:"tabSize"`

	/*InsertSpaces defined:
	 * Prefer spaces over tabs.
	 */
	InsertSpaces bool `json:"insertSpaces"`

	/*TrimTrailingWhitespace defined:
	 * Trim trailing whitespaces on a line.
	 *
	 * @since 3.15.0
	 */
	TrimTrailingWhitespace bool `json:"trimTrailingWhitespace,omitempty"`

	/*InsertFinalNewline defined:
	 * Insert a newline character at the end of the file if one does not exist.
	 *
	 * @since 3.15.0
	 */
	InsertFinalNewline bool `json:"insertFinalNewline,omitempty"`

	/*TrimFinalNewlines defined:
	 * Trim all newlines after the final newline at the end of the file.
	 *
	 * @since 3.15.0
	 */
	TrimFinalNewlines bool `json:"trimFinalNewlines,omitempty"`

	/*Key defined:
	 * Signature for further properties.
	 */
	Key map[string]bool `json:"key"` // [key: string]: boolean | number | string | undefined;
}

/*DocumentLink defined:
 * A document link is a range in a text document that links to an internal or external resource, like another
 * text document or a web site.
 */
type DocumentLink struct {

	/*Range defined:
	 * The range this link applies to.
	 */
	Range Range `json:"range"`

	/*Target defined:
	 * The uri this link points to.
	 */
	Target string `json:"target,omitempty"`

	/*Tooltip defined:
	 * The tooltip text when you hover over this link.
	 *
	 * If a tooltip is provided, is will be displayed in a string that includes instructions on how to
	 * trigger the link, such as `{0} (ctrl + click)`. The specific instructions vary depending on OS,
	 * user settings, and localization.
	 *
	 * @since 3.15.0
	 */
	Tooltip string `json:"tooltip,omitempty"`

	/*Data defined:
	 * A data entry field that is preserved on a document link between a
	 * DocumentLinkRequest and a DocumentLinkResolveRequest.
	 */
	Data interface{} `json:"data,omitempty"`
}

/*SelectionRange defined:
 * A selection range represents a part of a selection hierarchy. A selection range
 * may have a parent selection range that contains it.
 */
type SelectionRange struct {

	/*Range defined:
	 * The [range](#Range) of this selection range.
	 */
	Range Range `json:"range"`

	/*Parent defined:
	 * The parent selection range containing this range. Therefore `parent.range` must contain `this.range`.
	 */
	Parent *SelectionRange `json:"parent,omitempty"`
}

/*TextDocument defined:
 * A simple text document. Not to be implemented.
 */
type TextDocument struct {

	/*URI defined:
	 * The associated URI for this document. Most documents have the __file__-scheme, indicating that they
	 * represent files on disk. However, some documents may have other schemes indicating that they are not
	 * available on disk.
	 *
	 * @readonly
	 */
	URI DocumentURI `json:"uri"`

	/*LanguageID defined:
	 * The identifier of the language associated with this document.
	 *
	 * @readonly
	 */
	LanguageID string `json:"languageId"`

	/*Version defined:
	 * The version number of this document (it will increase after each
	 * change, including undo/redo).
	 *
	 * @readonly
	 */
	Version float64 `json:"version"`

	/*LineCount defined:
	 * The number of lines in this document.
	 *
	 * @readonly
	 */
	LineCount float64 `json:"lineCount"`
}

/*TextDocumentChangeEvent defined:
 * Event to signal changes to a simple text document.
 */
type TextDocumentChangeEvent struct {

	/*Document defined:
	 * The document that has changed.
	 */
	Document TextDocument `json:"document"`
}

// TextDocumentWillSaveEvent is
type TextDocumentWillSaveEvent struct {

	/*Document defined:
	 * The document that will be saved
	 */
	Document TextDocument `json:"document"`

	/*Reason defined:
	 * The reason why save was triggered.
	 */
	Reason TextDocumentSaveReason `json:"reason"`
}

/*TextDocumentContentChangeEvent defined:
 * An event describing a change to a text document. If range and rangeLength are omitted
 * the new text is considered to be the full content of the document.
 */
type TextDocumentContentChangeEvent struct {

	/*Range defined:
	 * The range of the document that changed.
	 */
	Range *Range `json:"range,omitempty"`

	/*RangeLength defined:
	 * The length of the range that got replaced.
	 */
	RangeLength float64 `json:"rangeLength,omitempty"`

	/*Text defined:
	 * The new text of the document.
	 */
	Text string `json:"text"`
}

// ProgressParams is
type ProgressParams struct {

	/*Token defined:
	 * The progress token provided by the client or server.
	 */
	Token ProgressToken `json:"token"`

	/*Value defined:
	 * The progress data.
	 */
	Value interface{} `json:"value"`
}

// SetTraceParams is
type SetTraceParams struct {

	// Value is
	Value TraceValues `json:"value"`
}

// LogTraceParams is
type LogTraceParams struct {

	// Message is
	Message string `json:"message"`

	// Verbose is
	Verbose string `json:"verbose,omitempty"`
}

// Tracer is
type Tracer struct {
}

// FoldingRangeKind defines constants
type FoldingRangeKind string

// ResourceOperationKind defines constants
type ResourceOperationKind string

// FailureHandlingKind defines constants
type FailureHandlingKind string

// InitializeError defines constants
type InitializeError float64

// MessageType defines constants
type MessageType float64

// TextDocumentSyncKind defines constants
type TextDocumentSyncKind float64

// FileChangeType defines constants
type FileChangeType float64

// WatchKind defines constants
type WatchKind float64

// CompletionTriggerKind defines constants
type CompletionTriggerKind float64

// SignatureHelpTriggerKind defines constants
type SignatureHelpTriggerKind float64

// DiagnosticSeverity defines constants
type DiagnosticSeverity float64

// DiagnosticTag defines constants
type DiagnosticTag float64

// MarkupKind defines constants
type MarkupKind string

// CompletionItemKind defines constants
type CompletionItemKind float64

// InsertTextFormat defines constants
type InsertTextFormat float64

// CompletionItemTag defines constants
type CompletionItemTag float64

// DocumentHighlightKind defines constants
type DocumentHighlightKind float64

// SymbolKind defines constants
type SymbolKind float64

// CodeActionKind defines constants
type CodeActionKind string

// TextDocumentSaveReason defines constants
type TextDocumentSaveReason float64

// ErrorCodes defines constants
type ErrorCodes float64

// Touch defines constants
type Touch float64

// Trace defines constants
type Trace string

// TraceFormat defines constants
type TraceFormat string

// ConnectionErrors defines constants
type ConnectionErrors float64

// ConnectionState defines constants
type ConnectionState float64

const (

	/*Comment defined:
	 * Folding range for a comment
	 */
	Comment FoldingRangeKind = "comment"

	/*Imports defined:
	 * Folding range for a imports or includes
	 */
	Imports FoldingRangeKind = "imports"

	/*Region defined:
	 * Folding range for a region (e.g. `#region`)
	 */
	Region FoldingRangeKind = "region"

	/*Create defined:
	 * Supports creating new files and folders.
	 */
	Create ResourceOperationKind = "create"

	/*Rename defined:
	 * Supports renaming existing files and folders.
	 */
	Rename ResourceOperationKind = "rename"

	/*Delete defined:
	 * Supports deleting existing files and folders.
	 */
	Delete ResourceOperationKind = "delete"

	/*Abort defined:
	 * Applying the workspace change is simply aborted if one of the changes provided
	 * fails. All operations executed before the failing operation stay executed.
	 */
	Abort FailureHandlingKind = "abort"

	/*Transactional defined:
	 * All operations are executed transactional. That means they either all
	 * succeed or no changes at all are applied to the workspace.
	 */
	Transactional FailureHandlingKind = "transactional"

	/*TextOnlyTransactional defined:
	 * If the workspace edit contains only textual file changes they are executed transactional.
	 * If resource changes (create, rename or delete file) are part of the change the failure
	 * handling startegy is abort.
	 */
	TextOnlyTransactional FailureHandlingKind = "textOnlyTransactional"

	/*Undo defined:
	 * The client tries to undo the operations already executed. But there is no
	 * guaruntee that this is succeeding.
	 */
	Undo FailureHandlingKind = "undo"

	/*UnknownProtocolVersion defined:
	 * If the protocol version provided by the client can't be handled by the server.
	 * @deprecated This initialize error got replaced by client capabilities. There is
	 * no version handshake in version 3.0x
	 */
	UnknownProtocolVersion InitializeError = 1

	/*Error defined:
	 * An error message.
	 */
	Error MessageType = 1

	/*Warning defined:
	 * A warning message.
	 */
	Warning MessageType = 2

	/*Info defined:
	 * An information message.
	 */
	Info MessageType = 3

	/*Log defined:
	 * A log message.
	 */
	Log MessageType = 4

	/*None defined:
	 * Documents should not be synced at all.
	 */
	None TextDocumentSyncKind = 0

	/*Full defined:
	 * Documents are synced by always sending the full content
	 * of the document.
	 */
	Full TextDocumentSyncKind = 1

	/*Incremental defined:
	 * Documents are synced by sending the full content on open.
	 * After that only incremental updates to the document are
	 * send.
	 */
	Incremental TextDocumentSyncKind = 2

	/*Created defined:
	 * The file got created.
	 */
	Created FileChangeType = 1

	/*Changed defined:
	 * The file got changed.
	 */
	Changed FileChangeType = 2

	/*Deleted defined:
	 * The file got deleted.
	 */
	Deleted FileChangeType = 3

	/*WatchCreate defined:
	 * Interested in create events.
	 */
	WatchCreate WatchKind = 1

	/*WatchChange defined:
	 * Interested in change events
	 */
	WatchChange WatchKind = 2

	/*WatchDelete defined:
	 * Interested in delete events
	 */
	WatchDelete WatchKind = 4

	/*Invoked defined:
	 * Completion was triggered by typing an identifier (24x7 code
	 * complete), manual invocation (e.g Ctrl+Space) or via API.
	 */
	Invoked CompletionTriggerKind = 1

	/*TriggerCharacter defined:
	 * Completion was triggered by a trigger character specified by
	 * the `triggerCharacters` properties of the `CompletionRegistrationOptions`.
	 */
	TriggerCharacter CompletionTriggerKind = 2

	/*TriggerForIncompleteCompletions defined:
	 * Completion was re-triggered as current completion list is incomplete
	 */
	TriggerForIncompleteCompletions CompletionTriggerKind = 3

	/*ContentChange defined:
	 * Signature help was triggered by the cursor moving or by the document content changing.
	 */
	ContentChange SignatureHelpTriggerKind = 3

	/*SeverityError defined:
	 * Reports an error.
	 */
	SeverityError DiagnosticSeverity = 1

	/*SeverityWarning defined:
	 * Reports a warning.
	 */
	SeverityWarning DiagnosticSeverity = 2

	/*SeverityInformation defined:
	 * Reports an information.
	 */
	SeverityInformation DiagnosticSeverity = 3

	/*SeverityHint defined:
	 * Reports a hint.
	 */
	SeverityHint DiagnosticSeverity = 4

	/*Unnecessary defined:
	 * Unused or unnecessary code.
	 *
	 * Clients are allowed to render diagnostics with this tag faded out instead of having
	 * an error squiggle.
	 */
	Unnecessary DiagnosticTag = 1

	/*Deprecated defined:
	 * Deprecated or obsolete code.
	 *
	 * Clients are allowed to rendered diagnostics with this tag strike through.
	 */
	Deprecated DiagnosticTag = 2

	/*PlainText defined:
	 * Plain text is supported as a content format
	 */
	PlainText MarkupKind = "plaintext"

	/*Markdown defined:
	 * Markdown is supported as a content format
	 */
	Markdown MarkupKind = "markdown"

	// TextCompletion is
	TextCompletion CompletionItemKind = 1

	// MethodCompletion is
	MethodCompletion CompletionItemKind = 2

	// FunctionCompletion is
	FunctionCompletion CompletionItemKind = 3

	// ConstructorCompletion is
	ConstructorCompletion CompletionItemKind = 4

	// FieldCompletion is
	FieldCompletion CompletionItemKind = 5

	// VariableCompletion is
	VariableCompletion CompletionItemKind = 6

	// ClassCompletion is
	ClassCompletion CompletionItemKind = 7

	// InterfaceCompletion is
	InterfaceCompletion CompletionItemKind = 8

	// ModuleCompletion is
	ModuleCompletion CompletionItemKind = 9

	// PropertyCompletion is
	PropertyCompletion CompletionItemKind = 10

	// UnitCompletion is
	UnitCompletion CompletionItemKind = 11

	// ValueCompletion is
	ValueCompletion CompletionItemKind = 12

	// EnumCompletion is
	EnumCompletion CompletionItemKind = 13

	// KeywordCompletion is
	KeywordCompletion CompletionItemKind = 14

	// SnippetCompletion is
	SnippetCompletion CompletionItemKind = 15

	// ColorCompletion is
	ColorCompletion CompletionItemKind = 16

	// FileCompletion is
	FileCompletion CompletionItemKind = 17

	// ReferenceCompletion is
	ReferenceCompletion CompletionItemKind = 18

	// FolderCompletion is
	FolderCompletion CompletionItemKind = 19

	// EnumMemberCompletion is
	EnumMemberCompletion CompletionItemKind = 20

	// ConstantCompletion is
	ConstantCompletion CompletionItemKind = 21

	// StructCompletion is
	StructCompletion CompletionItemKind = 22

	// EventCompletion is
	EventCompletion CompletionItemKind = 23

	// OperatorCompletion is
	OperatorCompletion CompletionItemKind = 24

	// TypeParameterCompletion is
	TypeParameterCompletion CompletionItemKind = 25

	/*PlainTextTextFormat defined:
	 * The primary text to be inserted is treated as a plain string.
	 */
	PlainTextTextFormat InsertTextFormat = 1

	/*SnippetTextFormat defined:
	 * The primary text to be inserted is treated as a snippet.
	 *
	 * A snippet can define tab stops and placeholders with `$1`, `$2`
	 * and `${3:foo}`. `$0` defines the final tab stop, it defaults to
	 * the end of the snippet. Placeholders with equal identifiers are linked,
	 * that is typing in one will update others too.
	 *
	 * See also: https://github.com/Microsoft/vscode/blob/master/src/vs/editor/contrib/snippet/common/snippet.md
	 */
	SnippetTextFormat InsertTextFormat = 2

	/*Text defined:
	 * A textual occurrence.
	 */
	Text DocumentHighlightKind = 1

	/*Read defined:
	 * Read-access of a symbol, like reading a variable.
	 */
	Read DocumentHighlightKind = 2

	/*Write defined:
	 * Write-access of a symbol, like writing to a variable.
	 */
	Write DocumentHighlightKind = 3

	// File is
	File SymbolKind = 1

	// Module is
	Module SymbolKind = 2

	// Namespace is
	Namespace SymbolKind = 3

	// Package is
	Package SymbolKind = 4

	// Class is
	Class SymbolKind = 5

	// Method is
	Method SymbolKind = 6

	// Property is
	Property SymbolKind = 7

	// Field is
	Field SymbolKind = 8

	// Constructor is
	Constructor SymbolKind = 9

	// Enum is
	Enum SymbolKind = 10

	// Interface is
	Interface SymbolKind = 11

	// Function is
	Function SymbolKind = 12

	// Variable is
	Variable SymbolKind = 13

	// Constant is
	Constant SymbolKind = 14

	// String is
	String SymbolKind = 15

	// Number is
	Number SymbolKind = 16

	// Boolean is
	Boolean SymbolKind = 17

	// Array is
	Array SymbolKind = 18

	// Object is
	Object SymbolKind = 19

	// Key is
	Key SymbolKind = 20

	// Null is
	Null SymbolKind = 21

	// EnumMember is
	EnumMember SymbolKind = 22

	// Struct is
	Struct SymbolKind = 23

	// Event is
	Event SymbolKind = 24

	// Operator is
	Operator SymbolKind = 25

	// TypeParameter is
	TypeParameter SymbolKind = 26

	/*Empty defined:
	 * Empty kind.
	 */
	Empty CodeActionKind = ""

	/*QuickFix defined:
	 * Base kind for quickfix actions: 'quickfix'
	 */
	QuickFix CodeActionKind = "quickfix"

	/*Refactor defined:
	 * Base kind for refactoring actions: 'refactor'
	 */
	Refactor CodeActionKind = "refactor"

	/*RefactorExtract defined:
	 * Base kind for refactoring extraction actions: 'refactor.extract'
	 *
	 * Example extract actions:
	 *
	 * - Extract method
	 * - Extract function
	 * - Extract variable
	 * - Extract interface from class
	 * - ...
	 */
	RefactorExtract CodeActionKind = "refactor.extract"

	/*RefactorInline defined:
	 * Base kind for refactoring inline actions: 'refactor.inline'
	 *
	 * Example inline actions:
	 *
	 * - Inline function
	 * - Inline variable
	 * - Inline constant
	 * - ...
	 */
	RefactorInline CodeActionKind = "refactor.inline"

	/*RefactorRewrite defined:
	 * Base kind for refactoring rewrite actions: 'refactor.rewrite'
	 *
	 * Example rewrite actions:
	 *
	 * - Convert JavaScript function to class
	 * - Add or remove parameter
	 * - Encapsulate field
	 * - Make method static
	 * - Move method to base class
	 * - ...
	 */
	RefactorRewrite CodeActionKind = "refactor.rewrite"

	/*Source defined:
	 * Base kind for source actions: `source`
	 *
	 * Source code actions apply to the entire file.
	 */
	Source CodeActionKind = "source"

	/*SourceOrganizeImports defined:
	 * Base kind for an organize imports source action: `source.organizeImports`
	 */
	SourceOrganizeImports CodeActionKind = "source.organizeImports"

	/*Manual defined:
	 * Manually triggered, e.g. by the user pressing save, by starting debugging,
	 * or by an API call.
	 */
	Manual TextDocumentSaveReason = 1

	/*AfterDelay defined:
	 * Automatic after a delay.
	 */
	AfterDelay TextDocumentSaveReason = 2

	/*FocusOut defined:
	 * When the editor lost focus.
	 */
	FocusOut TextDocumentSaveReason = 3

	// MessageWriteError is
	MessageWriteError ErrorCodes = 1

	// MessageReadError is
	MessageReadError ErrorCodes = 2

	// First is
	First Touch = 1

	// Last is
	Last Touch = 2

	// JSON is
	JSON TraceFormat = "json"

	/*Closed defined:
	 * The connection is closed.
	 */
	Closed ConnectionErrors = 1

	/*Disposed defined:
	 * The connection got disposed.
	 */
	Disposed ConnectionErrors = 2

	/*AlreadyListening defined:
	 * The connection is already in listening mode.
	 */
	AlreadyListening ConnectionErrors = 3

	// New is
	New ConnectionState = 1

	// Listening is
	Listening ConnectionState = 2
)

// DocumentFilter is a type
/**
 * A document filter denotes a document by different properties like
 * the [language](#TextDocument.languageId), the [scheme](#Uri.scheme) of
 * its resource, or a glob-pattern that is applied to the [path](#TextDocument.fileName).
 *
 * Glob patterns can have the following syntax:
 * - `*` to match one or more characters in a path segment
 * - `?` to match on one character in a path segment
 * - `**` to match any number of path segments, including none
 * - `{}` to group conditions (e.g. `**​/*.{ts,js}` matches all TypeScript and JavaScript files)
 * - `[]` to declare a range of characters to match in a path segment (e.g., `example.[0-9]` to match on `example.0`, `example.1`, …)
 * - `[!...]` to negate a range of characters to match in a path segment (e.g., `example.[!0-9]` to match on `example.a`, `example.b`, but not `example.0`)
 *
 * @sample A language filter that applies to typescript files on disk: `{ language: 'typescript', scheme: 'file' }`
 * @sample A language filter that applies to all package.json paths: `{ language: 'json', pattern: '**package.json' }`
 */
type DocumentFilter = struct {

	/*Language defined: A language id, like `typescript`. */
	Language string `json:"language,omitempty"`

	/*Scheme defined: A Uri [scheme](#Uri.scheme), like `file` or `untitled`. */
	Scheme string `json:"scheme,omitempty"`

	/*Pattern defined: A glob pattern, like `*.{ts,js}`. */
	Pattern string `json:"pattern,omitempty"`
}

// DocumentSelector is a type
/**
 * A document selector is the combination of one or many document filters.
 *
 * @sample `let sel:DocumentSelector = [{ language: 'typescript' }, { language: 'json', pattern: '**∕tsconfig.json' }]`;
 */
type DocumentSelector = []DocumentFilter

// DocumentURI is a type
/**
 * A tagging type for string properties that are actually URIs.
 */
type DocumentURI = string

// MarkedString is a type
/**
 * MarkedString can be used to render human readable text. It is either a markdown string
 * or a code-block that provides a language and a code snippet. The language identifier
 * is semantically equal to the optional language identifier in fenced code blocks in GitHub
 * issues. See https://help.github.com/articles/creating-and-highlighting-code-blocks/#syntax-highlighting
 *
 * The pair of a language and a value is an equivalent to markdown:
 * ```${language}
 * ${value}
 * ```
 *
 * Note that markdown strings will be sanitized - that means html will be escaped.
 * @deprecated use MarkupContent instead.
 */
type MarkedString = string

// DefinitionLink is a type
/**
 * Information about where a symbol is defined.
 *
 * Provides additional metadata over normal [location](#Location) definitions, including the range of
 * the defining symbol
 */
type DefinitionLink = LocationLink

// DeclarationLink is a type
/**
 * Information about where a symbol is declared.
 *
 * Provides additional metadata over normal [location](#Location) declarations, including the range of
 * the declaring symbol.
 *
 * Servers should prefer returning `DeclarationLink` over `Declaration` if supported
 * by the client.
 */
type DeclarationLink = LocationLink

// LSPMessageType is a type
/**
 * A LSP Log Entry.
 */
type LSPMessageType = string

// ProgressToken is a type
type ProgressToken = interface{} // number | string
// TraceValues is a type
type TraceValues = string
