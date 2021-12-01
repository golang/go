// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package protocol contains data types and code for LSP jsonrpcs
// generated automatically from vscode-languageserver-node
// commit: d959faf4be476a6e0a08d5612e91fcac14ff9929
// last fetched Wed Dec 01 2021 09:27:34 GMT-0500 (Eastern Standard Time)
package protocol

// Code generated (see typescript/README.md) DO NOT EDIT.

import "encoding/json"

/**
 * A special text edit with an additional change annotation.
 *
 * @since 3.16.0.
 */
type AnnotatedTextEdit struct {
	/**
	 * The actual identifier of the change annotation
	 */
	AnnotationID ChangeAnnotationIdentifier `json:"annotationId"`
	TextEdit
}

/**
 * The parameters passed via a apply workspace edit request.
 */
type ApplyWorkspaceEditParams struct {
	/**
	 * An optional label of the workspace edit. This label is
	 * presented in the user interface for example on an undo
	 * stack to undo the workspace edit.
	 */
	Label string `json:"label,omitempty"`
	/**
	 * The edits to apply.
	 */
	Edit WorkspaceEdit `json:"edit"`
}

/**
 * The result returned from the apply workspace edit request.
 *
 * @since 3.17 renamed from ApplyWorkspaceEditResponse
 */
type ApplyWorkspaceEditResult struct {
	/**
	 * Indicates whether the edit was applied or not.
	 */
	Applied bool `json:"applied"`
	/**
	 * An optional textual description for why the edit was not applied.
	 * This may be used by the server for diagnostic logging or to provide
	 * a suitable error for a request that triggered the edit.
	 */
	FailureReason string `json:"failureReason,omitempty"`
	/**
	 * Depending on the client's failure handling strategy `failedChange` might
	 * contain the index of the change that failed. This property is only available
	 * if the client signals a `failureHandlingStrategy` in its client capabilities.
	 */
	FailedChange uint32 `json:"failedChange,omitempty"`
}

/**
 * @since 3.16.0
 */
type CallHierarchyClientCapabilities struct {
	/**
	 * Whether implementation supports dynamic registration. If this is set to `true`
	 * the client supports the new `(TextDocumentRegistrationOptions & StaticRegistrationOptions)`
	 * return value for the corresponding server capability as well.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
}

/**
 * Represents an incoming call, e.g. a caller of a method or constructor.
 *
 * @since 3.16.0
 */
type CallHierarchyIncomingCall struct {
	/**
	 * The item that makes the call.
	 */
	From CallHierarchyItem `json:"from"`
	/**
	 * The ranges at which the calls appear. This is relative to the caller
	 * denoted by [`this.from`](#CallHierarchyIncomingCall.from).
	 */
	FromRanges []Range `json:"fromRanges"`
}

/**
 * The parameter of a `callHierarchy/incomingCalls` request.
 *
 * @since 3.16.0
 */
type CallHierarchyIncomingCallsParams struct {
	Item CallHierarchyItem `json:"item"`
	WorkDoneProgressParams
	PartialResultParams
}

/**
 * Represents programming constructs like functions or constructors in the context
 * of call hierarchy.
 *
 * @since 3.16.0
 */
type CallHierarchyItem struct {
	/**
	 * The name of this item.
	 */
	Name string `json:"name"`
	/**
	 * The kind of this item.
	 */
	Kind SymbolKind `json:"kind"`
	/**
	 * Tags for this item.
	 */
	Tags []SymbolTag `json:"tags,omitempty"`
	/**
	 * More detail for this item, e.g. the signature of a function.
	 */
	Detail string `json:"detail,omitempty"`
	/**
	 * The resource identifier of this item.
	 */
	URI DocumentURI `json:"uri"`
	/**
	 * The range enclosing this symbol not including leading/trailing whitespace but everything else, e.g. comments and code.
	 */
	Range Range `json:"range"`
	/**
	 * The range that should be selected and revealed when this symbol is being picked, e.g. the name of a function.
	 * Must be contained by the [`range`](#CallHierarchyItem.range).
	 */
	SelectionRange Range `json:"selectionRange"`
	/**
	 * A data entry field that is preserved between a call hierarchy prepare and
	 * incoming calls or outgoing calls requests.
	 */
	Data LSPAny `json:"data,omitempty"`
}

/**
 * Call hierarchy options used during static registration.
 *
 * @since 3.16.0
 */
type CallHierarchyOptions struct {
	WorkDoneProgressOptions
}

/**
 * Represents an outgoing call, e.g. calling a getter from a method or a method from a constructor etc.
 *
 * @since 3.16.0
 */
type CallHierarchyOutgoingCall struct {
	/**
	 * The item that is called.
	 */
	To CallHierarchyItem `json:"to"`
	/**
	 * The range at which this item is called. This is the range relative to the caller, e.g the item
	 * passed to [`provideCallHierarchyOutgoingCalls`](#CallHierarchyItemProvider.provideCallHierarchyOutgoingCalls)
	 * and not [`this.to`](#CallHierarchyOutgoingCall.to).
	 */
	FromRanges []Range `json:"fromRanges"`
}

/**
 * The parameter of a `callHierarchy/outgoingCalls` request.
 *
 * @since 3.16.0
 */
type CallHierarchyOutgoingCallsParams struct {
	Item CallHierarchyItem `json:"item"`
	WorkDoneProgressParams
	PartialResultParams
}

/**
 * The parameter of a `textDocument/prepareCallHierarchy` request.
 *
 * @since 3.16.0
 */
type CallHierarchyPrepareParams struct {
	TextDocumentPositionParams
	WorkDoneProgressParams
}

/**
 * Call hierarchy options used during static or dynamic registration.
 *
 * @since 3.16.0
 */
type CallHierarchyRegistrationOptions struct {
	TextDocumentRegistrationOptions
	CallHierarchyOptions
	StaticRegistrationOptions
}

type CancelParams struct {
	/**
	 * The request id to cancel.
	 */
	ID interface{} /*number | string*/ `json:"id"`
}

/**
 * Additional information that describes document changes.
 *
 * @since 3.16.0
 */
type ChangeAnnotation struct {
	/**
	       * A human-readable string describing the actual change. The string
	  	 * is rendered prominent in the user interface.
	*/
	Label string `json:"label"`
	/**
	       * A flag which indicates that user confirmation is needed
	  	 * before applying the change.
	*/
	NeedsConfirmation bool `json:"needsConfirmation,omitempty"`
	/**
	 * A human-readable string which is rendered less prominent in
	 * the user interface.
	 */
	Description string `json:"description,omitempty"`
}

/**
 * An identifier to refer to a change annotation stored with a workspace edit.
 */
type ChangeAnnotationIdentifier = string

type ClientCapabilities struct {
	/**
	 * The workspace client capabilities
	 */
	Workspace Workspace2Gn `json:"workspace,omitempty"`
	/**
	 * Text document specific client capabilities.
	 */
	TextDocument TextDocumentClientCapabilities `json:"textDocument,omitempty"`
	/**
	 * Window specific client capabilities.
	 */
	Window struct {
		/**
		 * Whether client supports server initiated progress using the
		 * `window/workDoneProgress/create` request.
		 *
		 * Since 3.15.0
		 */
		WorkDoneProgress bool `json:"workDoneProgress,omitempty"`
		/**
		 * Capabilities specific to the showMessage request.
		 *
		 * @since 3.16.0
		 */
		ShowMessage ShowMessageRequestClientCapabilities `json:"showMessage,omitempty"`
		/**
		 * Capabilities specific to the showDocument request.
		 *
		 * @since 3.16.0
		 */
		ShowDocument ShowDocumentClientCapabilities `json:"showDocument,omitempty"`
	} `json:"window,omitempty"`
	/**
	 * General client capabilities.
	 *
	 * @since 3.16.0
	 */
	General GeneralClientCapabilities `json:"general,omitempty"`
	/**
	 * Experimental client capabilities.
	 */
	Experimental interface{} `json:"experimental,omitempty"`
}

/**
 * A code action represents a change that can be performed in code, e.g. to fix a problem or
 * to refactor code.
 *
 * A CodeAction must set either `edit` and/or a `command`. If both are supplied, the `edit` is applied first, then the `command` is executed.
 */
type CodeAction struct {
	/**
	 * A short, human-readable, title for this code action.
	 */
	Title string `json:"title"`
	/**
	 * The kind of the code action.
	 *
	 * Used to filter code actions.
	 */
	Kind CodeActionKind `json:"kind,omitempty"`
	/**
	 * The diagnostics that this code action resolves.
	 */
	Diagnostics []Diagnostic `json:"diagnostics,omitempty"`
	/**
	 * Marks this as a preferred action. Preferred actions are used by the `auto fix` command and can be targeted
	 * by keybindings.
	 *
	 * A quick fix should be marked preferred if it properly addresses the underlying error.
	 * A refactoring should be marked preferred if it is the most reasonable choice of actions to take.
	 *
	 * @since 3.15.0
	 */
	IsPreferred bool `json:"isPreferred,omitempty"`
	/**
	 * Marks that the code action cannot currently be applied.
	 *
	 * Clients should follow the following guidelines regarding disabled code actions:
	 *
	 *   - Disabled code actions are not shown in automatic [lightbulb](https://code.visualstudio.com/docs/editor/editingevolved#_code-action)
	 *     code action menu.
	 *
	 *   - Disabled actions are shown as faded out in the code action menu when the user request a more specific type
	 *     of code action, such as refactorings.
	 *
	 *   - If the user has a [keybinding](https://code.visualstudio.com/docs/editor/refactoring#_keybindings-for-code-actions)
	 *     that auto applies a code action and only a disabled code actions are returned, the client should show the user an
	 *     error message with `reason` in the editor.
	 *
	 * @since 3.16.0
	 */
	Disabled *struct {
		/**
		 * Human readable description of why the code action is currently disabled.
		 *
		 * This is displayed in the code actions UI.
		 */
		Reason string `json:"reason"`
	} `json:"disabled,omitempty"`
	/**
	 * The workspace edit this code action performs.
	 */
	Edit WorkspaceEdit `json:"edit,omitempty"`
	/**
	 * A command this code action executes. If a code action
	 * provides a edit and a command, first the edit is
	 * executed and then the command.
	 */
	Command *Command `json:"command,omitempty"`
	/**
	 * A data entry field that is preserved on a code action between
	 * a `textDocument/codeAction` and a `codeAction/resolve` request.
	 *
	 * @since 3.16.0
	 */
	Data LSPAny `json:"data,omitempty"`
}

/**
 * The Client Capabilities of a [CodeActionRequest](#CodeActionRequest).
 */
type CodeActionClientCapabilities struct {
	/**
	 * Whether code action supports dynamic registration.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
	/**
	 * The client support code action literals of type `CodeAction` as a valid
	 * response of the `textDocument/codeAction` request. If the property is not
	 * set the request can only return `Command` literals.
	 *
	 * @since 3.8.0
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
	/**
	 * Whether code action supports the `isPreferred` property.
	 *
	 * @since 3.15.0
	 */
	IsPreferredSupport bool `json:"isPreferredSupport,omitempty"`
	/**
	 * Whether code action supports the `disabled` property.
	 *
	 * @since 3.16.0
	 */
	DisabledSupport bool `json:"disabledSupport,omitempty"`
	/**
	 * Whether code action supports the `data` property which is
	 * preserved between a `textDocument/codeAction` and a
	 * `codeAction/resolve` request.
	 *
	 * @since 3.16.0
	 */
	DataSupport bool `json:"dataSupport,omitempty"`
	/**
	 * Whether the client support resolving additional code action
	 * properties via a separate `codeAction/resolve` request.
	 *
	 * @since 3.16.0
	 */
	ResolveSupport struct {
		/**
		 * The properties that a client can resolve lazily.
		 */
		Properties []string `json:"properties"`
	} `json:"resolveSupport,omitempty"`
	/**
	 * Whether th client honors the change annotations in
	 * text edits and resource operations returned via the
	 * `CodeAction#edit` property by for example presenting
	 * the workspace edit in the user interface and asking
	 * for confirmation.
	 *
	 * @since 3.16.0
	 */
	HonorsChangeAnnotations bool `json:"honorsChangeAnnotations,omitempty"`
}

/**
 * Contains additional diagnostic information about the context in which
 * a [code action](#CodeActionProvider.provideCodeActions) is run.
 */
type CodeActionContext struct {
	/**
	 * An array of diagnostics known on the client side overlapping the range provided to the
	 * `textDocument/codeAction` request. They are provided so that the server knows which
	 * errors are currently presented to the user for the given range. There is no guarantee
	 * that these accurately reflect the error state of the resource. The primary parameter
	 * to compute code actions is the provided range.
	 */
	Diagnostics []Diagnostic `json:"diagnostics"`
	/**
	 * Requested kind of actions to return.
	 *
	 * Actions not of this kind are filtered out by the client before being shown. So servers
	 * can omit computing them.
	 */
	Only []CodeActionKind `json:"only,omitempty"`
	/**
	 * The reason why code actions were requested.
	 *
	 * @since 3.17.0
	 */
	TriggerKind CodeActionTriggerKind `json:"triggerKind,omitempty"`
}

/**
 * A set of predefined code action kinds
 */
type CodeActionKind string

/**
 * Provider options for a [CodeActionRequest](#CodeActionRequest).
 */
type CodeActionOptions struct {
	/**
	 * CodeActionKinds that this server may return.
	 *
	 * The list of kinds may be generic, such as `CodeActionKind.Refactor`, or the server
	 * may list out every specific kind they provide.
	 */
	CodeActionKinds []CodeActionKind `json:"codeActionKinds,omitempty"`
	/**
	 * The server provides support to resolve additional
	 * information for a code action.
	 *
	 * @since 3.16.0
	 */
	ResolveProvider bool `json:"resolveProvider,omitempty"`
	WorkDoneProgressOptions
}

/**
 * The parameters of a [CodeActionRequest](#CodeActionRequest).
 */
type CodeActionParams struct {
	/**
	 * The document in which the command was invoked.
	 */
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	/**
	 * The range for which the command was invoked.
	 */
	Range Range `json:"range"`
	/**
	 * Context carrying additional information.
	 */
	Context CodeActionContext `json:"context"`
	WorkDoneProgressParams
	PartialResultParams
}

/**
 * The reason why code actions were requested.
 *
 * @since 3.17.0 - proposed state
 */
type CodeActionTriggerKind float64

/**
 * Structure to capture a description for an error code.
 *
 * @since 3.16.0
 */
type CodeDescription struct {
	/**
	 * An URI to open with more information about the diagnostic error.
	 */
	Href URI `json:"href"`
}

/**
 * A code lens represents a [command](#Command) that should be shown along with
 * source text, like the number of references, a way to run tests, etc.
 *
 * A code lens is _unresolved_ when no command is associated to it. For performance
 * reasons the creation of a code lens and resolving should be done to two stages.
 */
type CodeLens struct {
	/**
	 * The range in which this code lens is valid. Should only span a single line.
	 */
	Range Range `json:"range"`
	/**
	 * The command this code lens represents.
	 */
	Command Command `json:"command,omitempty"`
	/**
	 * A data entry field that is preserved on a code lens item between
	 * a [CodeLensRequest](#CodeLensRequest) and a [CodeLensResolveRequest]
	 * (#CodeLensResolveRequest)
	 */
	Data LSPAny `json:"data,omitempty"`
}

/**
 * The client capabilities  of a [CodeLensRequest](#CodeLensRequest).
 */
type CodeLensClientCapabilities struct {
	/**
	 * Whether code lens supports dynamic registration.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
}

/**
 * Code Lens provider options of a [CodeLensRequest](#CodeLensRequest).
 */
type CodeLensOptions struct {
	/**
	 * Code lens has a resolve provider as well.
	 */
	ResolveProvider bool `json:"resolveProvider,omitempty"`
	WorkDoneProgressOptions
}

/**
 * The parameters of a [CodeLensRequest](#CodeLensRequest).
 */
type CodeLensParams struct {
	/**
	 * The document to request code lens for.
	 */
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	WorkDoneProgressParams
	PartialResultParams
}

/**
 * @since 3.16.0
 */
type CodeLensWorkspaceClientCapabilities struct {
	/**
	 * Whether the client implementation supports a refresh request sent from the
	 * server to the client.
	 *
	 * Note that this event is global and will force the client to refresh all
	 * code lenses currently shown. It should be used with absolute care and is
	 * useful for situation where a server for example detect a project wide
	 * change that requires such a calculation.
	 */
	RefreshSupport bool `json:"refreshSupport,omitempty"`
}

/**
 * Represents a color in RGBA space.
 */
type Color struct {
	/**
	 * The red component of this color in the range [0-1].
	 */
	Red Decimal `json:"red"`
	/**
	 * The green component of this color in the range [0-1].
	 */
	Green Decimal `json:"green"`
	/**
	 * The blue component of this color in the range [0-1].
	 */
	Blue Decimal `json:"blue"`
	/**
	 * The alpha component of this color in the range [0-1].
	 */
	Alpha Decimal `json:"alpha"`
}

/**
 * Represents a color range from a document.
 */
type ColorInformation struct {
	/**
	 * The range in the document where this color appears.
	 */
	Range Range `json:"range"`
	/**
	 * The actual color value for this color range.
	 */
	Color Color `json:"color"`
}

type ColorPresentation struct {
	/**
	 * The label of this color presentation. It will be shown on the color
	 * picker header. By default this is also the text that is inserted when selecting
	 * this color presentation.
	 */
	Label string `json:"label"`
	/**
	 * An [edit](#TextEdit) which is applied to a document when selecting
	 * this presentation for the color.  When `falsy` the [label](#ColorPresentation.label)
	 * is used.
	 */
	TextEdit TextEdit `json:"textEdit,omitempty"`
	/**
	 * An optional array of additional [text edits](#TextEdit) that are applied when
	 * selecting this color presentation. Edits must not overlap with the main [edit](#ColorPresentation.textEdit) nor with themselves.
	 */
	AdditionalTextEdits []TextEdit `json:"additionalTextEdits,omitempty"`
}

/**
 * Parameters for a [ColorPresentationRequest](#ColorPresentationRequest).
 */
type ColorPresentationParams struct {
	/**
	 * The text document.
	 */
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	/**
	 * The color to request presentations for.
	 */
	Color Color `json:"color"`
	/**
	 * The range where the color would be inserted. Serves as a context.
	 */
	Range Range `json:"range"`
	WorkDoneProgressParams
	PartialResultParams
}

/**
 * Represents a reference to a command. Provides a title which
 * will be used to represent a command in the UI and, optionally,
 * an array of arguments which will be passed to the command handler
 * function when invoked.
 */
type Command struct {
	/**
	 * Title of the command, like `save`.
	 */
	Title string `json:"title"`
	/**
	 * The identifier of the actual command handler.
	 */
	Command string `json:"command"`
	/**
	 * Arguments that the command handler should be
	 * invoked with.
	 */
	Arguments []json.RawMessage `json:"arguments,omitempty"`
}

/**
 * Completion client capabilities
 */
type CompletionClientCapabilities struct {
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
		/**
		 * Client supports to kee
		 */

		/**
		 * Client supports the tag property on a completion item. Clients supporting
		 * tags have to handle unknown tags gracefully. Clients especially need to
		 * preserve unknown tags when sending a completion item back to the server in
		 * a resolve call.
		 *
		 * @since 3.15.0
		 */
		TagSupport struct {
			/**
			 * The tags supported by the client.
			 */
			ValueSet []CompletionItemTag `json:"valueSet"`
		} `json:"tagSupport,omitempty"`
		/**
		 * Client support insert replace edit to control different behavior if a
		 * completion item is inserted in the text or should replace text.
		 *
		 * @since 3.16.0
		 */
		InsertReplaceSupport bool `json:"insertReplaceSupport,omitempty"`
		/**
		 * Indicates which properties a client can resolve lazily on a completion
		 * item. Before version 3.16.0 only the predefined properties `documentation`
		 * and `details` could be resolved lazily.
		 *
		 * @since 3.16.0
		 */
		ResolveSupport struct {
			/**
			 * The properties that a client can resolve lazily.
			 */
			Properties []string `json:"properties"`
		} `json:"resolveSupport,omitempty"`
		/**
		 * The client supports the `insertTextMode` property on
		 * a completion item to override the whitespace handling mode
		 * as defined by the client (see `insertTextMode`).
		 *
		 * @since 3.16.0
		 */
		InsertTextModeSupport struct {
			ValueSet []InsertTextMode `json:"valueSet"`
		} `json:"insertTextModeSupport,omitempty"`
		/**
		 * The client has support for completion item label
		 * details (see also `CompletionItemLabelDetails`).
		 *
		 * @since 3.17.0 - proposed state
		 */
		LabelDetailsSupport bool `json:"labelDetailsSupport,omitempty"`
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
	 * Defines how the client handles whitespace and indentation
	 * when accepting a completion item that uses multi line
	 * text in either `insertText` or `textEdit`.
	 *
	 * @since 3.17.0 - proposed state
	 */
	InsertTextMode InsertTextMode `json:"insertTextMode,omitempty"`
	/**
	 * The client supports to send additional context information for a
	 * `textDocument/completion` request.
	 */
	ContextSupport bool `json:"contextSupport,omitempty"`
	/**
	 * The client supports the following `CompletionList` specific
	 * capabilities.
	 *
	 * @since 3.17.0 - proposed state
	 */
	CompletionList struct {
		/**
		 * The client supports the the following itemDefaults on
		 * a completion list.
		 *
		 * The value lists the supported property names of the
		 * `CompletionList.itemDefaults` object. If omitted
		 * no properties are supported.
		 *
		 * @since 3.17.0 - proposed state
		 */
		ItemDefaults []string `json:"itemDefaults,omitempty"`
	} `json:"completionList,omitempty"`
}

/**
 * Contains additional information about the context in which a completion request is triggered.
 */
type CompletionContext struct {
	/**
	 * How the completion was triggered.
	 */
	TriggerKind CompletionTriggerKind `json:"triggerKind"`
	/**
	 * The trigger character (a single character) that has trigger code complete.
	 * Is undefined if `triggerKind !== CompletionTriggerKind.TriggerCharacter`
	 */
	TriggerCharacter string `json:"triggerCharacter,omitempty"`
}

/**
 * A completion item represents a text snippet that is
 * proposed to complete text that is being typed.
 */
type CompletionItem struct {
	/**
	 * The label of this completion item.
	 *
	 * The label property is also by default the text that
	 * is inserted when selecting this completion.
	 *
	 * If label details are provided the label itself should
	 * be an unqualified name of the completion item.
	 */
	Label string `json:"label"`
	/**
	 * Additional details for the label
	 *
	 * @since 3.17.0 - proposed state
	 */
	LabelDetails CompletionItemLabelDetails `json:"labelDetails,omitempty"`
	/**
	 * The kind of this completion item. Based of the kind
	 * an icon is chosen by the editor.
	 */
	Kind CompletionItemKind `json:"kind,omitempty"`
	/**
	 * Tags for this completion item.
	 *
	 * @since 3.15.0
	 */
	Tags []CompletionItemTag `json:"tags,omitempty"`
	/**
	 * A human-readable string with additional information
	 * about this item, like type or symbol information.
	 */
	Detail string `json:"detail,omitempty"`
	/**
	 * A human-readable string that represents a doc-comment.
	 */
	Documentation string/*string | MarkupContent*/ `json:"documentation,omitempty"`
	/**
	 * Indicates if this item is deprecated.
	 * @deprecated Use `tags` instead.
	 */
	Deprecated bool `json:"deprecated,omitempty"`
	/**
	 * Select this item when showing.
	 *
	 * *Note* that only one completion item can be selected and that the
	 * tool / client decides which item that is. The rule is that the *first*
	 * item of those that match best is selected.
	 */
	Preselect bool `json:"preselect,omitempty"`
	/**
	 * A string that should be used when comparing this item
	 * with other items. When `falsy` the [label](#CompletionItem.label)
	 * is used.
	 */
	SortText string `json:"sortText,omitempty"`
	/**
	 * A string that should be used when filtering a set of
	 * completion items. When `falsy` the [label](#CompletionItem.label)
	 * is used.
	 */
	FilterText string `json:"filterText,omitempty"`
	/**
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
	/**
	 * The format of the insert text. The format applies to both the `insertText` property
	 * and the `newText` property of a provided `textEdit`. If omitted defaults to
	 * `InsertTextFormat.PlainText`.
	 *
	 * Please note that the insertTextFormat doesn't apply to `additionalTextEdits`.
	 */
	InsertTextFormat InsertTextFormat `json:"insertTextFormat,omitempty"`
	/**
	 * How whitespace and indentation is handled during completion
	 * item insertion. If ignored the clients default value depends on
	 * the `textDocument.completion.insertTextMode` client capability.
	 *
	 * @since 3.16.0
	 */
	InsertTextMode InsertTextMode `json:"insertTextMode,omitempty"`
	/**
	 * An [edit](#TextEdit) which is applied to a document when selecting
	 * this completion. When an edit is provided the value of
	 * [insertText](#CompletionItem.insertText) is ignored.
	 *
	 * Most editors support two different operation when accepting a completion item. One is to insert a
	 * completion text and the other is to replace an existing text with a completion text. Since this can
	 * usually not predetermined by a server it can report both ranges. Clients need to signal support for
	 * `InsertReplaceEdits` via the `textDocument.completion.insertReplaceSupport` client capability
	 * property.
	 *
	 * *Note 1:* The text edit's range as well as both ranges from a insert replace edit must be a
	 * [single line] and they must contain the position at which completion has been requested.
	 * *Note 2:* If an `InsertReplaceEdit` is returned the edit's insert range must be a prefix of
	 * the edit's replace range, that means it must be contained and starting at the same position.
	 *
	 * @since 3.16.0 additional type `InsertReplaceEdit`
	 */
	TextEdit *TextEdit/*TextEdit | InsertReplaceEdit*/ `json:"textEdit,omitempty"`
	/**
	 * An optional array of additional [text edits](#TextEdit) that are applied when
	 * selecting this completion. Edits must not overlap (including the same insert position)
	 * with the main [edit](#CompletionItem.textEdit) nor with themselves.
	 *
	 * Additional text edits should be used to change text unrelated to the current cursor position
	 * (for example adding an import statement at the top of the file if the completion item will
	 * insert an unqualified type).
	 */
	AdditionalTextEdits []TextEdit `json:"additionalTextEdits,omitempty"`
	/**
	 * An optional set of characters that when pressed while this completion is active will accept it first and
	 * then type that character. *Note* that all commit characters should have `length=1` and that superfluous
	 * characters will be ignored.
	 */
	CommitCharacters []string `json:"commitCharacters,omitempty"`
	/**
	 * An optional [command](#Command) that is executed *after* inserting this completion. *Note* that
	 * additional modifications to the current document should be described with the
	 * [additionalTextEdits](#CompletionItem.additionalTextEdits)-property.
	 */
	Command *Command `json:"command,omitempty"`
	/**
	 * A data entry field that is preserved on a completion item between a
	 * [CompletionRequest](#CompletionRequest) and a [CompletionResolveRequest](#CompletionResolveRequest).
	 */
	Data LSPAny `json:"data,omitempty"`
}

/**
 * The kind of a completion entry.
 */
type CompletionItemKind float64

/**
 * Additional details for a completion item label.
 *
 * @since 3.17.0 - proposed state
 */
type CompletionItemLabelDetails struct {
	/**
	 * An optional string which is rendered less prominently directly after {@link CompletionItem.label label},
	 * without any spacing. Should be used for function signatures or type annotations.
	 */
	Detail string `json:"detail,omitempty"`
	/**
	 * An optional string which is rendered less prominently after {@link CompletionItem.detail}. Should be used
	 * for fully qualified names or file path.
	 */
	Description string `json:"description,omitempty"`
}

/**
 * Completion item tags are extra annotations that tweak the rendering of a completion
 * item.
 *
 * @since 3.15.0
 */
type CompletionItemTag float64

/**
 * Represents a collection of [completion items](#CompletionItem) to be presented
 * in the editor.
 */
type CompletionList struct {
	/**
	 * This list it not complete. Further typing results in recomputing this list.
	 */
	IsIncomplete bool `json:"isIncomplete"`
	/**
	 * In many cases the items of an actual completion result share the same
	 * value for properties like `commitCharacters` or the range of a text
	 * edit. A completion list can therefore define item defaults which will
	 * be used if a completion item itself doesn't specify the value.
	 *
	 * If a completion list specifies a default value and a completion item
	 * also specifies a corresponding value the one from the item is used.
	 *
	 * Servers are only allowed to return default values if the client
	 * signals support for this via the `completionList.itemDefaults`
	 * capability.
	 *
	 * @since 3.17.0 - proposed state
	 */
	ItemDefaults struct {
		/**
		 * A default commit character set.
		 *
		 * @since 3.17.0 - proposed state
		 */
		CommitCharacters []string `json:"commitCharacters,omitempty"`
		/**
		 * A default edit range
		 *
		 * @since 3.17.0 - proposed state
		 */
		EditRange Range/*Range | { insert: Range; replace: Range; }*/ `json:"editRange,omitempty"`
		/**
		 * A default insert text format
		 *
		 * @since 3.17.0 - proposed state
		 */
		InsertTextFormat InsertTextFormat `json:"insertTextFormat,omitempty"`
		/**
		 * A default insert text mode
		 *
		 * @since 3.17.0 - proposed state
		 */
		InsertTextMode InsertTextMode `json:"insertTextMode,omitempty"`
	} `json:"itemDefaults,omitempty"`
	/**
	 * The completion items.
	 */
	Items []CompletionItem `json:"items"`
}

/**
 * Completion options.
 */
type CompletionOptions struct {
	/**
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
	/**
	 * The list of all possible characters that commit a completion. This field can be used
	 * if clients don't support individual commit characters per completion item. See
	 * `ClientCapabilities.textDocument.completion.completionItem.commitCharactersSupport`
	 *
	 * If a server provides both `allCommitCharacters` and commit characters on an individual
	 * completion item the ones on the completion item win.
	 *
	 * @since 3.2.0
	 */
	AllCommitCharacters []string `json:"allCommitCharacters,omitempty"`
	/**
	 * The server provides support to resolve additional
	 * information for a completion item.
	 */
	ResolveProvider bool `json:"resolveProvider,omitempty"`
	/**
	 * The server supports the following `CompletionItem` specific
	 * capabilities.
	 *
	 * @since 3.17.0 - proposed state
	 */
	CompletionItem struct {
		/**
		 * The server has support for completion item label
		 * details (see also `CompletionItemLabelDetails`) when
		 * receiving a completion item in a resolve call.
		 *
		 * @since 3.17.0 - proposed state
		 */
		LabelDetailsSupport bool `json:"labelDetailsSupport,omitempty"`
	} `json:"completionItem,omitempty"`
	WorkDoneProgressOptions
}

/**
 * Completion parameters
 */
type CompletionParams struct {
	/**
	 * The completion context. This is only available it the client specifies
	 * to send this using the client capability `textDocument.completion.contextSupport === true`
	 */
	Context CompletionContext `json:"context,omitempty"`
	TextDocumentPositionParams
	WorkDoneProgressParams
	PartialResultParams
}

/**
 * How a completion was triggered
 */
type CompletionTriggerKind float64

type ConfigurationClientCapabilities struct {
	/**
	 * The workspace client capabilities
	 */
	Workspace Workspace3Gn `json:"workspace,omitempty"`
}

type ConfigurationItem struct {
	/**
	 * The scope to get the configuration section for.
	 */
	ScopeURI string `json:"scopeUri,omitempty"`
	/**
	 * The configuration section asked for.
	 */
	Section string `json:"section,omitempty"`
}

/**
 * The parameters of a configuration request.
 */
type ConfigurationParams struct {
	Items []ConfigurationItem `json:"items"`
}

/**
 * Create file operation.
 */
type CreateFile struct {
	/**
	 * A create
	 */
	Kind string `json:"kind"`
	/**
	 * The resource to create.
	 */
	URI DocumentURI `json:"uri"`
	/**
	 * Additional options
	 */
	Options CreateFileOptions `json:"options,omitempty"`
	ResourceOperation
}

/**
 * Options to create a file.
 */
type CreateFileOptions struct {
	/**
	 * Overwrite existing file. Overwrite wins over `ignoreIfExists`
	 */
	Overwrite bool `json:"overwrite,omitempty"`
	/**
	 * Ignore if exists.
	 */
	IgnoreIfExists bool `json:"ignoreIfExists,omitempty"`
}

/**
 * The parameters sent in file create requests/notifications.
 *
 * @since 3.16.0
 */
type CreateFilesParams struct {
	/**
	 * An array of all files/folders created in this operation.
	 */
	Files []FileCreate `json:"files"`
}

/**
 * Defines a decimal number. Since decimal numbers are very
 * rare in the language server specification we denote the
 * exact range with every decimal using the mathematics
 * interval notations (e.g. [0, 1] denotes all decimals d with
 * 0 <= d <= 1.
 */
type Decimal = float64

/**
 * The declaration of a symbol representation as one or many [locations](#Location).
 */
type Declaration = []Location /*Location | Location[]*/

/**
 * @since 3.14.0
 */
type DeclarationClientCapabilities struct {
	/**
	 * Whether declaration supports dynamic registration. If this is set to `true`
	 * the client supports the new `DeclarationRegistrationOptions` return value
	 * for the corresponding server capability as well.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
	/**
	 * The client supports additional metadata in the form of declaration links.
	 */
	LinkSupport bool `json:"linkSupport,omitempty"`
}

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

type DeclarationOptions struct {
	WorkDoneProgressOptions
}

type DeclarationParams struct {
	TextDocumentPositionParams
	WorkDoneProgressParams
	PartialResultParams
}

type DeclarationRegistrationOptions struct {
	DeclarationOptions
	TextDocumentRegistrationOptions
	StaticRegistrationOptions
}

/**
 * The definition of a symbol represented as one or many [locations](#Location).
 * For most programming languages there is only one location at which a symbol is
 * defined.
 *
 * Servers should prefer returning `DefinitionLink` over `Definition` if supported
 * by the client.
 */
type Definition = []Location /*Location | Location[]*/

/**
 * Client Capabilities for a [DefinitionRequest](#DefinitionRequest).
 */
type DefinitionClientCapabilities struct {
	/**
	 * Whether definition supports dynamic registration.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
	/**
	 * The client supports additional metadata in the form of definition links.
	 *
	 * @since 3.14.0
	 */
	LinkSupport bool `json:"linkSupport,omitempty"`
}

/**
 * Information about where a symbol is defined.
 *
 * Provides additional metadata over normal [location](#Location) definitions, including the range of
 * the defining symbol
 */
type DefinitionLink = LocationLink

/**
 * Server Capabilities for a [DefinitionRequest](#DefinitionRequest).
 */
type DefinitionOptions struct {
	WorkDoneProgressOptions
}

/**
 * Parameters for a [DefinitionRequest](#DefinitionRequest).
 */
type DefinitionParams struct {
	TextDocumentPositionParams
	WorkDoneProgressParams
	PartialResultParams
}

/**
 * Delete file operation
 */
type DeleteFile struct {
	/**
	 * A delete
	 */
	Kind string `json:"kind"`
	/**
	 * The file to delete.
	 */
	URI DocumentURI `json:"uri"`
	/**
	 * Delete options.
	 */
	Options DeleteFileOptions `json:"options,omitempty"`
	ResourceOperation
}

/**
 * Delete file options
 */
type DeleteFileOptions struct {
	/**
	 * Delete the content recursively if a folder is denoted.
	 */
	Recursive bool `json:"recursive,omitempty"`
	/**
	 * Ignore the operation if the file doesn't exist.
	 */
	IgnoreIfNotExists bool `json:"ignoreIfNotExists,omitempty"`
}

/**
 * The parameters sent in file delete requests/notifications.
 *
 * @since 3.16.0
 */
type DeleteFilesParams struct {
	/**
	 * An array of all files/folders deleted in this operation.
	 */
	Files []FileDelete `json:"files"`
}

/**
 * Represents a diagnostic, such as a compiler error or warning. Diagnostic objects
 * are only valid in the scope of a resource.
 */
type Diagnostic struct {
	/**
	 * The range at which the message applies
	 */
	Range Range `json:"range"`
	/**
	 * The diagnostic's severity. Can be omitted. If omitted it is up to the
	 * client to interpret diagnostics as error, warning, info or hint.
	 */
	Severity DiagnosticSeverity `json:"severity,omitempty"`
	/**
	 * The diagnostic's code, which usually appear in the user interface.
	 */
	Code interface{}/*integer | string*/ `json:"code,omitempty"`
	/**
	 * An optional property to describe the error code.
	 * Requires the code field (above) to be present/not null.
	 *
	 * @since 3.16.0
	 */
	CodeDescription *CodeDescription `json:"codeDescription,omitempty"`
	/**
	 * A human-readable string describing the source of this
	 * diagnostic, e.g. 'typescript' or 'super lint'. It usually
	 * appears in the user interface.
	 */
	Source string `json:"source,omitempty"`
	/**
	 * The diagnostic's message. It usually appears in the user interface
	 */
	Message string `json:"message"`
	/**
	 * Additional metadata about the diagnostic.
	 *
	 * @since 3.15.0
	 */
	Tags []DiagnosticTag `json:"tags,omitempty"`
	/**
	 * An array of related diagnostic information, e.g. when symbol-names within
	 * a scope collide all definitions can be marked via this property.
	 */
	RelatedInformation []DiagnosticRelatedInformation `json:"relatedInformation,omitempty"`
	/**
	 * A data entry field that is preserved between a `textDocument/publishDiagnostics`
	 * notification and `textDocument/codeAction` request.
	 *
	 * @since 3.16.0
	 */
	Data LSPAny `json:"data,omitempty"`
}

/**
 * Represents a related message and source code location for a diagnostic. This should be
 * used to point to code locations that cause or related to a diagnostics, e.g when duplicating
 * a symbol in a scope.
 */
type DiagnosticRelatedInformation struct {
	/**
	 * The location of this related diagnostic information.
	 */
	Location Location `json:"location"`
	/**
	 * The message of this related diagnostic information.
	 */
	Message string `json:"message"`
}

/**
 * The diagnostic's severity.
 */
type DiagnosticSeverity float64

/**
 * The diagnostic tags.
 *
 * @since 3.15.0
 */
type DiagnosticTag float64

type DidChangeConfigurationClientCapabilities struct {
	/**
	 * Did change configuration notification supports dynamic registration.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
}

/**
 * The parameters of a change configuration notification.
 */
type DidChangeConfigurationParams struct {
	/**
	 * The actual changed settings
	 */
	Settings LSPAny `json:"settings"`
}

/**
 * The change text document notification's parameters.
 */
type DidChangeTextDocumentParams struct {
	/**
	 * The document that did change. The version number points
	 * to the version after all provided content changes have
	 * been applied.
	 */
	TextDocument VersionedTextDocumentIdentifier `json:"textDocument"`
	/**
	 * The actual content changes. The content changes describe single state changes
	 * to the document. So if there are two content changes c1 (at array index 0) and
	 * c2 (at array index 1) for a document in state S then c1 moves the document from
	 * S to S' and c2 from S' to S''. So c1 is computed on the state S and c2 is computed
	 * on the state S'.
	 *
	 * To mirror the content of a document using change events use the following approach:
	 * - start with the same initial content
	 * - apply the 'textDocument/didChange' notifications in the order you receive them.
	 * - apply the `TextDocumentContentChangeEvent`s in a single notification in the order
	 *   you receive them.
	 */
	ContentChanges []TextDocumentContentChangeEvent `json:"contentChanges"`
}

type DidChangeWatchedFilesClientCapabilities struct {
	/**
	 * Did change watched files notification supports dynamic registration. Please note
	 * that the current protocol doesn't support static configuration for file changes
	 * from the server side.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
}

/**
 * The watched files change notification's parameters.
 */
type DidChangeWatchedFilesParams struct {
	/**
	 * The actual file events.
	 */
	Changes []FileEvent `json:"changes"`
}

/**
 * Describe options to be used when registered for text document change events.
 */
type DidChangeWatchedFilesRegistrationOptions struct {
	/**
	 * The watchers to register.
	 */
	Watchers []FileSystemWatcher `json:"watchers"`
}

/**
 * The parameters of a `workspace/didChangeWorkspaceFolders` notification.
 */
type DidChangeWorkspaceFoldersParams struct {
	/**
	 * The actual workspace folder change event.
	 */
	Event WorkspaceFoldersChangeEvent `json:"event"`
}

/**
 * The parameters send in a close text document notification
 */
type DidCloseTextDocumentParams struct {
	/**
	 * The document that was closed.
	 */
	TextDocument TextDocumentIdentifier `json:"textDocument"`
}

/**
 * The parameters send in a open text document notification
 */
type DidOpenTextDocumentParams struct {
	/**
	 * The document that was opened.
	 */
	TextDocument TextDocumentItem `json:"textDocument"`
}

/**
 * The parameters send in a save text document notification
 */
type DidSaveTextDocumentParams struct {
	/**
	 * The document that was closed.
	 */
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	/**
	 * Optional the content when saved. Depends on the includeText value
	 * when the save notification was requested.
	 */
	Text *string `json:"text,omitempty"`
}

type DocumentColorClientCapabilities struct {
	/**
	 * Whether implementation supports dynamic registration. If this is set to `true`
	 * the client supports the new `DocumentColorRegistrationOptions` return value
	 * for the corresponding server capability as well.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
}

type DocumentColorOptions struct {
	WorkDoneProgressOptions
}

/**
 * Parameters for a [DocumentColorRequest](#DocumentColorRequest).
 */
type DocumentColorParams struct {
	/**
	 * The text document.
	 */
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	WorkDoneProgressParams
	PartialResultParams
}

type DocumentColorRegistrationOptions struct {
	TextDocumentRegistrationOptions
	StaticRegistrationOptions
	DocumentColorOptions
}

/**
 * Parameters of the document diagnostic request.
 *
 * @since 3.17.0 - proposed state
 */
type DocumentDiagnosticParams struct {
	/**
	 * The text document.
	 */
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	/**
	 * The additional identifier  provided during registration.
	 */
	Identifier string `json:"identifier,omitempty"`
	/**
	 * The result id of a previous response if provided.
	 */
	PreviousResultID string `json:"previousResultId,omitempty"`
	WorkDoneProgressParams
	PartialResultParams
}

/**
 * The result of a document diagnostic pull request. A report can
 * either be a full report containing all diagnostics for the
 * requested document or a unchanged report indicating that nothing
 * has changed in terms of diagnostics in comparison to the last
 * pull request.
 *
 * @since 3.17.0 - proposed state
 */
type DocumentDiagnosticReport = interface{} /*RelatedFullDocumentDiagnosticReport | RelatedUnchangedDocumentDiagnosticReport*/

/**
 * A document filter denotes a document by different properties like
 * the [language](#TextDocument.languageId), the [scheme](#Uri.scheme) of
 * its resource, or a glob-pattern that is applied to the [path](#TextDocument.fileName).
 *
 * Glob patterns can have the following syntax:
 * - `*` to match one or more characters in a path segment
 * - `?` to match on one character in a path segment
 * - `**` to match any number of path segments, including none
 * - `{}` to group sub patterns into an OR expression. (e.g. `**​/*.{ts,js}` matches all TypeScript and JavaScript files)
 * - `[]` to declare a range of characters to match in a path segment (e.g., `example.[0-9]` to match on `example.0`, `example.1`, …)
 * - `[!...]` to negate a range of characters to match in a path segment (e.g., `example.[!0-9]` to match on `example.a`, `example.b`, but not `example.0`)
 *
 * @sample A language filter that applies to typescript files on disk: `{ language: 'typescript', scheme: 'file' }`
 * @sample A language filter that applies to all package.json paths: `{ language: 'json', pattern: '**package.json' }`
 */
type DocumentFilter = struct {
	/** A language id, like `typescript`. */
	Language string `json:"language"`
	/** A Uri [scheme](#Uri.scheme), like `file` or `untitled`. */
	Scheme string `json:"scheme,omitempty"`
	/** A glob pattern, like `*.{ts,js}`. */
	Pattern string `json:"pattern,omitempty"`
}

/**
 * Client capabilities of a [DocumentFormattingRequest](#DocumentFormattingRequest).
 */
type DocumentFormattingClientCapabilities struct {
	/**
	 * Whether formatting supports dynamic registration.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
}

/**
 * Provider options for a [DocumentFormattingRequest](#DocumentFormattingRequest).
 */
type DocumentFormattingOptions struct {
	WorkDoneProgressOptions
}

/**
 * The parameters of a [DocumentFormattingRequest](#DocumentFormattingRequest).
 */
type DocumentFormattingParams struct {
	/**
	 * The document to format.
	 */
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	/**
	 * The format options
	 */
	Options FormattingOptions `json:"options"`
	WorkDoneProgressParams
}

/**
 * A document highlight is a range inside a text document which deserves
 * special attention. Usually a document highlight is visualized by changing
 * the background color of its range.
 */
type DocumentHighlight struct {
	/**
	 * The range this highlight applies to.
	 */
	Range Range `json:"range"`
	/**
	 * The highlight kind, default is [text](#DocumentHighlightKind.Text).
	 */
	Kind DocumentHighlightKind `json:"kind,omitempty"`
}

/**
 * Client Capabilities for a [DocumentHighlightRequest](#DocumentHighlightRequest).
 */
type DocumentHighlightClientCapabilities struct {
	/**
	 * Whether document highlight supports dynamic registration.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
}

/**
 * A document highlight kind.
 */
type DocumentHighlightKind float64

/**
 * Provider options for a [DocumentHighlightRequest](#DocumentHighlightRequest).
 */
type DocumentHighlightOptions struct {
	WorkDoneProgressOptions
}

/**
 * Parameters for a [DocumentHighlightRequest](#DocumentHighlightRequest).
 */
type DocumentHighlightParams struct {
	TextDocumentPositionParams
	WorkDoneProgressParams
	PartialResultParams
}

/**
 * A document link is a range in a text document that links to an internal or external resource, like another
 * text document or a web site.
 */
type DocumentLink struct {
	/**
	 * The range this link applies to.
	 */
	Range Range `json:"range"`
	/**
	 * The uri this link points to.
	 */
	Target string `json:"target,omitempty"`
	/**
	 * The tooltip text when you hover over this link.
	 *
	 * If a tooltip is provided, is will be displayed in a string that includes instructions on how to
	 * trigger the link, such as `{0} (ctrl + click)`. The specific instructions vary depending on OS,
	 * user settings, and localization.
	 *
	 * @since 3.15.0
	 */
	Tooltip string `json:"tooltip,omitempty"`
	/**
	 * A data entry field that is preserved on a document link between a
	 * DocumentLinkRequest and a DocumentLinkResolveRequest.
	 */
	Data LSPAny `json:"data,omitempty"`
}

/**
 * The client capabilities of a [DocumentLinkRequest](#DocumentLinkRequest).
 */
type DocumentLinkClientCapabilities struct {
	/**
	 * Whether document link supports dynamic registration.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
	/**
	 * Whether the client support the `tooltip` property on `DocumentLink`.
	 *
	 * @since 3.15.0
	 */
	TooltipSupport bool `json:"tooltipSupport,omitempty"`
}

/**
 * Provider options for a [DocumentLinkRequest](#DocumentLinkRequest).
 */
type DocumentLinkOptions struct {
	/**
	 * Document links have a resolve provider as well.
	 */
	ResolveProvider bool `json:"resolveProvider,omitempty"`
	WorkDoneProgressOptions
}

/**
 * The parameters of a [DocumentLinkRequest](#DocumentLinkRequest).
 */
type DocumentLinkParams struct {
	/**
	 * The document to provide document links for.
	 */
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	WorkDoneProgressParams
	PartialResultParams
}

/**
 * Client capabilities of a [DocumentOnTypeFormattingRequest](#DocumentOnTypeFormattingRequest).
 */
type DocumentOnTypeFormattingClientCapabilities struct {
	/**
	 * Whether on type formatting supports dynamic registration.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
}

/**
 * Provider options for a [DocumentOnTypeFormattingRequest](#DocumentOnTypeFormattingRequest).
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
 * The parameters of a [DocumentOnTypeFormattingRequest](#DocumentOnTypeFormattingRequest).
 */
type DocumentOnTypeFormattingParams struct {
	/**
	 * The document to format.
	 */
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	/**
	 * The position at which this request was send.
	 */
	Position Position `json:"position"`
	/**
	 * The character that has been typed.
	 */
	Ch string `json:"ch"`
	/**
	 * The format options.
	 */
	Options FormattingOptions `json:"options"`
}

/**
 * Client capabilities of a [DocumentRangeFormattingRequest](#DocumentRangeFormattingRequest).
 */
type DocumentRangeFormattingClientCapabilities struct {
	/**
	 * Whether range formatting supports dynamic registration.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
}

/**
 * Provider options for a [DocumentRangeFormattingRequest](#DocumentRangeFormattingRequest).
 */
type DocumentRangeFormattingOptions struct {
	WorkDoneProgressOptions
}

/**
 * The parameters of a [DocumentRangeFormattingRequest](#DocumentRangeFormattingRequest).
 */
type DocumentRangeFormattingParams struct {
	/**
	 * The document to format.
	 */
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	/**
	 * The range to format
	 */
	Range Range `json:"range"`
	/**
	 * The format options
	 */
	Options FormattingOptions `json:"options"`
	WorkDoneProgressParams
}

/**
 * A document selector is the combination of one or many document filters.
 *
 * @sample `let sel:DocumentSelector = [{ language: 'typescript' }, { language: 'json', pattern: '**∕tsconfig.json' }]`;
 *
 * The use of a string as a document filter is deprecated @since 3.16.0.
 */
type DocumentSelector = []string /*string | DocumentFilter*/

/**
 * Represents programming constructs like variables, classes, interfaces etc.
 * that appear in a document. Document symbols can be hierarchical and they
 * have two ranges: one that encloses its definition and one that points to
 * its most interesting range, e.g. the range of an identifier.
 */
type DocumentSymbol struct {
	/**
	 * The name of this symbol. Will be displayed in the user interface and therefore must not be
	 * an empty string or a string only consisting of white spaces.
	 */
	Name string `json:"name"`
	/**
	 * More detail for this symbol, e.g the signature of a function.
	 */
	Detail string `json:"detail,omitempty"`
	/**
	 * The kind of this symbol.
	 */
	Kind SymbolKind `json:"kind"`
	/**
	 * Tags for this document symbol.
	 *
	 * @since 3.16.0
	 */
	Tags []SymbolTag `json:"tags,omitempty"`
	/**
	 * Indicates if this symbol is deprecated.
	 *
	 * @deprecated Use tags instead
	 */
	Deprecated bool `json:"deprecated,omitempty"`
	/**
	 * The range enclosing this symbol not including leading/trailing whitespace but everything else
	 * like comments. This information is typically used to determine if the the clients cursor is
	 * inside the symbol to reveal in the symbol in the UI.
	 */
	Range Range `json:"range"`
	/**
	 * The range that should be selected and revealed when this symbol is being picked, e.g the name of a function.
	 * Must be contained by the the `range`.
	 */
	SelectionRange Range `json:"selectionRange"`
	/**
	 * Children of this symbol, e.g. properties of a class.
	 */
	Children []DocumentSymbol `json:"children,omitempty"`
}

/**
 * Client Capabilities for a [DocumentSymbolRequest](#DocumentSymbolRequest).
 */
type DocumentSymbolClientCapabilities struct {
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
	/**
	 * The client supports tags on `SymbolInformation`. Tags are supported on
	 * `DocumentSymbol` if `hierarchicalDocumentSymbolSupport` is set to true.
	 * Clients supporting tags have to handle unknown tags gracefully.
	 *
	 * @since 3.16.0
	 */
	TagSupport struct {
		/**
		 * The tags supported by the client.
		 */
		ValueSet []SymbolTag `json:"valueSet"`
	} `json:"tagSupport,omitempty"`
	/**
	 * The client supports an additional label presented in the UI when
	 * registering a document symbol provider.
	 *
	 * @since 3.16.0
	 */
	LabelSupport bool `json:"labelSupport,omitempty"`
}

/**
 * Provider options for a [DocumentSymbolRequest](#DocumentSymbolRequest).
 */
type DocumentSymbolOptions struct {
	/**
	 * A human-readable string that is shown when multiple outlines trees
	 * are shown for the same document.
	 *
	 * @since 3.16.0
	 */
	Label string `json:"label,omitempty"`
	WorkDoneProgressOptions
}

/**
 * Parameters for a [DocumentSymbolRequest](#DocumentSymbolRequest).
 */
type DocumentSymbolParams struct {
	/**
	 * The text document.
	 */
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	WorkDoneProgressParams
	PartialResultParams
}

/**
 * A tagging type for string properties that are actually document URIs.
 */
type DocumentURI string

/**
 * The client capabilities of a [ExecuteCommandRequest](#ExecuteCommandRequest).
 */
type ExecuteCommandClientCapabilities struct {
	/**
	 * Execute command supports dynamic registration.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
}

/**
 * The server capabilities of a [ExecuteCommandRequest](#ExecuteCommandRequest).
 */
type ExecuteCommandOptions struct {
	/**
	 * The commands to be executed on the server
	 */
	Commands []string `json:"commands"`
	WorkDoneProgressOptions
}

/**
 * The parameters of a [ExecuteCommandRequest](#ExecuteCommandRequest).
 */
type ExecuteCommandParams struct {
	/**
	 * The identifier of the actual command handler.
	 */
	Command string `json:"command"`
	/**
	 * Arguments that the command should be invoked with.
	 */
	Arguments []json.RawMessage `json:"arguments,omitempty"`
	WorkDoneProgressParams
}

type FailureHandlingKind string

/**
 * The file event type
 */
type FileChangeType float64

/**
 * Represents information on a file/folder create.
 *
 * @since 3.16.0
 */
type FileCreate struct {
	/**
	 * A file:// URI for the location of the file/folder being created.
	 */
	URI string `json:"uri"`
}

/**
 * Represents information on a file/folder delete.
 *
 * @since 3.16.0
 */
type FileDelete struct {
	/**
	 * A file:// URI for the location of the file/folder being deleted.
	 */
	URI string `json:"uri"`
}

/**
 * An event describing a file change.
 */
type FileEvent struct {
	/**
	 * The file's uri.
	 */
	URI DocumentURI `json:"uri"`
	/**
	 * The change type.
	 */
	Type FileChangeType `json:"type"`
}

/**
 * Capabilities relating to events from file operations by the user in the client.
 *
 * These events do not come from the file system, they come from user operations
 * like renaming a file in the UI.
 *
 * @since 3.16.0
 */
type FileOperationClientCapabilities struct {
	/**
	 * Whether the client supports dynamic registration for file requests/notifications.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
	/**
	 * The client has support for sending didCreateFiles notifications.
	 */
	DidCreate bool `json:"didCreate,omitempty"`
	/**
	 * The client has support for willCreateFiles requests.
	 */
	WillCreate bool `json:"willCreate,omitempty"`
	/**
	 * The client has support for sending didRenameFiles notifications.
	 */
	DidRename bool `json:"didRename,omitempty"`
	/**
	 * The client has support for willRenameFiles requests.
	 */
	WillRename bool `json:"willRename,omitempty"`
	/**
	 * The client has support for sending didDeleteFiles notifications.
	 */
	DidDelete bool `json:"didDelete,omitempty"`
	/**
	 * The client has support for willDeleteFiles requests.
	 */
	WillDelete bool `json:"willDelete,omitempty"`
}

/**
 * A filter to describe in which file operation requests or notifications
 * the server is interested in.
 *
 * @since 3.16.0
 */
type FileOperationFilter struct {
	/**
	 * A Uri like `file` or `untitled`.
	 */
	Scheme string `json:"scheme,omitempty"`
	/**
	 * The actual file operation pattern.
	 */
	Pattern FileOperationPattern `json:"pattern"`
}

/**
 * Options for notifications/requests for user operations on files.
 *
 * @since 3.16.0
 */
type FileOperationOptions struct {
	/**
	* The server is interested in didCreateFiles notifications.
	 */
	DidCreate FileOperationRegistrationOptions `json:"didCreate,omitempty"`
	/**
	* The server is interested in willCreateFiles requests.
	 */
	WillCreate FileOperationRegistrationOptions `json:"willCreate,omitempty"`
	/**
	* The server is interested in didRenameFiles notifications.
	 */
	DidRename FileOperationRegistrationOptions `json:"didRename,omitempty"`
	/**
	* The server is interested in willRenameFiles requests.
	 */
	WillRename FileOperationRegistrationOptions `json:"willRename,omitempty"`
	/**
	* The server is interested in didDeleteFiles file notifications.
	 */
	DidDelete FileOperationRegistrationOptions `json:"didDelete,omitempty"`
	/**
	* The server is interested in willDeleteFiles file requests.
	 */
	WillDelete FileOperationRegistrationOptions `json:"willDelete,omitempty"`
}

/**
 * A pattern to describe in which file operation requests or notifications
 * the server is interested in.
 *
 * @since 3.16.0
 */
type FileOperationPattern struct {
	/**
	 * The glob pattern to match. Glob patterns can have the following syntax:
	 * - `*` to match one or more characters in a path segment
	 * - `?` to match on one character in a path segment
	 * - `**` to match any number of path segments, including none
	 * - `{}` to group sub patterns into an OR expression. (e.g. `**​/*.{ts,js}` matches all TypeScript and JavaScript files)
	 * - `[]` to declare a range of characters to match in a path segment (e.g., `example.[0-9]` to match on `example.0`, `example.1`, …)
	 * - `[!...]` to negate a range of characters to match in a path segment (e.g., `example.[!0-9]` to match on `example.a`, `example.b`, but not `example.0`)
	 */
	Glob string `json:"glob"`
	/**
	 * Whether to match files or folders with this pattern.
	 *
	 * Matches both if undefined.
	 */
	Matches FileOperationPatternKind `json:"matches,omitempty"`
	/**
	 * Additional options used during matching.
	 */
	Options FileOperationPatternOptions `json:"options,omitempty"`
}

/**
 * A pattern kind describing if a glob pattern matches a file a folder or
 * both.
 *
 * @since 3.16.0
 */
type FileOperationPatternKind string

/**
 * Matching options for the file operation pattern.
 *
 * @since 3.16.0
 */
type FileOperationPatternOptions struct {
	/**
	 * The pattern should be matched ignoring casing.
	 */
	IgnoreCase bool `json:"ignoreCase,omitempty"`
}

/**
 * The options to register for file operations.
 *
 * @since 3.16.0
 */
type FileOperationRegistrationOptions struct {
	/**
	 * The actual filters.
	 */
	Filters []FileOperationFilter `json:"filters"`
}

/**
 * Represents information on a file/folder rename.
 *
 * @since 3.16.0
 */
type FileRename struct {
	/**
	 * A file:// URI for the original location of the file/folder being renamed.
	 */
	OldURI string `json:"oldUri"`
	/**
	 * A file:// URI for the new location of the file/folder being renamed.
	 */
	NewURI string `json:"newUri"`
}

type FileSystemWatcher struct {
	/**
	 * The  glob pattern to watch. Glob patterns can have the following syntax:
	 * - `*` to match one or more characters in a path segment
	 * - `?` to match on one character in a path segment
	 * - `**` to match any number of path segments, including none
	 * - `{}` to group conditions (e.g. `**​/*.{ts,js}` matches all TypeScript and JavaScript files)
	 * - `[]` to declare a range of characters to match in a path segment (e.g., `example.[0-9]` to match on `example.0`, `example.1`, …)
	 * - `[!...]` to negate a range of characters to match in a path segment (e.g., `example.[!0-9]` to match on `example.a`, `example.b`, but not `example.0`)
	 */
	GlobPattern string `json:"globPattern"`
	/**
	 * The kind of events of interest. If omitted it defaults
	 * to WatchKind.Create | WatchKind.Change | WatchKind.Delete
	 * which is 7.
	 */
	Kind uint32 `json:"kind,omitempty"`
}

/**
 * Represents a folding range. To be valid, start and end line must be bigger than zero and smaller
 * than the number of lines in the document. Clients are free to ignore invalid ranges.
 */
type FoldingRange struct {
	/**
	 * The zero-based start line of the range to fold. The folded area starts after the line's last character.
	 * To be valid, the end must be zero or larger and smaller than the number of lines in the document.
	 */
	StartLine uint32 `json:"startLine"`
	/**
	 * The zero-based character offset from where the folded range starts. If not defined, defaults to the length of the start line.
	 */
	StartCharacter uint32 `json:"startCharacter,omitempty"`
	/**
	 * The zero-based end line of the range to fold. The folded area ends with the line's last character.
	 * To be valid, the end must be zero or larger and smaller than the number of lines in the document.
	 */
	EndLine uint32 `json:"endLine"`
	/**
	 * The zero-based character offset before the folded range ends. If not defined, defaults to the length of the end line.
	 */
	EndCharacter uint32 `json:"endCharacter,omitempty"`
	/**
	 * Describes the kind of the folding range such as `comment' or 'region'. The kind
	 * is used to categorize folding ranges and used by commands like 'Fold all comments'. See
	 * [FoldingRangeKind](#FoldingRangeKind) for an enumeration of standardized kinds.
	 */
	Kind string `json:"kind,omitempty"`
}

type FoldingRangeClientCapabilities struct {
	/**
	 * Whether implementation supports dynamic registration for folding range providers. If this is set to `true`
	 * the client supports the new `FoldingRangeRegistrationOptions` return value for the corresponding server
	 * capability as well.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
	/**
	 * The maximum number of folding ranges that the client prefers to receive per document. The value serves as a
	 * hint, servers are free to follow the limit.
	 */
	RangeLimit uint32 `json:"rangeLimit,omitempty"`
	/**
	 * If set, the client signals that it only supports folding complete lines. If set, client will
	 * ignore specified `startCharacter` and `endCharacter` properties in a FoldingRange.
	 */
	LineFoldingOnly bool `json:"lineFoldingOnly,omitempty"`
}

/**
 * Enum of known range kinds
 */
type FoldingRangeKind string

type FoldingRangeOptions struct {
	WorkDoneProgressOptions
}

/**
 * Parameters for a [FoldingRangeRequest](#FoldingRangeRequest).
 */
type FoldingRangeParams struct {
	/**
	 * The text document.
	 */
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	WorkDoneProgressParams
	PartialResultParams
}

type FoldingRangeRegistrationOptions struct {
	TextDocumentRegistrationOptions
	FoldingRangeOptions
	StaticRegistrationOptions
}

/**
 * Value-object describing what options formatting should use.
 */
type FormattingOptions struct {
	/**
	 * Size of a tab in spaces.
	 */
	TabSize uint32 `json:"tabSize"`
	/**
	 * Prefer spaces over tabs.
	 */
	InsertSpaces bool `json:"insertSpaces"`
	/**
	 * Trim trailing whitespaces on a line.
	 *
	 * @since 3.15.0
	 */
	TrimTrailingWhitespace bool `json:"trimTrailingWhitespace,omitempty"`
	/**
	 * Insert a newline character at the end of the file if one does not exist.
	 *
	 * @since 3.15.0
	 */
	InsertFinalNewline bool `json:"insertFinalNewline,omitempty"`
	/**
	 * Trim all newlines after the final newline at the end of the file.
	 *
	 * @since 3.15.0
	 */
	TrimFinalNewlines bool `json:"trimFinalNewlines,omitempty"`
}

/**
 * A diagnostic report with a full set of problems.
 *
 * @since 3.17.0 - proposed state
 */
type FullDocumentDiagnosticReport struct {
	/**
	 * A full document diagnostic report.
	 */
	Kind string `json:"kind"`
	/**
	 * An optional result id. If provided it will
	 * be sent on the next diagnostic request for the
	 * same document.
	 */
	ResultID string `json:"resultId,omitempty"`
	/**
	 * The actual items.
	 */
	Items []Diagnostic `json:"items"`
}

/**
 * General client capabilities.
 *
 * @since 3.16.0
 */
type GeneralClientCapabilities struct {
	/**
	 * Client capability that signals how the client
	 * handles stale requests (e.g. a request
	 * for which the client will not process the response
	 * anymore since the information is outdated).
	 *
	 * @since 3.17.0
	 */
	StaleRequestSupport struct {
		/**
		 * The client will actively cancel the request.
		 */
		Cancel bool `json:"cancel"`
		/**
		 * The list of requests for which the client
		 * will retry the request if it receives a
		 * response with error code `ContentModified`
		 */
		RetryOnContentModified []string `json:"retryOnContentModified"`
	} `json:"staleRequestSupport,omitempty"`
	/**
	 * Client capabilities specific to regular expressions.
	 *
	 * @since 3.16.0
	 */
	RegularExpressions RegularExpressionsClientCapabilities `json:"regularExpressions,omitempty"`
	/**
	 * Client capabilities specific to the client's markdown parser.
	 *
	 * @since 3.16.0
	 */
	Markdown MarkdownClientCapabilities `json:"markdown,omitempty"`
}

/**
 * The result of a hover request.
 */
type Hover struct {
	/**
	 * The hover's content
	 */
	Contents MarkupContent/*MarkupContent | MarkedString | MarkedString[]*/ `json:"contents"`
	/**
	 * An optional range
	 */
	Range Range `json:"range,omitempty"`
}

type HoverClientCapabilities struct {
	/**
	 * Whether hover supports dynamic registration.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
	/**
	 * Client supports the follow content formats for the content
	 * property. The order describes the preferred format of the client.
	 */
	ContentFormat []MarkupKind `json:"contentFormat,omitempty"`
}

/**
 * Hover options.
 */
type HoverOptions struct {
	WorkDoneProgressOptions
}

/**
 * Parameters for a [HoverRequest](#HoverRequest).
 */
type HoverParams struct {
	TextDocumentPositionParams
	WorkDoneProgressParams
}

/**
 * @since 3.6.0
 */
type ImplementationClientCapabilities struct {
	/**
	 * Whether implementation supports dynamic registration. If this is set to `true`
	 * the client supports the new `ImplementationRegistrationOptions` return value
	 * for the corresponding server capability as well.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
	/**
	 * The client supports additional metadata in the form of definition links.
	 *
	 * @since 3.14.0
	 */
	LinkSupport bool `json:"linkSupport,omitempty"`
}

type ImplementationOptions struct {
	WorkDoneProgressOptions
}

type ImplementationParams struct {
	TextDocumentPositionParams
	WorkDoneProgressParams
	PartialResultParams
}

type ImplementationRegistrationOptions struct {
	TextDocumentRegistrationOptions
	ImplementationOptions
	StaticRegistrationOptions
}

/**
 * Known error codes for an `InitializeError`;
 */
type InitializeError float64

type InitializeParams struct {
	/**
	 * The process Id of the parent process that started
	 * the server.
	 */
	ProcessID int32/*integer | null*/ `json:"processId"`
	/**
	 * Information about the client
	 *
	 * @since 3.15.0
	 */
	ClientInfo struct {
		/**
		 * The name of the client as defined by the client.
		 */
		Name string `json:"name"`
		/**
		 * The client's version as defined by the client.
		 */
		Version string `json:"version,omitempty"`
	} `json:"clientInfo,omitempty"`
	/**
	 * The locale the client is currently showing the user interface
	 * in. This must not necessarily be the locale of the operating
	 * system.
	 *
	 * Uses IETF language tags as the value's syntax
	 * (See https://en.wikipedia.org/wiki/IETF_language_tag)
	 *
	 * @since 3.16.0
	 */
	Locale string `json:"locale,omitempty"`
	/**
	 * The rootPath of the workspace. Is null
	 * if no folder is open.
	 *
	 * @deprecated in favour of rootUri.
	 */
	RootPath string/*string | null*/ `json:"rootPath,omitempty"`
	/**
	 * The rootUri of the workspace. Is null if no
	 * folder is open. If both `rootPath` and `rootUri` are set
	 * `rootUri` wins.
	 *
	 * @deprecated in favour of workspaceFolders.
	 */
	RootURI DocumentURI/*DocumentUri | null*/ `json:"rootUri"`
	/**
	 * The capabilities provided by the client (editor or tool)
	 */
	Capabilities ClientCapabilities `json:"capabilities"`
	/**
	 * User provided initialization options.
	 */
	InitializationOptions LSPAny `json:"initializationOptions,omitempty"`
	/**
	 * The initial trace setting. If omitted trace is disabled ('off').
	 */
	Trace string/* 'off' | 'messages' | 'compact' | 'verbose' */ `json:"trace,omitempty"`
	/**
	 * The actual configured workspace folders.
	 */
	WorkspaceFolders []WorkspaceFolder/*WorkspaceFolder[] | null*/ `json:"workspaceFolders"`
}

/**
 * The result returned from an initialize request.
 */
type InitializeResult struct {
	/**
	 * The capabilities the language server provides.
	 */
	Capabilities ServerCapabilities `json:"capabilities"`
	/**
	 * Information about the server.
	 *
	 * @since 3.15.0
	 */
	ServerInfo struct {
		/**
		 * The name of the server as defined by the server.
		 */
		Name string `json:"name"`
		/**
		 * The server's version as defined by the server.
		 */
		Version string `json:"version,omitempty"`
	} `json:"serverInfo,omitempty"`
}

type InitializedParams struct {
}

/**
 * Inline value information can be provided by different means:
 * - directly as a text value (class InlineValueText).
 * - as a name to use for a variable lookup (class InlineValueVariableLookup)
 * - as an evaluatable expression (class InlineValueEvaluatableExpression)
 * The InlineValue types combines all inline value types into one type.
 *
 * @since 3.17.0 - proposed state
 */
type InlineValue = interface{} /* InlineValueText | InlineValueVariableLookup | InlineValueEvaluatableExpression*/

/**
 * Provide an inline value through an expression evaluation.
 * If only a range is specified, the expression will be extracted from the underlying document.
 * An optional expression can be used to override the extracted expression.
 *
 * @since 3.17.0 - proposed state
 */
type InlineValueEvaluatableExpression struct {
	/**
	 * The document range for which the inline value applies.
	 * The range is used to extract the evaluatable expression from the underlying document.
	 */
	Range Range `json:"range"`
	/**
	 * If specified the expression overrides the extracted expression.
	 */
	Expression string `json:"expression,omitempty"`
}

/**
 * Provide inline value as text.
 *
 * @since 3.17.0 - proposed state
 */
type InlineValueText struct {
	/**
	 * The document range for which the inline value applies.
	 */
	Range Range `json:"range"`
	/**
	 * The text of the inline value.
	 */
	Text string `json:"text"`
}

/**
 * Provide inline value through a variable lookup.
 * If only a range is specified, the variable name will be extracted from the underlying document.
 * An optional variable name can be used to override the extracted name.
 *
 * @since 3.17.0 - proposed state
 */
type InlineValueVariableLookup struct {
	/**
	 * The document range for which the inline value applies.
	 * The range is used to extract the variable name from the underlying document.
	 */
	Range Range `json:"range"`
	/**
	 * If specified the name of the variable to look up.
	 */
	VariableName string `json:"variableName,omitempty"`
	/**
	 * How to perform the lookup.
	 */
	CaseSensitiveLookup bool `json:"caseSensitiveLookup"`
}

/**
 * Client capabilities specific to inline values.
 *
 * @since 3.17.0 - proposed state
 */
type InlineValuesClientCapabilities struct {
	/**
	 * Whether implementation supports dynamic registration for inline value providers.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
}

/**
 * @since 3.17.0 - proposed state
 */
type InlineValuesContext struct {
	/**
	 * The document range where execution has stopped.
	 * Typically the end position of the range denotes the line where the inline values are shown.
	 */
	StoppedLocation Range `json:"stoppedLocation"`
}

/**
 * Inline values options used during static registration.
 *
 * @since 3.17.0 - proposed state
 */
type InlineValuesOptions struct {
	WorkDoneProgressOptions
}

/**
 * A parameter literal used in inline values requests.
 *
 * @since 3.17.0 - proposed state
 */
type InlineValuesParams struct {
	/**
	 * The text document.
	 */
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	/**
	 * The visible document range for which inline values should be computed.
	 */
	ViewPort Range `json:"viewPort"`
	/**
	 * Additional information about the context in which inline values were
	 * requested.
	 */
	Context InlineValuesContext `json:"context"`
	WorkDoneProgressParams
}

/**
 * Inline value options used during static or dynamic registration.
 *
 * @since 3.17.0 - proposed state
 */
type InlineValuesRegistrationOptions struct {
	InlineValuesOptions
	TextDocumentRegistrationOptions
	StaticRegistrationOptions
}

/**
 * Client workspace capabilities specific to inline values.
 *
 * @since 3.17.0 - proposed state
 */
type InlineValuesWorkspaceClientCapabilities struct {
	/**
	 * Whether the client implementation supports a refresh request sent from the
	 * server to the client.
	 *
	 * Note that this event is global and will force the client to refresh all
	 * inline values currently shown. It should be used with absolute care and is
	 * useful for situation where a server for example detect a project wide
	 * change that requires such a calculation.
	 */
	RefreshSupport bool `json:"refreshSupport,omitempty"`
}

/**
 * A special text edit to provide an insert and a replace operation.
 *
 * @since 3.16.0
 */
type InsertReplaceEdit struct {
	/**
	 * The string to be inserted.
	 */
	NewText string `json:"newText"`
	/**
	 * The range if the insert is requested
	 */
	Insert Range `json:"insert"`
	/**
	 * The range if the replace is requested.
	 */
	Replace Range `json:"replace"`
}

/**
 * Defines whether the insert text in a completion item should be interpreted as
 * plain text or a snippet.
 */
type InsertTextFormat float64

/**
 * How whitespace and indentation is handled during completion
 * item insertion.
 *
 * @since 3.16.0
 */
type InsertTextMode float64

/**
 * The LSP any type
 *
 * @since 3.17.0
 */
type LSPAny = interface{} /* LSPObject | LSPArray | string | int32 | uint32 | Decimal | bool | float64*/

/**
 * LSP arrays.
 *
 * @since 3.17.0
 */
type LSPArray = []LSPAny

/**
 * LSP object definition.
 *
 * @since 3.17.0
 */
type LSPObject = map[string]interface{} /*[key: string]: LSPAny*/

/**
 * Client capabilities for the linked editing range request.
 *
 * @since 3.16.0
 */
type LinkedEditingRangeClientCapabilities struct {
	/**
	 * Whether implementation supports dynamic registration. If this is set to `true`
	 * the client supports the new `(TextDocumentRegistrationOptions & StaticRegistrationOptions)`
	 * return value for the corresponding server capability as well.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
}

type LinkedEditingRangeOptions struct {
	WorkDoneProgressOptions
}

type LinkedEditingRangeParams struct {
	TextDocumentPositionParams
	WorkDoneProgressParams
}

type LinkedEditingRangeRegistrationOptions struct {
	TextDocumentRegistrationOptions
	LinkedEditingRangeOptions
	StaticRegistrationOptions
}

/**
 * The result of a linked editing range request.
 *
 * @since 3.16.0
 */
type LinkedEditingRanges struct {
	/**
	 * A list of ranges that can be edited together. The ranges must have
	 * identical length and contain identical text content. The ranges cannot overlap.
	 */
	Ranges []Range `json:"ranges"`
	/**
	 * An optional word pattern (regular expression) that describes valid contents for
	 * the given ranges. If no pattern is provided, the client configuration's word
	 * pattern will be used.
	 */
	WordPattern string `json:"wordPattern,omitempty"`
}

/**
 * Represents a location inside a resource, such as a line
 * inside a text file.
 */
type Location struct {
	URI   DocumentURI `json:"uri"`
	Range Range       `json:"range"`
}

/**
 * Represents the connection of two locations. Provides additional metadata over normal [locations](#Location),
 * including an origin range.
 */
type LocationLink struct {
	/**
	 * Span of the origin of this link.
	 *
	 * Used as the underlined span for mouse definition hover. Defaults to the word range at
	 * the definition position.
	 */
	OriginSelectionRange Range `json:"originSelectionRange,omitempty"`
	/**
	 * The target resource identifier of this link.
	 */
	TargetURI DocumentURI `json:"targetUri"`
	/**
	 * The full target range of this link. If the target for example is a symbol then target range is the
	 * range enclosing this symbol not including leading/trailing whitespace but everything else
	 * like comments. This information is typically used to highlight the range in the editor.
	 */
	TargetRange Range `json:"targetRange"`
	/**
	 * The range that should be selected and revealed when this link is being followed, e.g the name of a function.
	 * Must be contained by the the `targetRange`. See also `DocumentSymbol#range`
	 */
	TargetSelectionRange Range `json:"targetSelectionRange"`
}

/**
 * The log message parameters.
 */
type LogMessageParams struct {
	/**
	 * The message type. See {@link MessageType}
	 */
	Type MessageType `json:"type"`
	/**
	 * The actual message
	 */
	Message string `json:"message"`
}

type LogTraceParams struct {
	Message string `json:"message"`
	Verbose string `json:"verbose,omitempty"`
}

/**
 * Client capabilities specific to the used markdown parser.
 *
 * @since 3.16.0
 */
type MarkdownClientCapabilities struct {
	/**
	 * The name of the parser.
	 */
	Parser string `json:"parser"`
	/**
	 * The version of the parser.
	 */
	Version string `json:"version,omitempty"`
	/**
	 * A list of HTML tags that the client allows / supports in
	 * Markdown.
	 *
	 * @since 3.17.0
	 */
	AllowedTags []string `json:"allowedTags,omitempty"`
}

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
type MarkedString = string /*string | { language: string; value: string }*/

/**
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
	/**
	 * The type of the Markup
	 */
	Kind MarkupKind `json:"kind"`
	/**
	 * The content itself
	 */
	Value string `json:"value"`
}

/**
 * Describes the content type that a client supports in various
 * result literals like `Hover`, `ParameterInfo` or `CompletionItem`.
 *
 * Please note that `MarkupKinds` must not start with a `$`. This kinds
 * are reserved for internal usage.
 */
type MarkupKind string

type MessageActionItem struct {
	/**
	 * A short title like 'Retry', 'Open Log' etc.
	 */
	Title string `json:"title"`
}

/**
 * The message type
 */
type MessageType float64

/**
 * Moniker definition to match LSIF 0.5 moniker definition.
 *
 * @since 3.16.0
 */
type Moniker struct {
	/**
	 * The scheme of the moniker. For example tsc or .Net
	 */
	Scheme string `json:"scheme"`
	/**
	 * The identifier of the moniker. The value is opaque in LSIF however
	 * schema owners are allowed to define the structure if they want.
	 */
	Identifier string `json:"identifier"`
	/**
	 * The scope in which the moniker is unique
	 */
	Unique UniquenessLevel `json:"unique"`
	/**
	 * The moniker kind if known.
	 */
	Kind MonikerKind `json:"kind,omitempty"`
}

/**
 * Client capabilities specific to the moniker request.
 *
 * @since 3.16.0
 */
type MonikerClientCapabilities struct {
	/**
	 * Whether moniker supports dynamic registration. If this is set to `true`
	 * the client supports the new `MonikerRegistrationOptions` return value
	 * for the corresponding server capability as well.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
}

/**
 * The moniker kind.
 *
 * @since 3.16.0
 */
type MonikerKind string

type MonikerOptions struct {
	WorkDoneProgressOptions
}

type MonikerParams struct {
	TextDocumentPositionParams
	WorkDoneProgressParams
	PartialResultParams
}

type MonikerRegistrationOptions struct {
	TextDocumentRegistrationOptions
	MonikerOptions
}

/**
 * A text document identifier to optionally denote a specific version of a text document.
 */
type OptionalVersionedTextDocumentIdentifier struct {
	/**
	 * The version number of this document. If a versioned text document identifier
	 * is sent from the server to the client and the file is not open in the editor
	 * (the server has not received an open notification before) the server can send
	 * `null` to indicate that the version is unknown and the content on disk is the
	 * truth (as specified with document content ownership).
	 */
	Version int32/*integer | null*/ `json:"version"`
	TextDocumentIdentifier
}

/**
 * Represents a parameter of a callable-signature. A parameter can
 * have a label and a doc-comment.
 */
type ParameterInformation struct {
	/**
	 * The label of this parameter information.
	 *
	 * Either a string or an inclusive start and exclusive end offsets within its containing
	 * signature label. (see SignatureInformation.label). The offsets are based on a UTF-16
	 * string representation as `Position` and `Range` does.
	 *
	 * *Note*: a label of type string should be a substring of its containing signature label.
	 * Its intended use case is to highlight the parameter label part in the `SignatureInformation.label`.
	 */
	Label string/*string | [uinteger, uinteger]*/ `json:"label"`
	/**
	 * The human-readable doc-comment of this signature. Will be shown
	 * in the UI but can be omitted.
	 */
	Documentation string/*string | MarkupContent*/ `json:"documentation,omitempty"`
}

type PartialResultParams struct {
	/**
	 * An optional token that a server can use to report partial results (e.g. streaming) to
	 * the client.
	 */
	PartialResultToken ProgressToken `json:"partialResultToken,omitempty"`
}

/**
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
	/**
	 * Line position in a document (zero-based).
	 */
	Line uint32 `json:"line"`
	/**
	 * Character offset on a line in a document (zero-based). Assuming that the line is
	 * represented as a string, the `character` value represents the gap between the
	 * `character` and `character + 1`.
	 *
	 * If the character value is greater than the line length it defaults back to the
	 * line length.
	 */
	Character uint32 `json:"character"`
}

type PrepareRenameParams struct {
	TextDocumentPositionParams
	WorkDoneProgressParams
}

type PrepareSupportDefaultBehavior = interface{}

/**
 * A previous result id in a workspace pull request.
 *
 * @since 3.17.0 - proposed state
 */
type PreviousResultID = struct {
	/**
	 * The URI for which the client knowns a
	 * result id.
	 */
	URI DocumentURI `json:"uri"`
	/**
	 * The value of the previous result id.
	 */
	Value string `json:"value"`
}

type ProgressParams struct {
	/**
	 * The progress token provided by the client or server.
	 */
	Token ProgressToken `json:"token"`
	/**
	 * The progress data.
	 */
	Value interface{} `json:"value"`
}

type ProgressToken = interface{} /*number | string*/

/**
 * The publish diagnostic client capabilities.
 */
type PublishDiagnosticsClientCapabilities struct {
	/**
	 * Whether the clients accepts diagnostics with related information.
	 */
	RelatedInformation bool `json:"relatedInformation,omitempty"`
	/**
	 * Client supports the tag property to provide meta data about a diagnostic.
	 * Clients supporting tags have to handle unknown tags gracefully.
	 *
	 * @since 3.15.0
	 */
	TagSupport struct {
		/**
		 * The tags supported by the client.
		 */
		ValueSet []DiagnosticTag `json:"valueSet"`
	} `json:"tagSupport,omitempty"`
	/**
	 * Whether the client interprets the version property of the
	 * `textDocument/publishDiagnostics` notification`s parameter.
	 *
	 * @since 3.15.0
	 */
	VersionSupport bool `json:"versionSupport,omitempty"`
	/**
	 * Client supports a codeDescription property
	 *
	 * @since 3.16.0
	 */
	CodeDescriptionSupport bool `json:"codeDescriptionSupport,omitempty"`
	/**
	 * Whether code action supports the `data` property which is
	 * preserved between a `textDocument/publishDiagnostics` and
	 * `textDocument/codeAction` request.
	 *
	 * @since 3.16.0
	 */
	DataSupport bool `json:"dataSupport,omitempty"`
}

/**
 * The publish diagnostic notification's parameters.
 */
type PublishDiagnosticsParams struct {
	/**
	 * The URI for which diagnostic information is reported.
	 */
	URI DocumentURI `json:"uri"`
	/**
	 * Optional the version number of the document the diagnostics are published for.
	 *
	 * @since 3.15.0
	 */
	Version int32 `json:"version,omitempty"`
	/**
	 * An array of diagnostic information items.
	 */
	Diagnostics []Diagnostic `json:"diagnostics"`
}

/**
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
	/**
	 * The range's start position
	 */
	Start Position `json:"start"`
	/**
	 * The range's end position.
	 */
	End Position `json:"end"`
}

/**
 * Client Capabilities for a [ReferencesRequest](#ReferencesRequest).
 */
type ReferenceClientCapabilities struct {
	/**
	 * Whether references supports dynamic registration.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
}

/**
 * Value-object that contains additional information when
 * requesting references.
 */
type ReferenceContext struct {
	/**
	 * Include the declaration of the current symbol.
	 */
	IncludeDeclaration bool `json:"includeDeclaration"`
}

/**
 * Reference options.
 */
type ReferenceOptions struct {
	WorkDoneProgressOptions
}

/**
 * Parameters for a [ReferencesRequest](#ReferencesRequest).
 */
type ReferenceParams struct {
	Context ReferenceContext `json:"context"`
	TextDocumentPositionParams
	WorkDoneProgressParams
	PartialResultParams
}

/**
 * General parameters to to register for an notification or to register a provider.
 */
type Registration struct {
	/**
	 * The id used to register the request. The id can be used to deregister
	 * the request again.
	 */
	ID string `json:"id"`
	/**
	 * The method to register for.
	 */
	Method string `json:"method"`
	/**
	 * Options necessary for the registration.
	 */
	RegisterOptions LSPAny `json:"registerOptions,omitempty"`
}

type RegistrationParams struct {
	Registrations []Registration `json:"registrations"`
}

/**
 * Client capabilities specific to regular expressions.
 *
 * @since 3.16.0
 */
type RegularExpressionsClientCapabilities struct {
	/**
	 * The engine's name.
	 */
	Engine string `json:"engine"`
	/**
	 * The engine's version.
	 */
	Version string `json:"version,omitempty"`
}

/**
 * A full diagnostic report with a set of related documents.
 *
 * @since 3.17.0 - proposed state
 */
type RelatedFullDocumentDiagnosticReport struct {
	/**
	 * Diagnostics of related documents. This information is useful
	 * in programming languages where code in a file A can generate
	 * diagnostics in a file B which A depends on. An example of
	 * such a language is C/C++ where marco definitions in a file
	 * a.cpp and result in errors in a header file b.hpp.
	 *
	 * @since 3.17.0 - proposed state
	 */
	RelatedDocuments map[string]interface{}/*[uri: string ** DocumentUri *]: FullDocumentDiagnosticReport | UnchangedDocumentDiagnosticReport;*/ `json:"relatedDocuments,omitempty"`
	FullDocumentDiagnosticReport
}

/**
 * An unchanged diagnostic report with a set of related documents.
 *
 * @since 3.17.0 - proposed state
 */
type RelatedUnchangedDocumentDiagnosticReport struct {
	/**
	 * Diagnostics of related documents. This information is useful
	 * in programming languages where code in a file A can generate
	 * diagnostics in a file B which A depends on. An example of
	 * such a language is C/C++ where marco definitions in a file
	 * a.cpp and result in errors in a header file b.hpp.
	 *
	 * @since 3.17.0 - proposed state
	 */
	RelatedDocuments map[string]interface{}/*[uri: string ** DocumentUri *]: FullDocumentDiagnosticReport | UnchangedDocumentDiagnosticReport;*/ `json:"relatedDocuments,omitempty"`
	UnchangedDocumentDiagnosticReport
}

type RenameClientCapabilities struct {
	/**
	 * Whether rename supports dynamic registration.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
	/**
	 * Client supports testing for validity of rename operations
	 * before execution.
	 *
	 * @since 3.12.0
	 */
	PrepareSupport bool `json:"prepareSupport,omitempty"`
	/**
	 * Client supports the default behavior result.
	 *
	 * The value indicates the default behavior used by the
	 * client.
	 *
	 * @since 3.16.0
	 */
	PrepareSupportDefaultBehavior PrepareSupportDefaultBehavior `json:"prepareSupportDefaultBehavior,omitempty"`
	/**
	 * Whether th client honors the change annotations in
	 * text edits and resource operations returned via the
	 * rename request's workspace edit by for example presenting
	 * the workspace edit in the user interface and asking
	 * for confirmation.
	 *
	 * @since 3.16.0
	 */
	HonorsChangeAnnotations bool `json:"honorsChangeAnnotations,omitempty"`
}

/**
 * Rename file operation
 */
type RenameFile struct {
	/**
	 * A rename
	 */
	Kind string `json:"kind"`
	/**
	 * The old (existing) location.
	 */
	OldURI DocumentURI `json:"oldUri"`
	/**
	 * The new location.
	 */
	NewURI DocumentURI `json:"newUri"`
	/**
	 * Rename options.
	 */
	Options RenameFileOptions `json:"options,omitempty"`
	ResourceOperation
}

/**
 * Rename file options
 */
type RenameFileOptions struct {
	/**
	 * Overwrite target if existing. Overwrite wins over `ignoreIfExists`
	 */
	Overwrite bool `json:"overwrite,omitempty"`
	/**
	 * Ignores if target exists.
	 */
	IgnoreIfExists bool `json:"ignoreIfExists,omitempty"`
}

/**
 * The parameters sent in file rename requests/notifications.
 *
 * @since 3.16.0
 */
type RenameFilesParams struct {
	/**
	 * An array of all files/folders renamed in this operation. When a folder is renamed, only
	 * the folder will be included, and not its children.
	 */
	Files []FileRename `json:"files"`
}

/**
 * Provider options for a [RenameRequest](#RenameRequest).
 */
type RenameOptions struct {
	/**
	 * Renames should be checked and tested before being executed.
	 *
	 * @since version 3.12.0
	 */
	PrepareProvider bool `json:"prepareProvider,omitempty"`
	WorkDoneProgressOptions
}

/**
 * The parameters of a [RenameRequest](#RenameRequest).
 */
type RenameParams struct {
	/**
	 * The document to rename.
	 */
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	/**
	 * The position at which this request was sent.
	 */
	Position Position `json:"position"`
	/**
	 * The new name of the symbol. If the given name is not valid the
	 * request must return a [ResponseError](#ResponseError) with an
	 * appropriate message set.
	 */
	NewName string `json:"newName"`
	WorkDoneProgressParams
}

/**
 * A generic resource operation.
 */
type ResourceOperation struct {
	/**
	 * The resource operation kind.
	 */
	Kind string `json:"kind"`
	/**
	 * An optional annotation identifier describing the operation.
	 *
	 * @since 3.16.0
	 */
	AnnotationID ChangeAnnotationIdentifier `json:"annotationId,omitempty"`
}

type ResourceOperationKind string

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
 * A selection range represents a part of a selection hierarchy. A selection range
 * may have a parent selection range that contains it.
 */
type SelectionRange struct {
	/**
	 * The [range](#Range) of this selection range.
	 */
	Range Range `json:"range"`
	/**
	 * The parent selection range containing this range. Therefore `parent.range` must contain `this.range`.
	 */
	Parent *SelectionRange `json:"parent,omitempty"`
}

type SelectionRangeClientCapabilities struct {
	/**
	 * Whether implementation supports dynamic registration for selection range providers. If this is set to `true`
	 * the client supports the new `SelectionRangeRegistrationOptions` return value for the corresponding server
	 * capability as well.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
}

type SelectionRangeOptions struct {
	WorkDoneProgressOptions
}

/**
 * A parameter literal used in selection range requests.
 */
type SelectionRangeParams struct {
	/**
	 * The text document.
	 */
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	/**
	 * The positions inside the text document.
	 */
	Positions []Position `json:"positions"`
	WorkDoneProgressParams
	PartialResultParams
}

type SelectionRangeRegistrationOptions struct {
	SelectionRangeOptions
	TextDocumentRegistrationOptions
	StaticRegistrationOptions
}

/**
 * @since 3.16.0
 */
type SemanticTokens struct {
	/**
	 * An optional result id. If provided and clients support delta updating
	 * the client will include the result id in the next semantic token request.
	 * A server can then instead of computing all semantic tokens again simply
	 * send a delta.
	 */
	ResultID string `json:"resultId,omitempty"`
	/**
	 * The actual tokens.
	 */
	Data []uint32 `json:"data"`
}

/**
 * @since 3.16.0
 */
type SemanticTokensClientCapabilities struct {
	/**
	 * Whether implementation supports dynamic registration. If this is set to `true`
	 * the client supports the new `(TextDocumentRegistrationOptions & StaticRegistrationOptions)`
	 * return value for the corresponding server capability as well.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
	/**
	 * Which requests the client supports and might send to the server
	 * depending on the server's capability. Please note that clients might not
	 * show semantic tokens or degrade some of the user experience if a range
	 * or full request is advertised by the client but not provided by the
	 * server. If for example the client capability `requests.full` and
	 * `request.range` are both set to true but the server only provides a
	 * range provider the client might not render a minimap correctly or might
	 * even decide to not show any semantic tokens at all.
	 */
	Requests struct {
		/**
		 * The client will send the `textDocument/semanticTokens/range` request if
		 * the server provides a corresponding handler.
		 */
		Range bool/*boolean | {		}*/ `json:"range,omitempty"`
		/**
		 * The client will send the `textDocument/semanticTokens/full` request if
		 * the server provides a corresponding handler.
		 */
		Full interface{}/*boolean | <elided struct>*/ `json:"full,omitempty"`
	} `json:"requests"`
	/**
	 * The token types that the client supports.
	 */
	TokenTypes []string `json:"tokenTypes"`
	/**
	 * The token modifiers that the client supports.
	 */
	TokenModifiers []string `json:"tokenModifiers"`
	/**
	 * The token formats the clients supports.
	 */
	Formats []TokenFormat `json:"formats"`
	/**
	 * Whether the client supports tokens that can overlap each other.
	 */
	OverlappingTokenSupport bool `json:"overlappingTokenSupport,omitempty"`
	/**
	 * Whether the client supports tokens that can span multiple lines.
	 */
	MultilineTokenSupport bool `json:"multilineTokenSupport,omitempty"`
	/**
	 * Whether the client allows the server to actively cancel a
	 * semantic token request, e.g. supports returning
	 * LSPErrorCodes.ServerCancelled. If a server does the client
	 * needs to retrigger the request.
	 *
	 * @since 3.17.0
	 */
	ServerCancelSupport bool `json:"serverCancelSupport,omitempty"`
	/**
	 * Whether the client uses semantic tokens to augment existing
	 * syntax tokens. If set to `true` client side created syntax
	 * tokens and semantic tokens are both used for colorization. If
	 * set to `false` the client only uses the returned semantic tokens
	 * for colorization.
	 *
	 * If the value is `undefined` then the client behavior is not
	 * specified.
	 *
	 * @since 3.17.0
	 */
	AugmentsSyntaxTokens bool `json:"augmentsSyntaxTokens,omitempty"`
}

/**
 * @since 3.16.0
 */
type SemanticTokensDelta struct {
	ResultID string `json:"resultId,omitempty"`
	/**
	 * The semantic token edits to transform a previous result into a new result.
	 */
	Edits []SemanticTokensEdit `json:"edits"`
}

/**
 * @since 3.16.0
 */
type SemanticTokensDeltaParams struct {
	/**
	 * The text document.
	 */
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	/**
	 * The result id of a previous response. The result Id can either point to a full response
	 * or a delta response depending on what was received last.
	 */
	PreviousResultID string `json:"previousResultId"`
	WorkDoneProgressParams
	PartialResultParams
}

/**
 * @since 3.16.0
 */
type SemanticTokensEdit struct {
	/**
	 * The start offset of the edit.
	 */
	Start uint32 `json:"start"`
	/**
	 * The count of elements to remove.
	 */
	DeleteCount uint32 `json:"deleteCount"`
	/**
	 * The elements to insert.
	 */
	Data []uint32 `json:"data,omitempty"`
}

/**
 * @since 3.16.0
 */
type SemanticTokensLegend struct {
	/**
	 * The token types a server uses.
	 */
	TokenTypes []string `json:"tokenTypes"`
	/**
	 * The token modifiers a server uses.
	 */
	TokenModifiers []string `json:"tokenModifiers"`
}

/**
 * @since 3.16.0
 */
type SemanticTokensOptions struct {
	/**
	 * The legend used by the server
	 */
	Legend SemanticTokensLegend `json:"legend"`
	/**
	 * Server supports providing semantic tokens for a specific range
	 * of a document.
	 */
	Range bool/*boolean | {	}*/ `json:"range,omitempty"`
	/**
	 * Server supports providing semantic tokens for a full document.
	 */
	Full interface{}/*boolean | <elided struct>*/ `json:"full,omitempty"`
	WorkDoneProgressOptions
}

/**
 * @since 3.16.0
 */
type SemanticTokensParams struct {
	/**
	 * The text document.
	 */
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	WorkDoneProgressParams
	PartialResultParams
}

/**
 * @since 3.16.0
 */
type SemanticTokensRangeParams struct {
	/**
	 * The text document.
	 */
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	/**
	 * The range the semantic tokens are requested for.
	 */
	Range Range `json:"range"`
	WorkDoneProgressParams
	PartialResultParams
}

/**
 * @since 3.16.0
 */
type SemanticTokensRegistrationOptions struct {
	TextDocumentRegistrationOptions
	SemanticTokensOptions
	StaticRegistrationOptions
}

/**
 * @since 3.16.0
 */
type SemanticTokensWorkspaceClientCapabilities struct {
	/**
	 * Whether the client implementation supports a refresh request sent from
	 * the server to the client.
	 *
	 * Note that this event is global and will force the client to refresh all
	 * semantic tokens currently shown. It should be used with absolute care
	 * and is useful for situation where a server for example detects a project
	 * wide change that requires such a calculation.
	 */
	RefreshSupport bool `json:"refreshSupport,omitempty"`
}

type ServerCapabilities struct {
	/**
	 * Defines how text documents are synced. Is either a detailed structure defining each notification or
	 * for backwards compatibility the TextDocumentSyncKind number.
	 */
	TextDocumentSync interface{}/*TextDocumentSyncOptions | TextDocumentSyncKind*/ `json:"textDocumentSync,omitempty"`
	/**
	 * The server provides completion support.
	 */
	CompletionProvider CompletionOptions `json:"completionProvider,omitempty"`
	/**
	 * The server provides hover support.
	 */
	HoverProvider bool/*boolean | HoverOptions*/ `json:"hoverProvider,omitempty"`
	/**
	 * The server provides signature help support.
	 */
	SignatureHelpProvider SignatureHelpOptions `json:"signatureHelpProvider,omitempty"`
	/**
	 * The server provides Goto Declaration support.
	 */
	DeclarationProvider interface{}/* bool | DeclarationOptions | DeclarationRegistrationOptions*/ `json:"declarationProvider,omitempty"`
	/**
	 * The server provides goto definition support.
	 */
	DefinitionProvider bool/*boolean | DefinitionOptions*/ `json:"definitionProvider,omitempty"`
	/**
	 * The server provides Goto Type Definition support.
	 */
	TypeDefinitionProvider interface{}/* bool | TypeDefinitionOptions | TypeDefinitionRegistrationOptions*/ `json:"typeDefinitionProvider,omitempty"`
	/**
	 * The server provides Goto Implementation support.
	 */
	ImplementationProvider interface{}/* bool | ImplementationOptions | ImplementationRegistrationOptions*/ `json:"implementationProvider,omitempty"`
	/**
	 * The server provides find references support.
	 */
	ReferencesProvider bool/*boolean | ReferenceOptions*/ `json:"referencesProvider,omitempty"`
	/**
	 * The server provides document highlight support.
	 */
	DocumentHighlightProvider bool/*boolean | DocumentHighlightOptions*/ `json:"documentHighlightProvider,omitempty"`
	/**
	 * The server provides document symbol support.
	 */
	DocumentSymbolProvider bool/*boolean | DocumentSymbolOptions*/ `json:"documentSymbolProvider,omitempty"`
	/**
	 * The server provides code actions. CodeActionOptions may only be
	 * specified if the client states that it supports
	 * `codeActionLiteralSupport` in its initial `initialize` request.
	 */
	CodeActionProvider interface{}/*boolean | CodeActionOptions*/ `json:"codeActionProvider,omitempty"`
	/**
	 * The server provides code lens.
	 */
	CodeLensProvider CodeLensOptions `json:"codeLensProvider,omitempty"`
	/**
	 * The server provides document link support.
	 */
	DocumentLinkProvider DocumentLinkOptions `json:"documentLinkProvider,omitempty"`
	/**
	 * The server provides color provider support.
	 */
	ColorProvider interface{}/* bool | DocumentColorOptions | DocumentColorRegistrationOptions*/ `json:"colorProvider,omitempty"`
	/**
	 * The server provides workspace symbol support.
	 */
	WorkspaceSymbolProvider bool/*boolean | WorkspaceSymbolOptions*/ `json:"workspaceSymbolProvider,omitempty"`
	/**
	 * The server provides document formatting.
	 */
	DocumentFormattingProvider bool/*boolean | DocumentFormattingOptions*/ `json:"documentFormattingProvider,omitempty"`
	/**
	 * The server provides document range formatting.
	 */
	DocumentRangeFormattingProvider bool/*boolean | DocumentRangeFormattingOptions*/ `json:"documentRangeFormattingProvider,omitempty"`
	/**
	 * The server provides document formatting on typing.
	 */
	DocumentOnTypeFormattingProvider DocumentOnTypeFormattingOptions `json:"documentOnTypeFormattingProvider,omitempty"`
	/**
	 * The server provides rename support. RenameOptions may only be
	 * specified if the client states that it supports
	 * `prepareSupport` in its initial `initialize` request.
	 */
	RenameProvider interface{}/*boolean | RenameOptions*/ `json:"renameProvider,omitempty"`
	/**
	 * The server provides folding provider support.
	 */
	FoldingRangeProvider interface{}/* bool | FoldingRangeOptions | FoldingRangeRegistrationOptions*/ `json:"foldingRangeProvider,omitempty"`
	/**
	 * The server provides selection range support.
	 */
	SelectionRangeProvider interface{}/* bool | SelectionRangeOptions | SelectionRangeRegistrationOptions*/ `json:"selectionRangeProvider,omitempty"`
	/**
	 * The server provides execute command support.
	 */
	ExecuteCommandProvider ExecuteCommandOptions `json:"executeCommandProvider,omitempty"`
	/**
	 * The server provides call hierarchy support.
	 *
	 * @since 3.16.0
	 */
	CallHierarchyProvider interface{}/* bool | CallHierarchyOptions | CallHierarchyRegistrationOptions*/ `json:"callHierarchyProvider,omitempty"`
	/**
	 * The server provides linked editing range support.
	 *
	 * @since 3.16.0
	 */
	LinkedEditingRangeProvider interface{}/* bool | LinkedEditingRangeOptions | LinkedEditingRangeRegistrationOptions*/ `json:"linkedEditingRangeProvider,omitempty"`
	/**
	 * The server provides semantic tokens support.
	 *
	 * @since 3.16.0
	 */
	SemanticTokensProvider interface{}/*SemanticTokensOptions | SemanticTokensRegistrationOptions*/ `json:"semanticTokensProvider,omitempty"`
	/**
	 * The workspace server capabilities
	 */
	Workspace Workspace5Gn `json:"workspace,omitempty"`
	/**
	 * The server provides moniker support.
	 *
	 * @since 3.16.0
	 */
	MonikerProvider interface{}/* bool | MonikerOptions | MonikerRegistrationOptions*/ `json:"monikerProvider,omitempty"`
	/**
	 * The server provides type hierarchy support.
	 *
	 * @since 3.17.0 - proposed state
	 */
	TypeHierarchyProvider interface{}/* bool | TypeHierarchyOptions | TypeHierarchyRegistrationOptions*/ `json:"typeHierarchyProvider,omitempty"`
	/**
	 * The server provides inline values.
	 *
	 * @since 3.17.0 - proposed state
	 */
	InlineValuesProvider interface{}/* bool | InlineValuesOptions | InlineValuesOptions | InlineValuesRegistrationOptions*/ `json:"inlineValuesProvider,omitempty"`
	/**
	 * Experimental server capabilities.
	 */
	Experimental interface{} `json:"experimental,omitempty"`
}

type SetTraceParams struct {
	Value TraceValues `json:"value"`
}

/**
 * Client capabilities for the show document request.
 *
 * @since 3.16.0
 */
type ShowDocumentClientCapabilities struct {
	/**
	 * The client has support for the show document
	 * request.
	 */
	Support bool `json:"support"`
}

/**
 * Params to show a document.
 *
 * @since 3.16.0
 */
type ShowDocumentParams struct {
	/**
	 * The document uri to show.
	 */
	URI URI `json:"uri"`
	/**
	 * Indicates to show the resource in an external program.
	 * To show for example `https://code.visualstudio.com/`
	 * in the default WEB browser set `external` to `true`.
	 */
	External bool `json:"external,omitempty"`
	/**
	 * An optional property to indicate whether the editor
	 * showing the document should take focus or not.
	 * Clients might ignore this property if an external
	 * program in started.
	 */
	TakeFocus bool `json:"takeFocus,omitempty"`
	/**
	 * An optional selection range if the document is a text
	 * document. Clients might ignore the property if an
	 * external program is started or the file is not a text
	 * file.
	 */
	Selection Range `json:"selection,omitempty"`
}

/**
 * The result of an show document request.
 *
 * @since 3.16.0
 */
type ShowDocumentResult struct {
	/**
	 * A boolean indicating if the show was successful.
	 */
	Success bool `json:"success"`
}

/**
 * The parameters of a notification message.
 */
type ShowMessageParams struct {
	/**
	 * The message type. See {@link MessageType}
	 */
	Type MessageType `json:"type"`
	/**
	 * The actual message
	 */
	Message string `json:"message"`
}

/**
 * Show message request client capabilities
 */
type ShowMessageRequestClientCapabilities struct {
	/**
	 * Capabilities specific to the `MessageActionItem` type.
	 */
	MessageActionItem struct {
		/**
		 * Whether the client supports additional attributes which
		 * are preserved and send back to the server in the
		 * request's response.
		 */
		AdditionalPropertiesSupport bool `json:"additionalPropertiesSupport,omitempty"`
	} `json:"messageActionItem,omitempty"`
}

type ShowMessageRequestParams struct {
	/**
	 * The message type. See {@link MessageType}
	 */
	Type MessageType `json:"type"`
	/**
	 * The actual message
	 */
	Message string `json:"message"`
	/**
	 * The message action items to present.
	 */
	Actions []MessageActionItem `json:"actions,omitempty"`
}

/**
 * Signature help represents the signature of something
 * callable. There can be multiple signature but only one
 * active and only one active parameter.
 */
type SignatureHelp struct {
	/**
	 * One or more signatures.
	 */
	Signatures []SignatureInformation `json:"signatures"`
	/**
	 * The active signature. If omitted or the value lies outside the
	 * range of `signatures` the value defaults to zero or is ignored if
	 * the `SignatureHelp` has no signatures.
	 *
	 * Whenever possible implementors should make an active decision about
	 * the active signature and shouldn't rely on a default value.
	 *
	 * In future version of the protocol this property might become
	 * mandatory to better express this.
	 */
	ActiveSignature uint32 `json:"activeSignature,omitempty"`
	/**
	 * The active parameter of the active signature. If omitted or the value
	 * lies outside the range of `signatures[activeSignature].parameters`
	 * defaults to 0 if the active signature has parameters. If
	 * the active signature has no parameters it is ignored.
	 * In future version of the protocol this property might become
	 * mandatory to better express the active parameter if the
	 * active signature does have any.
	 */
	ActiveParameter uint32 `json:"activeParameter,omitempty"`
}

/**
 * Client Capabilities for a [SignatureHelpRequest](#SignatureHelpRequest).
 */
type SignatureHelpClientCapabilities struct {
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
		/**
		 * Client capabilities specific to parameter information.
		 */
		ParameterInformation struct {
			/**
			 * The client supports processing label offsets instead of a
			 * simple label string.
			 *
			 * @since 3.14.0
			 */
			LabelOffsetSupport bool `json:"labelOffsetSupport,omitempty"`
		} `json:"parameterInformation,omitempty"`
		/**
		 * The client support the `activeParameter` property on `SignatureInformation`
		 * literal.
		 *
		 * @since 3.16.0
		 */
		ActiveParameterSupport bool `json:"activeParameterSupport,omitempty"`
	} `json:"signatureInformation,omitempty"`
	/**
	 * The client supports to send additional context information for a
	 * `textDocument/signatureHelp` request. A client that opts into
	 * contextSupport will also support the `retriggerCharacters` on
	 * `SignatureHelpOptions`.
	 *
	 * @since 3.15.0
	 */
	ContextSupport bool `json:"contextSupport,omitempty"`
}

/**
 * Additional information about the context in which a signature help request was triggered.
 *
 * @since 3.15.0
 */
type SignatureHelpContext struct {
	/**
	 * Action that caused signature help to be triggered.
	 */
	TriggerKind SignatureHelpTriggerKind `json:"triggerKind"`
	/**
	 * Character that caused signature help to be triggered.
	 *
	 * This is undefined when `triggerKind !== SignatureHelpTriggerKind.TriggerCharacter`
	 */
	TriggerCharacter string `json:"triggerCharacter,omitempty"`
	/**
	 * `true` if signature help was already showing when it was triggered.
	 *
	 * Retrigger occurs when the signature help is already active and can be caused by actions such as
	 * typing a trigger character, a cursor move, or document content changes.
	 */
	IsRetrigger bool `json:"isRetrigger"`
	/**
	 * The currently active `SignatureHelp`.
	 *
	 * The `activeSignatureHelp` has its `SignatureHelp.activeSignature` field updated based on
	 * the user navigating through available signatures.
	 */
	ActiveSignatureHelp SignatureHelp `json:"activeSignatureHelp,omitempty"`
}

/**
 * Server Capabilities for a [SignatureHelpRequest](#SignatureHelpRequest).
 */
type SignatureHelpOptions struct {
	/**
	 * List of characters that trigger signature help.
	 */
	TriggerCharacters []string `json:"triggerCharacters,omitempty"`
	/**
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

/**
 * Parameters for a [SignatureHelpRequest](#SignatureHelpRequest).
 */
type SignatureHelpParams struct {
	/**
	 * The signature help context. This is only available if the client specifies
	 * to send this using the client capability `textDocument.signatureHelp.contextSupport === true`
	 *
	 * @since 3.15.0
	 */
	Context SignatureHelpContext `json:"context,omitempty"`
	TextDocumentPositionParams
	WorkDoneProgressParams
}

/**
 * How a signature help was triggered.
 *
 * @since 3.15.0
 */
type SignatureHelpTriggerKind float64

/**
 * Represents the signature of something callable. A signature
 * can have a label, like a function-name, a doc-comment, and
 * a set of parameters.
 */
type SignatureInformation struct {
	/**
	 * The label of this signature. Will be shown in
	 * the UI.
	 */
	Label string `json:"label"`
	/**
	 * The human-readable doc-comment of this signature. Will be shown
	 * in the UI but can be omitted.
	 */
	Documentation string/*string | MarkupContent*/ `json:"documentation,omitempty"`
	/**
	 * The parameters of this signature.
	 */
	Parameters []ParameterInformation `json:"parameters,omitempty"`
	/**
	 * The index of the active parameter.
	 *
	 * If provided, this is used in place of `SignatureHelp.activeParameter`.
	 *
	 * @since 3.16.0
	 */
	ActiveParameter uint32 `json:"activeParameter,omitempty"`
}

/**
 * Static registration options to be returned in the initialize
 * request.
 */
type StaticRegistrationOptions struct {
	/**
	 * The id used to register the request. The id can be used to deregister
	 * the request again. See also Registration#id.
	 */
	ID string `json:"id,omitempty"`
}

/**
 * Represents information about programming constructs like variables, classes,
 * interfaces etc.
 */
type SymbolInformation struct {
	/**
	 * The name of this symbol.
	 */
	Name string `json:"name"`
	/**
	 * The kind of this symbol.
	 */
	Kind SymbolKind `json:"kind"`
	/**
	 * Tags for this completion item.
	 *
	 * @since 3.16.0
	 */
	Tags []SymbolTag `json:"tags,omitempty"`
	/**
	 * Indicates if this symbol is deprecated.
	 *
	 * @deprecated Use tags instead
	 */
	Deprecated bool `json:"deprecated,omitempty"`
	/**
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
	/**
	 * The name of the symbol containing this symbol. This information is for
	 * user interface purposes (e.g. to render a qualifier in the user interface
	 * if necessary). It can't be used to re-infer a hierarchy for the document
	 * symbols.
	 */
	ContainerName string `json:"containerName,omitempty"`
}

/**
 * A symbol kind.
 */
type SymbolKind float64

/**
 * Symbol tags are extra annotations that tweak the rendering of a symbol.
 * @since 3.16
 */
type SymbolTag float64

/**
 * Text document specific client capabilities.
 */
type TextDocumentClientCapabilities struct {
	/**
	 * Defines which synchronization capabilities the client supports.
	 */
	Synchronization TextDocumentSyncClientCapabilities `json:"synchronization,omitempty"`
	/**
	 * Capabilities specific to the `textDocument/completion`
	 */
	Completion CompletionClientCapabilities `json:"completion,omitempty"`
	/**
	 * Capabilities specific to the `textDocument/hover`
	 */
	Hover HoverClientCapabilities `json:"hover,omitempty"`
	/**
	 * Capabilities specific to the `textDocument/signatureHelp`
	 */
	SignatureHelp SignatureHelpClientCapabilities `json:"signatureHelp,omitempty"`
	/**
	 * Capabilities specific to the `textDocument/declaration`
	 *
	 * @since 3.14.0
	 */
	Declaration DeclarationClientCapabilities `json:"declaration,omitempty"`
	/**
	 * Capabilities specific to the `textDocument/definition`
	 */
	Definition DefinitionClientCapabilities `json:"definition,omitempty"`
	/**
	 * Capabilities specific to the `textDocument/typeDefinition`
	 *
	 * @since 3.6.0
	 */
	TypeDefinition TypeDefinitionClientCapabilities `json:"typeDefinition,omitempty"`
	/**
	 * Capabilities specific to the `textDocument/implementation`
	 *
	 * @since 3.6.0
	 */
	Implementation ImplementationClientCapabilities `json:"implementation,omitempty"`
	/**
	 * Capabilities specific to the `textDocument/references`
	 */
	References ReferenceClientCapabilities `json:"references,omitempty"`
	/**
	 * Capabilities specific to the `textDocument/documentHighlight`
	 */
	DocumentHighlight DocumentHighlightClientCapabilities `json:"documentHighlight,omitempty"`
	/**
	 * Capabilities specific to the `textDocument/documentSymbol`
	 */
	DocumentSymbol DocumentSymbolClientCapabilities `json:"documentSymbol,omitempty"`
	/**
	 * Capabilities specific to the `textDocument/codeAction`
	 */
	CodeAction CodeActionClientCapabilities `json:"codeAction,omitempty"`
	/**
	 * Capabilities specific to the `textDocument/codeLens`
	 */
	CodeLens CodeLensClientCapabilities `json:"codeLens,omitempty"`
	/**
	 * Capabilities specific to the `textDocument/documentLink`
	 */
	DocumentLink DocumentLinkClientCapabilities `json:"documentLink,omitempty"`
	/**
	 * Capabilities specific to the `textDocument/documentColor`
	 */
	ColorProvider DocumentColorClientCapabilities `json:"colorProvider,omitempty"`
	/**
	 * Capabilities specific to the `textDocument/formatting`
	 */
	Formatting DocumentFormattingClientCapabilities `json:"formatting,omitempty"`
	/**
	 * Capabilities specific to the `textDocument/rangeFormatting`
	 */
	RangeFormatting DocumentRangeFormattingClientCapabilities `json:"rangeFormatting,omitempty"`
	/**
	 * Capabilities specific to the `textDocument/onTypeFormatting`
	 */
	OnTypeFormatting DocumentOnTypeFormattingClientCapabilities `json:"onTypeFormatting,omitempty"`
	/**
	 * Capabilities specific to the `textDocument/rename`
	 */
	Rename RenameClientCapabilities `json:"rename,omitempty"`
	/**
	 * Capabilities specific to `textDocument/foldingRange` request.
	 *
	 * @since 3.10.0
	 */
	FoldingRange FoldingRangeClientCapabilities `json:"foldingRange,omitempty"`
	/**
	 * Capabilities specific to `textDocument/selectionRange` request.
	 *
	 * @since 3.15.0
	 */
	SelectionRange SelectionRangeClientCapabilities `json:"selectionRange,omitempty"`
	/**
	 * Capabilities specific to `textDocument/publishDiagnostics` notification.
	 */
	PublishDiagnostics PublishDiagnosticsClientCapabilities `json:"publishDiagnostics,omitempty"`
	/**
	 * Capabilities specific to the various call hierarchy request.
	 *
	 * @since 3.16.0
	 */
	CallHierarchy CallHierarchyClientCapabilities `json:"callHierarchy,omitempty"`
	/**
	 * Capabilities specific to the various semantic token request.
	 *
	 * @since 3.16.0
	 */
	SemanticTokens SemanticTokensClientCapabilities `json:"semanticTokens,omitempty"`
	/**
	 * Capabilities specific to the linked editing range request.
	 *
	 * @since 3.16.0
	 */
	LinkedEditingRange LinkedEditingRangeClientCapabilities `json:"linkedEditingRange,omitempty"`
	/**
	 * Client capabilities specific to the moniker request.
	 *
	 * @since 3.16.0
	 */
	Moniker MonikerClientCapabilities `json:"moniker,omitempty"`
	/**
	 * Capabilities specific to the various type hierarchy requests.
	 *
	 * @since 3.17.0 - proposed state
	 */
	TypeHierarchy TypeHierarchyClientCapabilities `json:"typeHierarchy,omitempty"`
	/**
	 * Capabilities specific to the `textDocument/inlineValues` request.
	 *
	 * @since 3.17.0 - proposed state
	 */
	InlineValues InlineValuesClientCapabilities `json:"inlineValues,omitempty"`
}

/**
 * An event describing a change to a text document. If range and rangeLength are omitted
 * the new text is considered to be the full content of the document.
 */
type TextDocumentContentChangeEvent = struct {
	/**
	 * The range of the document that changed.
	 */
	Range *Range `json:"range,omitempty"`
	/**
	 * The optional length of the range that got replaced.
	 *
	 * @deprecated use range instead.
	 */
	RangeLength uint32 `json:"rangeLength,omitempty"`
	/**
	 * The new text for the provided range.
	 */
	Text string `json:"text"`
}

/**
 * Describes textual changes on a text document. A TextDocumentEdit describes all changes
 * on a document version Si and after they are applied move the document to version Si+1.
 * So the creator of a TextDocumentEdit doesn't need to sort the array of edits or do any
 * kind of ordering. However the edits must be non overlapping.
 */
type TextDocumentEdit struct {
	/**
	 * The text document to change.
	 */
	TextDocument OptionalVersionedTextDocumentIdentifier `json:"textDocument"`
	/**
	 * The edits to be applied.
	 *
	 * @since 3.16.0 - support for AnnotatedTextEdit. This is guarded using a
	 * client capability.
	 */
	Edits []TextEdit/*TextEdit | AnnotatedTextEdit*/ `json:"edits"`
}

/**
 * A literal to identify a text document in the client.
 */
type TextDocumentIdentifier struct {
	/**
	 * The text document's uri.
	 */
	URI DocumentURI `json:"uri"`
}

/**
 * An item to transfer a text document from the client to the
 * server.
 */
type TextDocumentItem struct {
	/**
	 * The text document's uri.
	 */
	URI DocumentURI `json:"uri"`
	/**
	 * The text document's language identifier
	 */
	LanguageID string `json:"languageId"`
	/**
	 * The version number of this document (it will increase after each
	 * change, including undo/redo).
	 */
	Version int32 `json:"version"`
	/**
	 * The content of the opened text document.
	 */
	Text string `json:"text"`
}

/**
 * A parameter literal used in requests to pass a text document and a position inside that
 * document.
 */
type TextDocumentPositionParams struct {
	/**
	 * The text document.
	 */
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	/**
	 * The position inside the text document.
	 */
	Position Position `json:"position"`
}

/**
 * General text document registration options.
 */
type TextDocumentRegistrationOptions struct {
	/**
	 * A document selector to identify the scope of the registration. If set to null
	 * the document selector provided on the client side will be used.
	 */
	DocumentSelector DocumentSelector /*DocumentSelector | null*/ `json:"documentSelector"`
}

/**
 * Represents reasons why a text document is saved.
 */
type TextDocumentSaveReason float64

type TextDocumentSyncClientCapabilities struct {
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
}

/**
 * Defines how the host (editor) should sync
 * document changes to the language server.
 */
type TextDocumentSyncKind float64

type TextDocumentSyncOptions struct {
	/**
	 * Open and close notifications are sent to the server. If omitted open close notification should not
	 * be sent.
	 */
	OpenClose bool `json:"openClose,omitempty"`
	/**
	 * Change notifications are sent to the server. See TextDocumentSyncKind.None, TextDocumentSyncKind.Full
	 * and TextDocumentSyncKind.Incremental. If omitted it defaults to TextDocumentSyncKind.None.
	 */
	Change TextDocumentSyncKind `json:"change,omitempty"`
	/**
	 * If present will save notifications are sent to the server. If omitted the notification should not be
	 * sent.
	 */
	WillSave bool `json:"willSave,omitempty"`
	/**
	 * If present will save wait until requests are sent to the server. If omitted the request should not be
	 * sent.
	 */
	WillSaveWaitUntil bool `json:"willSaveWaitUntil,omitempty"`
	/**
	 * If present save notifications are sent to the server. If omitted the notification should not be
	 * sent.
	 */
	Save SaveOptions/*boolean | SaveOptions*/ `json:"save,omitempty"`
}

/**
 * A text edit applicable to a text document.
 */
type TextEdit struct {
	/**
	 * The range of the text document to be manipulated. To insert
	 * text into a document create a range where start === end.
	 */
	Range Range `json:"range"`
	/**
	 * The string to be inserted. For delete operations use an
	 * empty string.
	 */
	NewText string `json:"newText"`
}

type TokenFormat = string

type TraceValues = string /* 'off' | 'messages' | 'compact' | 'verbose' */

/**
 * Since 3.6.0
 */
type TypeDefinitionClientCapabilities struct {
	/**
	 * Whether implementation supports dynamic registration. If this is set to `true`
	 * the client supports the new `TypeDefinitionRegistrationOptions` return value
	 * for the corresponding server capability as well.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
	/**
	 * The client supports additional metadata in the form of definition links.
	 *
	 * Since 3.14.0
	 */
	LinkSupport bool `json:"linkSupport,omitempty"`
}

type TypeDefinitionOptions struct {
	WorkDoneProgressOptions
}

type TypeDefinitionParams struct {
	TextDocumentPositionParams
	WorkDoneProgressParams
	PartialResultParams
}

type TypeDefinitionRegistrationOptions struct {
	TextDocumentRegistrationOptions
	TypeDefinitionOptions
	StaticRegistrationOptions
}

/**
 * @since 3.17.0 - proposed state
 */
type TypeHierarchyClientCapabilities struct {
	/**
	 * Whether implementation supports dynamic registration. If this is set to `true`
	 * the client supports the new `(TextDocumentRegistrationOptions & StaticRegistrationOptions)`
	 * return value for the corresponding server capability as well.
	 */
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
}

/**
 * @since 3.17.0 - proposed state
 */
type TypeHierarchyItem struct {
	/**
	 * The name of this item.
	 */
	Name string `json:"name"`
	/**
	 * The kind of this item.
	 */
	Kind SymbolKind `json:"kind"`
	/**
	 * Tags for this item.
	 */
	Tags []SymbolTag `json:"tags,omitempty"`
	/**
	 * More detail for this item, e.g. the signature of a function.
	 */
	Detail string `json:"detail,omitempty"`
	/**
	 * The resource identifier of this item.
	 */
	URI DocumentURI `json:"uri"`
	/**
	 * The range enclosing this symbol not including leading/trailing whitespace
	 * but everything else, e.g. comments and code.
	 */
	Range Range `json:"range"`
	/**
	 * The range that should be selected and revealed when this symbol is being
	 * picked, e.g. the name of a function. Must be contained by the
	 * [`range`](#TypeHierarchyItem.range).
	 */
	SelectionRange Range `json:"selectionRange"`
	/**
	 * A data entry field that is preserved between a type hierarchy prepare and
	 * supertypes or subtypes requests. It could also be used to identify the
	 * type hierarchy in the server, helping improve the performance on
	 * resolving supertypes and subtypes.
	 */
	Data LSPAny `json:"data,omitempty"`
}

/**
 * Type hierarchy options used during static registration.
 *
 * @since 3.17.0 - proposed state
 */
type TypeHierarchyOptions struct {
	WorkDoneProgressOptions
}

/**
 * The parameter of a `textDocument/prepareTypeHierarchy` request.
 *
 * @since 3.17.0 - proposed state
 */
type TypeHierarchyPrepareParams struct {
	TextDocumentPositionParams
	WorkDoneProgressParams
}

/**
 * Type hierarchy options used during static or dynamic registration.
 *
 * @since 3.17.0 - proposed state
 */
type TypeHierarchyRegistrationOptions struct {
	TextDocumentRegistrationOptions
	TypeHierarchyOptions
	StaticRegistrationOptions
}

/**
 * The parameter of a `typeHierarchy/subtypes` request.
 *
 * @since 3.17.0 - proposed state
 */
type TypeHierarchySubtypesParams struct {
	Item TypeHierarchyItem `json:"item"`
	WorkDoneProgressParams
	PartialResultParams
}

/**
 * The parameter of a `typeHierarchy/supertypes` request.
 *
 * @since 3.17.0 - proposed state
 */
type TypeHierarchySupertypesParams struct {
	Item TypeHierarchyItem `json:"item"`
	WorkDoneProgressParams
	PartialResultParams
}

/**
 * A tagging type for string properties that are actually URIs
 *
 * @since 3.16.0
 */
type URI = string

/**
 * A diagnostic report indicating that the last returned
 * report is still accurate.
 *
 * @since 3.17.0 - proposed state
 */
type UnchangedDocumentDiagnosticReport struct {
	/**
	 * A document diagnostic report indicating
	 * no changes to the last result. A server can
	 * only return `unchanged` if result ids are
	 * provided.
	 */
	Kind string `json:"kind"`
	/**
	 * A result id which will be sent on the next
	 * diagnostic request for the same document.
	 */
	ResultID string `json:"resultId"`
}

/**
 * Moniker uniqueness level to define scope of the moniker.
 *
 * @since 3.16.0
 */
type UniquenessLevel string

/**
 * General parameters to unregister a request or notification.
 */
type Unregistration struct {
	/**
	 * The id used to unregister the request or notification. Usually an id
	 * provided during the register request.
	 */
	ID string `json:"id"`
	/**
	 * The method to unregister for.
	 */
	Method string `json:"method"`
}

type UnregistrationParams struct {
	Unregisterations []Unregistration `json:"unregisterations"`
}

/**
 * A text document identifier to denote a specific version of a text document.
 */
type VersionedTextDocumentIdentifier struct {
	/**
	 * The version number of this document.
	 */
	Version int32 `json:"version"`
	TextDocumentIdentifier
}

type WatchKind float64

/**
 * The parameters send in a will save text document notification.
 */
type WillSaveTextDocumentParams struct {
	/**
	 * The document that will be saved.
	 */
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	/**
	 * The 'TextDocumentSaveReason'.
	 */
	Reason TextDocumentSaveReason `json:"reason"`
}

type WorkDoneProgressBegin struct {
	Kind string `json:"kind"`
	/**
	 * Mandatory title of the progress operation. Used to briefly inform about
	 * the kind of operation being performed.
	 *
	 * Examples: "Indexing" or "Linking dependencies".
	 */
	Title string `json:"title"`
	/**
	 * Controls if a cancel button should show to allow the user to cancel the
	 * long running operation. Clients that don't support cancellation are allowed
	 * to ignore the setting.
	 */
	Cancellable bool `json:"cancellable,omitempty"`
	/**
	 * Optional, more detailed associated progress message. Contains
	 * complementary information to the `title`.
	 *
	 * Examples: "3/25 files", "project/src/module2", "node_modules/some_dep".
	 * If unset, the previous progress message (if any) is still valid.
	 */
	Message string `json:"message,omitempty"`
	/**
	 * Optional progress percentage to display (value 100 is considered 100%).
	 * If not provided infinite progress is assumed and clients are allowed
	 * to ignore the `percentage` value in subsequent in report notifications.
	 *
	 * The value should be steadily rising. Clients are free to ignore values
	 * that are not following this rule. The value range is [0, 100].
	 */
	Percentage uint32 `json:"percentage,omitempty"`
}

type WorkDoneProgressCancelParams struct {
	/**
	 * The token to be used to report progress.
	 */
	Token ProgressToken `json:"token"`
}

type WorkDoneProgressClientCapabilities struct {
	/**
	 * Window specific client capabilities.
	 */
	Window struct {
		/**
		 * Whether client supports server initiated progress using the
		 * `window/workDoneProgress/create` request.
		 *
		 * Since 3.15.0
		 */
		WorkDoneProgress bool `json:"workDoneProgress,omitempty"`
		/**
		 * Capabilities specific to the showMessage request.
		 *
		 * @since 3.16.0
		 */
		ShowMessage ShowMessageRequestClientCapabilities `json:"showMessage,omitempty"`
		/**
		 * Capabilities specific to the showDocument request.
		 *
		 * @since 3.16.0
		 */
		ShowDocument ShowDocumentClientCapabilities `json:"showDocument,omitempty"`
	} `json:"window,omitempty"`
}

type WorkDoneProgressCreateParams struct {
	/**
	 * The token to be used to report progress.
	 */
	Token ProgressToken `json:"token"`
}

type WorkDoneProgressEnd struct {
	Kind string `json:"kind"`
	/**
	 * Optional, a final message indicating to for example indicate the outcome
	 * of the operation.
	 */
	Message string `json:"message,omitempty"`
}

type WorkDoneProgressOptions struct {
	WorkDoneProgress bool `json:"workDoneProgress,omitempty"`
}

type WorkDoneProgressParams struct {
	/**
	 * An optional token that a server can use to report work done progress.
	 */
	WorkDoneToken ProgressToken `json:"workDoneToken,omitempty"`
}

type WorkDoneProgressReport struct {
	Kind string `json:"kind"`
	/**
	 * Controls enablement state of a cancel button.
	 *
	 * Clients that don't support cancellation or don't support controlling the button's
	 * enablement state are allowed to ignore the property.
	 */
	Cancellable bool `json:"cancellable,omitempty"`
	/**
	 * Optional, more detailed associated progress message. Contains
	 * complementary information to the `title`.
	 *
	 * Examples: "3/25 files", "project/src/module2", "node_modules/some_dep".
	 * If unset, the previous progress message (if any) is still valid.
	 */
	Message string `json:"message,omitempty"`
	/**
	 * Optional progress percentage to display (value 100 is considered 100%).
	 * If not provided infinite progress is assumed and clients are allowed
	 * to ignore the `percentage` value in subsequent in report notifications.
	 *
	 * The value should be steadily rising. Clients are free to ignore values
	 * that are not following this rule. The value range is [0, 100]
	 */
	Percentage uint32 `json:"percentage,omitempty"`
}

/**
 * Workspace specific client capabilities.
 */
type WorkspaceClientCapabilities struct {
	/**
	 * The client supports applying batch edits
	 * to the workspace by supporting the request
	 * 'workspace/applyEdit'
	 */
	ApplyEdit bool `json:"applyEdit,omitempty"`
	/**
	 * Capabilities specific to `WorkspaceEdit`s
	 */
	WorkspaceEdit WorkspaceEditClientCapabilities `json:"workspaceEdit,omitempty"`
	/**
	 * Capabilities specific to the `workspace/didChangeConfiguration` notification.
	 */
	DidChangeConfiguration DidChangeConfigurationClientCapabilities `json:"didChangeConfiguration,omitempty"`
	/**
	 * Capabilities specific to the `workspace/didChangeWatchedFiles` notification.
	 */
	DidChangeWatchedFiles DidChangeWatchedFilesClientCapabilities `json:"didChangeWatchedFiles,omitempty"`
	/**
	 * Capabilities specific to the `workspace/symbol` request.
	 */
	Symbol WorkspaceSymbolClientCapabilities `json:"symbol,omitempty"`
	/**
	 * Capabilities specific to the `workspace/executeCommand` request.
	 */
	ExecuteCommand ExecuteCommandClientCapabilities `json:"executeCommand,omitempty"`
	/**
	 * Capabilities specific to the semantic token requests scoped to the
	 * workspace.
	 *
	 * @since 3.16.0.
	 */
	SemanticTokens SemanticTokensWorkspaceClientCapabilities `json:"semanticTokens,omitempty"`
	/**
	 * Capabilities specific to the code lens requests scoped to the
	 * workspace.
	 *
	 * @since 3.16.0.
	 */
	CodeLens CodeLensWorkspaceClientCapabilities `json:"codeLens,omitempty"`
	/**
	 * The client has support for file notifications/requests for user operations on files.
	 *
	 * Since 3.16.0
	 */
	FileOperations FileOperationClientCapabilities `json:"fileOperations,omitempty"`
	/**
	 * Capabilities specific to the inline values requests scoped to the
	 * workspace.
	 *
	 * @since 3.17.0.
	 */
	InlineValues InlineValuesWorkspaceClientCapabilities `json:"inlineValues,omitempty"`
}

/**
 * Parameters of the workspace diagnostic request.
 *
 * @since 3.17.0 - proposed state
 */
type WorkspaceDiagnosticParams struct {
	/**
	 * The additional identifier provided during registration.
	 */
	Identifier string `json:"identifier,omitempty"`
	/**
	 * The currently known diagnostic reports with their
	 * previous result ids.
	 */
	PreviousResultIds []PreviousResultID `json:"previousResultIds"`
	WorkDoneProgressParams
	PartialResultParams
}

/**
 * A workspace diagnostic report.
 *
 * @since 3.17.0 - proposed state
 */
type WorkspaceDiagnosticReport struct {
	Items []WorkspaceDocumentDiagnosticReport `json:"items"`
}

/**
 * A workspace diagnostic document report.
 *
 * @since 3.17.0 - proposed state
 */
type WorkspaceDocumentDiagnosticReport = interface{} /*WorkspaceFullDocumentDiagnosticReport | WorkspaceUnchangedDocumentDiagnosticReport*/

/**
 * A workspace edit represents changes to many resources managed in the workspace. The edit
 * should either provide `changes` or `documentChanges`. If documentChanges are present
 * they are preferred over `changes` if the client can handle versioned document edits.
 *
 * Since version 3.13.0 a workspace edit can contain resource operations as well. If resource
 * operations are present clients need to execute the operations in the order in which they
 * are provided. So a workspace edit for example can consist of the following two changes:
 * (1) a create file a.txt and (2) a text document edit which insert text into file a.txt.
 *
 * An invalid sequence (e.g. (1) delete file a.txt and (2) insert text into file a.txt) will
 * cause failure of the operation. How the client recovers from the failure is described by
 * the client capability: `workspace.workspaceEdit.failureHandling`
 */
type WorkspaceEdit struct {
	/**
	 * Holds changes to existing resources.
	 */
	Changes map[DocumentURI][]TextEdit/*[uri: DocumentUri]: TextEdit[];*/ `json:"changes,omitempty"`
	/**
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
	DocumentChanges []TextDocumentEdit/*TextDocumentEdit | CreateFile | RenameFile | DeleteFile*/ `json:"documentChanges,omitempty"`
	/**
	 * A map of change annotations that can be referenced in `AnnotatedTextEdit`s or create, rename and
	 * delete file / folder operations.
	 *
	 * Whether clients honor this property depends on the client capability `workspace.changeAnnotationSupport`.
	 *
	 * @since 3.16.0
	 */
	ChangeAnnotations map[string]ChangeAnnotationIdentifier/*[id: ChangeAnnotationIdentifier]: ChangeAnnotation;*/ `json:"changeAnnotations,omitempty"`
}

type WorkspaceEditClientCapabilities struct {
	/**
	 * The client supports versioned document changes in `WorkspaceEdit`s
	 */
	DocumentChanges bool `json:"documentChanges,omitempty"`
	/**
	 * The resource operations the client supports. Clients should at least
	 * support 'create', 'rename' and 'delete' files and folders.
	 *
	 * @since 3.13.0
	 */
	ResourceOperations []ResourceOperationKind `json:"resourceOperations,omitempty"`
	/**
	 * The failure handling strategy of a client if applying the workspace edit
	 * fails.
	 *
	 * @since 3.13.0
	 */
	FailureHandling FailureHandlingKind `json:"failureHandling,omitempty"`
	/**
	 * Whether the client normalizes line endings to the client specific
	 * setting.
	 * If set to `true` the client will normalize line ending characters
	 * in a workspace edit containing to the client specific new line
	 * character.
	 *
	 * @since 3.16.0
	 */
	NormalizesLineEndings bool `json:"normalizesLineEndings,omitempty"`
	/**
	 * Whether the client in general supports change annotations on text edits,
	 * create file, rename file and delete file changes.
	 *
	 * @since 3.16.0
	 */
	ChangeAnnotationSupport struct {
		/**
		 * Whether the client groups edits with equal labels into tree nodes,
		 * for instance all edits labelled with "Changes in Strings" would
		 * be a tree node.
		 */
		GroupsOnLabel bool `json:"groupsOnLabel,omitempty"`
	} `json:"changeAnnotationSupport,omitempty"`
}

type WorkspaceFolder struct {
	/**
	 * The associated URI for this workspace folder.
	 */
	URI string `json:"uri"`
	/**
	 * The name of the workspace folder. Used to refer to this
	 * workspace folder in the user interface.
	 */
	Name string `json:"name"`
}

/**
 * The workspace folder change event.
 */
type WorkspaceFoldersChangeEvent struct {
	/**
	 * The array of added workspace folders
	 */
	Added []WorkspaceFolder `json:"added"`
	/**
	 * The array of the removed workspace folders
	 */
	Removed []WorkspaceFolder `json:"removed"`
}

type WorkspaceFoldersClientCapabilities struct {
	/**
	 * The workspace client capabilities
	 */
	Workspace Workspace6Gn `json:"workspace,omitempty"`
}

type WorkspaceFoldersInitializeParams struct {
	/**
	 * The actual configured workspace folders.
	 */
	WorkspaceFolders []WorkspaceFolder /*WorkspaceFolder[] | null*/ `json:"workspaceFolders"`
}

type WorkspaceFoldersServerCapabilities struct {
	/**
	 * The workspace server capabilities
	 */
	Workspace Workspace8Gn `json:"workspace,omitempty"`
}

/**
 * A full document diagnostic report for a workspace diagnostic result.
 *
 * @since 3.17.0 - proposed state
 */
type WorkspaceFullDocumentDiagnosticReport struct {
	/**
	 * The URI for which diagnostic information is reported.
	 */
	URI DocumentURI `json:"uri"`
	/**
	 * The version number for which the diagnostics are reported.
	 * If the document is not marked as open `null` can be provided.
	 */
	Version int32/*integer | null*/ `json:"version"`
	FullDocumentDiagnosticReport
}

/**
 * A special workspace symbol that supports locations without a range
 *
 * @since 3.17.0 - proposed state
 */
type WorkspaceSymbol struct {
	/**
	 * The location of the symbol.
	 *
	 * See SymbolInformation#location for more details.
	 */
	Location Location/*Location | { uri: DocumentUri; }*/ `json:"location"`
	/**
	 * A data entry field that is preserved on a workspace symbol between a
	 * workspace symbol request and a workspace symbol resolve request.
	 */
	Data LSPAny `json:"data,omitempty"`
}

/**
 * Client capabilities for a [WorkspaceSymbolRequest](#WorkspaceSymbolRequest).
 */
type WorkspaceSymbolClientCapabilities struct {
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
	/**
	 * The client supports tags on `SymbolInformation`.
	 * Clients supporting tags have to handle unknown tags gracefully.
	 *
	 * @since 3.16.0
	 */
	TagSupport struct {
		/**
		 * The tags supported by the client.
		 */
		ValueSet []SymbolTag `json:"valueSet"`
	} `json:"tagSupport,omitempty"`
	/**
	 * The client support partial workspace symbols. The client will send the
	 * request `workspaceSymbol/resolve` to the server to resolve additional
	 * properties.
	 *
	 * @since 3.17.0 - proposedState
	 */
	ResolveSupport struct {
		/**
		 * The properties that a client can resolve lazily. Usually
		 * `location.range`
		 */
		Properties []string `json:"properties"`
	} `json:"resolveSupport,omitempty"`
}

/**
 * Server capabilities for a [WorkspaceSymbolRequest](#WorkspaceSymbolRequest).
 */
type WorkspaceSymbolOptions struct {
	/**
	 * The server provides support to resolve additional
	 * information for a workspace symbol.
	 *
	 * @since 3.17.0 - proposed state
	 */
	ResolveProvider bool `json:"resolveProvider,omitempty"`
	WorkDoneProgressOptions
}

/**
 * The parameters of a [WorkspaceSymbolRequest](#WorkspaceSymbolRequest).
 */
type WorkspaceSymbolParams struct {
	/**
	 * A query string to filter symbols by. Clients may send an empty
	 * string here to request all symbols.
	 */
	Query string `json:"query"`
	WorkDoneProgressParams
	PartialResultParams
}

/**
 * An unchanged document diagnostic report for a workspace diagnostic result.
 *
 * @since 3.17.0 - proposed state
 */
type WorkspaceUnchangedDocumentDiagnosticReport struct {
	/**
	 * The URI for which diagnostic information is reported.
	 */
	URI DocumentURI `json:"uri"`
	/**
	 * The version number for which the diagnostics are reported.
	 * If the document is not marked as open `null` can be provided.
	 */
	Version int32/*integer | null*/ `json:"version"`
	UnchangedDocumentDiagnosticReport
}

const (
	/**
	 * Empty kind.
	 */

	Empty CodeActionKind = ""
	/**
	 * Base kind for quickfix actions: 'quickfix'
	 */

	QuickFix CodeActionKind = "quickfix"
	/**
	 * Base kind for refactoring actions: 'refactor'
	 */

	Refactor CodeActionKind = "refactor"
	/**
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
	/**
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
	/**
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
	/**
	 * Base kind for source actions: `source`
	 *
	 * Source code actions apply to the entire file.
	 */

	Source CodeActionKind = "source"
	/**
	 * Base kind for an organize imports source action: `source.organizeImports`
	 */

	SourceOrganizeImports CodeActionKind = "source.organizeImports"
	/**
	 * Base kind for auto-fix source actions: `source.fixAll`.
	 *
	 * Fix all actions automatically fix errors that have a clear fix that do not require user input.
	 * They should not suppress errors or perform unsafe fixes such as generating new types or classes.
	 *
	 * @since 3.15.0
	 */

	SourceFixAll CodeActionKind = "source.fixAll"
	/**
	 * Code actions were explicitly requested by the user or by an extension.
	 */

	CodeActionInvoked CodeActionTriggerKind = 1
	/**
	 * Code actions were requested automatically.
	 *
	 * This typically happens when current selection in a file changes, but can
	 * also be triggered when file content changes.
	 */

	CodeActionAutomatic     CodeActionTriggerKind = 2
	TextCompletion          CompletionItemKind    = 1
	MethodCompletion        CompletionItemKind    = 2
	FunctionCompletion      CompletionItemKind    = 3
	ConstructorCompletion   CompletionItemKind    = 4
	FieldCompletion         CompletionItemKind    = 5
	VariableCompletion      CompletionItemKind    = 6
	ClassCompletion         CompletionItemKind    = 7
	InterfaceCompletion     CompletionItemKind    = 8
	ModuleCompletion        CompletionItemKind    = 9
	PropertyCompletion      CompletionItemKind    = 10
	UnitCompletion          CompletionItemKind    = 11
	ValueCompletion         CompletionItemKind    = 12
	EnumCompletion          CompletionItemKind    = 13
	KeywordCompletion       CompletionItemKind    = 14
	SnippetCompletion       CompletionItemKind    = 15
	ColorCompletion         CompletionItemKind    = 16
	FileCompletion          CompletionItemKind    = 17
	ReferenceCompletion     CompletionItemKind    = 18
	FolderCompletion        CompletionItemKind    = 19
	EnumMemberCompletion    CompletionItemKind    = 20
	ConstantCompletion      CompletionItemKind    = 21
	StructCompletion        CompletionItemKind    = 22
	EventCompletion         CompletionItemKind    = 23
	OperatorCompletion      CompletionItemKind    = 24
	TypeParameterCompletion CompletionItemKind    = 25
	/**
	 * Render a completion as obsolete, usually using a strike-out.
	 */

	ComplDeprecated CompletionItemTag = 1
	/**
	 * Completion was triggered by typing an identifier (24x7 code
	 * complete), manual invocation (e.g Ctrl+Space) or via API.
	 */

	Invoked CompletionTriggerKind = 1
	/**
	 * Completion was triggered by a trigger character specified by
	 * the `triggerCharacters` properties of the `CompletionRegistrationOptions`.
	 */

	TriggerCharacter CompletionTriggerKind = 2
	/**
	 * Completion was re-triggered as current completion list is incomplete
	 */

	TriggerForIncompleteCompletions CompletionTriggerKind = 3
	/**
	 * Reports an error.
	 */

	SeverityError DiagnosticSeverity = 1
	/**
	 * Reports a warning.
	 */

	SeverityWarning DiagnosticSeverity = 2
	/**
	 * Reports an information.
	 */

	SeverityInformation DiagnosticSeverity = 3
	/**
	 * Reports a hint.
	 */

	SeverityHint DiagnosticSeverity = 4
	/**
	 * Unused or unnecessary code.
	 *
	 * Clients are allowed to render diagnostics with this tag faded out instead of having
	 * an error squiggle.
	 */

	Unnecessary DiagnosticTag = 1
	/**
	 * Deprecated or obsolete code.
	 *
	 * Clients are allowed to rendered diagnostics with this tag strike through.
	 */

	Deprecated DiagnosticTag = 2
	/**
	 * A textual occurrence.
	 */

	Text DocumentHighlightKind = 1
	/**
	 * Read-access of a symbol, like reading a variable.
	 */

	Read DocumentHighlightKind = 2
	/**
	 * Write-access of a symbol, like writing to a variable.
	 */

	Write DocumentHighlightKind = 3
	/**
	 * Applying the workspace change is simply aborted if one of the changes provided
	 * fails. All operations executed before the failing operation stay executed.
	 */

	Abort FailureHandlingKind = "abort"
	/**
	 * All operations are executed transactional. That means they either all
	 * succeed or no changes at all are applied to the workspace.
	 */

	Transactional FailureHandlingKind = "transactional"
	/**
	 * If the workspace edit contains only textual file changes they are executed transactional.
	 * If resource changes (create, rename or delete file) are part of the change the failure
	 * handling strategy is abort.
	 */

	TextOnlyTransactional FailureHandlingKind = "textOnlyTransactional"
	/**
	 * The client tries to undo the operations already executed. But there is no
	 * guarantee that this is succeeding.
	 */

	Undo FailureHandlingKind = "undo"
	/**
	 * The file got created.
	 */

	Created FileChangeType = 1
	/**
	 * The file got changed.
	 */

	Changed FileChangeType = 2
	/**
	 * The file got deleted.
	 */

	Deleted FileChangeType = 3
	/**
	 * The pattern matches a file only.
	 */

	FileOp FileOperationPatternKind = "file"
	/**
	 * The pattern matches a folder only.
	 */

	FolderOp FileOperationPatternKind = "folder"
	/**
	 * Folding range for a comment
	 */
	Comment FoldingRangeKind = "comment"
	/**
	 * Folding range for a imports or includes
	 */
	Imports FoldingRangeKind = "imports"
	/**
	 * Folding range for a region (e.g. `#region`)
	 */
	Region FoldingRangeKind = "region"
	/**
	 * If the protocol version provided by the client can't be handled by the server.
	 * @deprecated This initialize error got replaced by client capabilities. There is
	 * no version handshake in version 3.0x
	 */

	UnknownProtocolVersion InitializeError = 1
	/**
	 * The primary text to be inserted is treated as a plain string.
	 */

	PlainTextTextFormat InsertTextFormat = 1
	/**
	 * The primary text to be inserted is treated as a snippet.
	 *
	 * A snippet can define tab stops and placeholders with `$1`, `$2`
	 * and `${3:foo}`. `$0` defines the final tab stop, it defaults to
	 * the end of the snippet. Placeholders with equal identifiers are linked,
	 * that is typing in one will update others too.
	 *
	 * See also: https://microsoft.github.io/language-server-protocol/specifications/specification-current/#snippet_syntax
	 */

	SnippetTextFormat InsertTextFormat = 2
	/**
	 * The insertion or replace strings is taken as it is. If the
	 * value is multi line the lines below the cursor will be
	 * inserted using the indentation defined in the string value.
	 * The client will not apply any kind of adjustments to the
	 * string.
	 */

	AsIs InsertTextMode = 1
	/**
	 * The editor adjusts leading whitespace of new lines so that
	 * they match the indentation up to the cursor of the line for
	 * which the item is accepted.
	 *
	 * Consider a line like this: <2tabs><cursor><3tabs>foo. Accepting a
	 * multi line completion item is indented using 2 tabs and all
	 * following lines inserted will be indented using 2 tabs as well.
	 */

	AdjustIndentation InsertTextMode = 2
	/**
	 * Plain text is supported as a content format
	 */

	PlainText MarkupKind = "plaintext"
	/**
	 * Markdown is supported as a content format
	 */

	Markdown MarkupKind = "markdown"
	/**
	 * An error message.
	 */

	Error MessageType = 1
	/**
	 * A warning message.
	 */

	Warning MessageType = 2
	/**
	 * An information message.
	 */

	Info MessageType = 3
	/**
	 * A log message.
	 */

	Log MessageType = 4
	/**
	 * The moniker represent a symbol that is imported into a project
	 */
	Import MonikerKind = "import"
	/**
	 * The moniker represents a symbol that is exported from a project
	 */
	Export MonikerKind = "export"
	/**
	 * The moniker represents a symbol that is local to a project (e.g. a local
	 * variable of a function, a class not visible outside the project, ...)
	 */
	Local MonikerKind = "local"
	/**
	 * Supports creating new files and folders.
	 */

	Create ResourceOperationKind = "create"
	/**
	 * Supports renaming existing files and folders.
	 */

	Rename ResourceOperationKind = "rename"
	/**
	 * Supports deleting existing files and folders.
	 */

	Delete ResourceOperationKind = "delete"
	/**
	 * Signature help was invoked manually by the user or by a command.
	 */

	SigInvoked SignatureHelpTriggerKind = 1
	/**
	 * Signature help was triggered by a trigger character.
	 */

	SigTriggerCharacter SignatureHelpTriggerKind = 2
	/**
	 * Signature help was triggered by the cursor moving or by the document content changing.
	 */

	SigContentChange SignatureHelpTriggerKind = 3
	File             SymbolKind               = 1
	Module           SymbolKind               = 2
	Namespace        SymbolKind               = 3
	Package          SymbolKind               = 4
	Class            SymbolKind               = 5
	Method           SymbolKind               = 6
	Property         SymbolKind               = 7
	Field            SymbolKind               = 8
	Constructor      SymbolKind               = 9
	Enum             SymbolKind               = 10
	Interface        SymbolKind               = 11
	Function         SymbolKind               = 12
	Variable         SymbolKind               = 13
	Constant         SymbolKind               = 14
	String           SymbolKind               = 15
	Number           SymbolKind               = 16
	Boolean          SymbolKind               = 17
	Array            SymbolKind               = 18
	Object           SymbolKind               = 19
	Key              SymbolKind               = 20
	Null             SymbolKind               = 21
	EnumMember       SymbolKind               = 22
	Struct           SymbolKind               = 23
	Event            SymbolKind               = 24
	Operator         SymbolKind               = 25
	TypeParameter    SymbolKind               = 26
	/**
	 * Render a symbol as obsolete, usually using a strike-out.
	 */

	DeprecatedSymbol SymbolTag = 1
	/**
	 * Manually triggered, e.g. by the user pressing save, by starting debugging,
	 * or by an API call.
	 */

	Manual TextDocumentSaveReason = 1
	/**
	 * Automatic after a delay.
	 */

	AfterDelay TextDocumentSaveReason = 2
	/**
	 * When the editor lost focus.
	 */

	FocusOut TextDocumentSaveReason = 3
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
	/**
	 * The moniker is only unique inside a document
	 */
	Document UniquenessLevel = "document"
	/**
	 * The moniker is unique inside a project for which a dump got created
	 */
	Project UniquenessLevel = "project"
	/**
	 * The moniker is unique inside the group to which a project belongs
	 */
	Group UniquenessLevel = "group"
	/**
	 * The moniker is unique inside the moniker scheme.
	 */
	Scheme UniquenessLevel = "scheme"
	/**
	 * The moniker is globally unique
	 */
	Global UniquenessLevel = "global"
	/**
	 * Interested in create events.
	 */

	WatchCreate WatchKind = 1
	/**
	 * Interested in change events
	 */

	WatchChange WatchKind = 2
	/**
	 * Interested in delete events
	 */

	WatchDelete WatchKind = 4
)

// Types created to name formal parameters and embedded structs
type ParamConfiguration struct {
	ConfigurationParams
	PartialResultParams
}
type ParamInitialize struct {
	InitializeParams
	WorkDoneProgressParams
}
type Workspace2Gn struct {
	/**
	 * The client supports applying batch edits
	 * to the workspace by supporting the request
	 * 'workspace/applyEdit'
	 */
	ApplyEdit bool `json:"applyEdit,omitempty"`

	/**
	 * Capabilities specific to `WorkspaceEdit`s
	 */
	WorkspaceEdit *WorkspaceEditClientCapabilities `json:"workspaceEdit,omitempty"`

	/**
	 * Capabilities specific to the `workspace/didChangeConfiguration` notification.
	 */
	DidChangeConfiguration DidChangeConfigurationClientCapabilities `json:"didChangeConfiguration,omitempty"`

	/**
	 * Capabilities specific to the `workspace/didChangeWatchedFiles` notification.
	 */
	DidChangeWatchedFiles DidChangeWatchedFilesClientCapabilities `json:"didChangeWatchedFiles,omitempty"`

	/**
	 * Capabilities specific to the `workspace/symbol` request.
	 */
	Symbol *WorkspaceSymbolClientCapabilities `json:"symbol,omitempty"`

	/**
	 * Capabilities specific to the `workspace/executeCommand` request.
	 */
	ExecuteCommand ExecuteCommandClientCapabilities `json:"executeCommand,omitempty"`

	/**
	 * Capabilities specific to the semantic token requests scoped to the
	 * workspace.
	 *
	 * @since 3.16.0.
	 */
	SemanticTokens SemanticTokensWorkspaceClientCapabilities `json:"semanticTokens,omitempty"`

	/**
	 * Capabilities specific to the code lens requests scoped to the
	 * workspace.
	 *
	 * @since 3.16.0.
	 */
	CodeLens CodeLensWorkspaceClientCapabilities `json:"codeLens,omitempty"`

	/**
	 * The client has support for file notifications/requests for user operations on files.
	 *
	 * Since 3.16.0
	 */
	FileOperations *FileOperationClientCapabilities `json:"fileOperations,omitempty"`

	/**
	 * Capabilities specific to the inline values requests scoped to the
	 * workspace.
	 *
	 * @since 3.17.0.
	 */
	InlineValues InlineValuesWorkspaceClientCapabilities `json:"inlineValues,omitempty"`

	/**
	 * The client has support for workspace folders
	 *
	 * @since 3.6.0
	 */
	WorkspaceFolders bool `json:"workspaceFolders,omitempty"`

	/**
	 * The client supports `workspace/configuration` requests.
	 *
	 * @since 3.6.0
	 */
	Configuration bool `json:"configuration,omitempty"`
}
type Workspace3Gn struct {
	/**
	 * The client supports applying batch edits
	 * to the workspace by supporting the request
	 * 'workspace/applyEdit'
	 */
	ApplyEdit bool `json:"applyEdit,omitempty"`

	/**
	 * Capabilities specific to `WorkspaceEdit`s
	 */
	WorkspaceEdit *WorkspaceEditClientCapabilities `json:"workspaceEdit,omitempty"`

	/**
	 * Capabilities specific to the `workspace/didChangeConfiguration` notification.
	 */
	DidChangeConfiguration DidChangeConfigurationClientCapabilities `json:"didChangeConfiguration,omitempty"`

	/**
	 * Capabilities specific to the `workspace/didChangeWatchedFiles` notification.
	 */
	DidChangeWatchedFiles DidChangeWatchedFilesClientCapabilities `json:"didChangeWatchedFiles,omitempty"`

	/**
	 * Capabilities specific to the `workspace/symbol` request.
	 */
	Symbol *WorkspaceSymbolClientCapabilities `json:"symbol,omitempty"`

	/**
	 * Capabilities specific to the `workspace/executeCommand` request.
	 */
	ExecuteCommand ExecuteCommandClientCapabilities `json:"executeCommand,omitempty"`

	/**
	 * Capabilities specific to the semantic token requests scoped to the
	 * workspace.
	 *
	 * @since 3.16.0.
	 */
	SemanticTokens SemanticTokensWorkspaceClientCapabilities `json:"semanticTokens,omitempty"`

	/**
	 * Capabilities specific to the code lens requests scoped to the
	 * workspace.
	 *
	 * @since 3.16.0.
	 */
	CodeLens CodeLensWorkspaceClientCapabilities `json:"codeLens,omitempty"`

	/**
	 * The client has support for file notifications/requests for user operations on files.
	 *
	 * Since 3.16.0
	 */
	FileOperations *FileOperationClientCapabilities `json:"fileOperations,omitempty"`

	/**
	 * Capabilities specific to the inline values requests scoped to the
	 * workspace.
	 *
	 * @since 3.17.0.
	 */
	InlineValues InlineValuesWorkspaceClientCapabilities `json:"inlineValues,omitempty"`

	/**
	 * The client has support for workspace folders
	 *
	 * @since 3.6.0
	 */
	WorkspaceFolders bool `json:"workspaceFolders,omitempty"`

	/**
	 * The client supports `workspace/configuration` requests.
	 *
	 * @since 3.6.0
	 */
	Configuration bool `json:"configuration,omitempty"`
}
type WorkspaceFolders4Gn struct {
	/**
	 * The Server has support for workspace folders
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
	ChangeNotifications string/*string | boolean*/ `json:"changeNotifications,omitempty"`
}
type Workspace5Gn struct {
	/**
	* The server is interested in notifications/requests for operations on files.
	*
	* @since 3.16.0
	 */
	FileOperations *FileOperationOptions `json:"fileOperations,omitempty"`

	WorkspaceFolders WorkspaceFolders4Gn `json:"workspaceFolders,omitempty"`
}
type Workspace6Gn struct {
	/**
	 * The client supports applying batch edits
	 * to the workspace by supporting the request
	 * 'workspace/applyEdit'
	 */
	ApplyEdit bool `json:"applyEdit,omitempty"`

	/**
	 * Capabilities specific to `WorkspaceEdit`s
	 */
	WorkspaceEdit *WorkspaceEditClientCapabilities `json:"workspaceEdit,omitempty"`

	/**
	 * Capabilities specific to the `workspace/didChangeConfiguration` notification.
	 */
	DidChangeConfiguration DidChangeConfigurationClientCapabilities `json:"didChangeConfiguration,omitempty"`

	/**
	 * Capabilities specific to the `workspace/didChangeWatchedFiles` notification.
	 */
	DidChangeWatchedFiles DidChangeWatchedFilesClientCapabilities `json:"didChangeWatchedFiles,omitempty"`

	/**
	 * Capabilities specific to the `workspace/symbol` request.
	 */
	Symbol *WorkspaceSymbolClientCapabilities `json:"symbol,omitempty"`

	/**
	 * Capabilities specific to the `workspace/executeCommand` request.
	 */
	ExecuteCommand ExecuteCommandClientCapabilities `json:"executeCommand,omitempty"`

	/**
	 * Capabilities specific to the semantic token requests scoped to the
	 * workspace.
	 *
	 * @since 3.16.0.
	 */
	SemanticTokens SemanticTokensWorkspaceClientCapabilities `json:"semanticTokens,omitempty"`

	/**
	 * Capabilities specific to the code lens requests scoped to the
	 * workspace.
	 *
	 * @since 3.16.0.
	 */
	CodeLens CodeLensWorkspaceClientCapabilities `json:"codeLens,omitempty"`

	/**
	 * The client has support for file notifications/requests for user operations on files.
	 *
	 * Since 3.16.0
	 */
	FileOperations *FileOperationClientCapabilities `json:"fileOperations,omitempty"`

	/**
	 * Capabilities specific to the inline values requests scoped to the
	 * workspace.
	 *
	 * @since 3.17.0.
	 */
	InlineValues InlineValuesWorkspaceClientCapabilities `json:"inlineValues,omitempty"`

	/**
	 * The client has support for workspace folders
	 *
	 * @since 3.6.0
	 */
	WorkspaceFolders bool `json:"workspaceFolders,omitempty"`

	/**
	 * The client supports `workspace/configuration` requests.
	 *
	 * @since 3.6.0
	 */
	Configuration bool `json:"configuration,omitempty"`
}
type WorkspaceFolders7Gn struct {
	/**
	 * The Server has support for workspace folders
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
	ChangeNotifications string/*string | boolean*/ `json:"changeNotifications,omitempty"`
}
type Workspace8Gn struct {
	/**
	* The server is interested in notifications/requests for operations on files.
	*
	* @since 3.16.0
	 */
	FileOperations *FileOperationOptions `json:"fileOperations,omitempty"`

	WorkspaceFolders WorkspaceFolders7Gn `json:"workspaceFolders,omitempty"`
}
