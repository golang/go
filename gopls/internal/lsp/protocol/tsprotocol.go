// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Code generated for LSP. DO NOT EDIT.

package protocol

// Code generated from protocol/metaModel.json at ref release/protocol/3.17.4-next.2 (hash 184c8a7f010d335582f24337fe182baa6f2fccdd).
// https://github.com/microsoft/vscode-languageserver-node/blob/release/protocol/3.17.4-next.2/protocol/metaModel.json
// LSP metaData.version = 3.17.0.

import "encoding/json"

// A special text edit with an additional change annotation.
//
// @since 3.16.0.
type AnnotatedTextEdit struct { // line 9702
	// The actual identifier of the change annotation
	AnnotationID ChangeAnnotationIdentifier `json:"annotationId"`
	TextEdit
}

// The parameters passed via an apply workspace edit request.
type ApplyWorkspaceEditParams struct { // line 6220
	// An optional label of the workspace edit. This label is
	// presented in the user interface for example on an undo
	// stack to undo the workspace edit.
	Label string `json:"label,omitempty"`
	// The edits to apply.
	Edit WorkspaceEdit `json:"edit"`
}

// The result returned from the apply workspace edit request.
//
// @since 3.17 renamed from ApplyWorkspaceEditResponse
type ApplyWorkspaceEditResult struct { // line 6243
	// Indicates whether the edit was applied or not.
	Applied bool `json:"applied"`
	// An optional textual description for why the edit was not applied.
	// This may be used by the server for diagnostic logging or to provide
	// a suitable error for a request that triggered the edit.
	FailureReason string `json:"failureReason,omitempty"`
	// Depending on the client's failure handling strategy `failedChange` might
	// contain the index of the change that failed. This property is only available
	// if the client signals a `failureHandlingStrategy` in its client capabilities.
	FailedChange uint32 `json:"failedChange,omitempty"`
}

// A base for all symbol information.
type BaseSymbolInformation struct { // line 9284
	// The name of this symbol.
	Name string `json:"name"`
	// The kind of this symbol.
	Kind SymbolKind `json:"kind"`
	// Tags for this symbol.
	//
	// @since 3.16.0
	Tags []SymbolTag `json:"tags,omitempty"`
	// The name of the symbol containing this symbol. This information is for
	// user interface purposes (e.g. to render a qualifier in the user interface
	// if necessary). It can't be used to re-infer a hierarchy for the document
	// symbols.
	ContainerName string `json:"containerName,omitempty"`
}

// @since 3.16.0
type CallHierarchyClientCapabilities struct { // line 12517
	// Whether implementation supports dynamic registration. If this is set to `true`
	// the client supports the new `(TextDocumentRegistrationOptions & StaticRegistrationOptions)`
	// return value for the corresponding server capability as well.
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
}

// Represents an incoming call, e.g. a caller of a method or constructor.
//
// @since 3.16.0
type CallHierarchyIncomingCall struct { // line 2852
	// The item that makes the call.
	From CallHierarchyItem `json:"from"`
	// The ranges at which the calls appear. This is relative to the caller
	// denoted by {@link CallHierarchyIncomingCall.from `this.from`}.
	FromRanges []Range `json:"fromRanges"`
}

// The parameter of a `callHierarchy/incomingCalls` request.
//
// @since 3.16.0
type CallHierarchyIncomingCallsParams struct { // line 2828
	Item CallHierarchyItem `json:"item"`
	WorkDoneProgressParams
	PartialResultParams
}

// Represents programming constructs like functions or constructors in the context
// of call hierarchy.
//
// @since 3.16.0
type CallHierarchyItem struct { // line 2729
	// The name of this item.
	Name string `json:"name"`
	// The kind of this item.
	Kind SymbolKind `json:"kind"`
	// Tags for this item.
	Tags []SymbolTag `json:"tags,omitempty"`
	// More detail for this item, e.g. the signature of a function.
	Detail string `json:"detail,omitempty"`
	// The resource identifier of this item.
	URI DocumentURI `json:"uri"`
	// The range enclosing this symbol not including leading/trailing whitespace but everything else, e.g. comments and code.
	Range Range `json:"range"`
	// The range that should be selected and revealed when this symbol is being picked, e.g. the name of a function.
	// Must be contained by the {@link CallHierarchyItem.range `range`}.
	SelectionRange Range `json:"selectionRange"`
	// A data entry field that is preserved between a call hierarchy prepare and
	// incoming calls or outgoing calls requests.
	Data interface{} `json:"data,omitempty"`
}

// Call hierarchy options used during static registration.
//
// @since 3.16.0
type CallHierarchyOptions struct { // line 6770
	WorkDoneProgressOptions
}

// Represents an outgoing call, e.g. calling a getter from a method or a method from a constructor etc.
//
// @since 3.16.0
type CallHierarchyOutgoingCall struct { // line 2902
	// The item that is called.
	To CallHierarchyItem `json:"to"`
	// The range at which this item is called. This is the range relative to the caller, e.g the item
	// passed to {@link CallHierarchyItemProvider.provideCallHierarchyOutgoingCalls `provideCallHierarchyOutgoingCalls`}
	// and not {@link CallHierarchyOutgoingCall.to `this.to`}.
	FromRanges []Range `json:"fromRanges"`
}

// The parameter of a `callHierarchy/outgoingCalls` request.
//
// @since 3.16.0
type CallHierarchyOutgoingCallsParams struct { // line 2878
	Item CallHierarchyItem `json:"item"`
	WorkDoneProgressParams
	PartialResultParams
}

// The parameter of a `textDocument/prepareCallHierarchy` request.
//
// @since 3.16.0
type CallHierarchyPrepareParams struct { // line 2711
	TextDocumentPositionParams
	WorkDoneProgressParams
}

// Call hierarchy options used during static or dynamic registration.
//
// @since 3.16.0
type CallHierarchyRegistrationOptions struct { // line 2806
	TextDocumentRegistrationOptions
	CallHierarchyOptions
	StaticRegistrationOptions
}
type CancelParams struct { // line 6415
	// The request id to cancel.
	ID interface{} `json:"id"`
}

// Additional information that describes document changes.
//
// @since 3.16.0
type ChangeAnnotation struct { // line 7067
	// A human-readable string describing the actual change. The string
	// is rendered prominent in the user interface.
	Label string `json:"label"`
	// A flag which indicates that user confirmation is needed
	// before applying the change.
	NeedsConfirmation bool `json:"needsConfirmation,omitempty"`
	// A human-readable string which is rendered less prominent in
	// the user interface.
	Description string `json:"description,omitempty"`
}

// An identifier to refer to a change annotation stored with a workspace edit.
type ChangeAnnotationIdentifier = string // (alias) line 14391
// Defines the capabilities provided by the client.
type ClientCapabilities struct { // line 10028
	// Workspace specific client capabilities.
	Workspace WorkspaceClientCapabilities `json:"workspace,omitempty"`
	// Text document specific client capabilities.
	TextDocument TextDocumentClientCapabilities `json:"textDocument,omitempty"`
	// Capabilities specific to the notebook document support.
	//
	// @since 3.17.0
	NotebookDocument *NotebookDocumentClientCapabilities `json:"notebookDocument,omitempty"`
	// Window specific client capabilities.
	Window WindowClientCapabilities `json:"window,omitempty"`
	// General client capabilities.
	//
	// @since 3.16.0
	General *GeneralClientCapabilities `json:"general,omitempty"`
	// Experimental client capabilities.
	Experimental interface{} `json:"experimental,omitempty"`
}

// A code action represents a change that can be performed in code, e.g. to fix a problem or
// to refactor code.
//
// A CodeAction must set either `edit` and/or a `command`. If both are supplied, the `edit` is applied first, then the `command` is executed.
type CodeAction struct { // line 5577
	// A short, human-readable, title for this code action.
	Title string `json:"title"`
	// The kind of the code action.
	//
	// Used to filter code actions.
	Kind CodeActionKind `json:"kind,omitempty"`
	// The diagnostics that this code action resolves.
	Diagnostics []Diagnostic `json:"diagnostics,omitempty"`
	// Marks this as a preferred action. Preferred actions are used by the `auto fix` command and can be targeted
	// by keybindings.
	//
	// A quick fix should be marked preferred if it properly addresses the underlying error.
	// A refactoring should be marked preferred if it is the most reasonable choice of actions to take.
	//
	// @since 3.15.0
	IsPreferred bool `json:"isPreferred,omitempty"`
	// Marks that the code action cannot currently be applied.
	//
	// Clients should follow the following guidelines regarding disabled code actions:
	//
	//   - Disabled code actions are not shown in automatic [lightbulbs](https://code.visualstudio.com/docs/editor/editingevolved#_code-action)
	//     code action menus.
	//
	//   - Disabled actions are shown as faded out in the code action menu when the user requests a more specific type
	//     of code action, such as refactorings.
	//
	//   - If the user has a [keybinding](https://code.visualstudio.com/docs/editor/refactoring#_keybindings-for-code-actions)
	//     that auto applies a code action and only disabled code actions are returned, the client should show the user an
	//     error message with `reason` in the editor.
	//
	// @since 3.16.0
	Disabled *PDisabledMsg_textDocument_codeAction `json:"disabled,omitempty"`
	// The workspace edit this code action performs.
	Edit *WorkspaceEdit `json:"edit,omitempty"`
	// A command this code action executes. If a code action
	// provides an edit and a command, first the edit is
	// executed and then the command.
	Command *Command `json:"command,omitempty"`
	// A data entry field that is preserved on a code action between
	// a `textDocument/codeAction` and a `codeAction/resolve` request.
	//
	// @since 3.16.0
	Data interface{} `json:"data,omitempty"`
}

// The Client Capabilities of a {@link CodeActionRequest}.
type CodeActionClientCapabilities struct { // line 12086
	// Whether code action supports dynamic registration.
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
	// The client support code action literals of type `CodeAction` as a valid
	// response of the `textDocument/codeAction` request. If the property is not
	// set the request can only return `Command` literals.
	//
	// @since 3.8.0
	CodeActionLiteralSupport PCodeActionLiteralSupportPCodeAction `json:"codeActionLiteralSupport,omitempty"`
	// Whether code action supports the `isPreferred` property.
	//
	// @since 3.15.0
	IsPreferredSupport bool `json:"isPreferredSupport,omitempty"`
	// Whether code action supports the `disabled` property.
	//
	// @since 3.16.0
	DisabledSupport bool `json:"disabledSupport,omitempty"`
	// Whether code action supports the `data` property which is
	// preserved between a `textDocument/codeAction` and a
	// `codeAction/resolve` request.
	//
	// @since 3.16.0
	DataSupport bool `json:"dataSupport,omitempty"`
	// Whether the client supports resolving additional code action
	// properties via a separate `codeAction/resolve` request.
	//
	// @since 3.16.0
	ResolveSupport *PResolveSupportPCodeAction `json:"resolveSupport,omitempty"`
	// Whether the client honors the change annotations in
	// text edits and resource operations returned via the
	// `CodeAction#edit` property by for example presenting
	// the workspace edit in the user interface and asking
	// for confirmation.
	//
	// @since 3.16.0
	HonorsChangeAnnotations bool `json:"honorsChangeAnnotations,omitempty"`
}

// Contains additional diagnostic information about the context in which
// a {@link CodeActionProvider.provideCodeActions code action} is run.
type CodeActionContext struct { // line 9350
	// An array of diagnostics known on the client side overlapping the range provided to the
	// `textDocument/codeAction` request. They are provided so that the server knows which
	// errors are currently presented to the user for the given range. There is no guarantee
	// that these accurately reflect the error state of the resource. The primary parameter
	// to compute code actions is the provided range.
	Diagnostics []Diagnostic `json:"diagnostics"`
	// Requested kind of actions to return.
	//
	// Actions not of this kind are filtered out by the client before being shown. So servers
	// can omit computing them.
	Only []CodeActionKind `json:"only,omitempty"`
	// The reason why code actions were requested.
	//
	// @since 3.17.0
	TriggerKind *CodeActionTriggerKind `json:"triggerKind,omitempty"`
}

// A set of predefined code action kinds
type CodeActionKind string // line 13719
// Provider options for a {@link CodeActionRequest}.
type CodeActionOptions struct { // line 9389
	// CodeActionKinds that this server may return.
	//
	// The list of kinds may be generic, such as `CodeActionKind.Refactor`, or the server
	// may list out every specific kind they provide.
	CodeActionKinds []CodeActionKind `json:"codeActionKinds,omitempty"`
	// The server provides support to resolve additional
	// information for a code action.
	//
	// @since 3.16.0
	ResolveProvider bool `json:"resolveProvider,omitempty"`
	WorkDoneProgressOptions
}

// The parameters of a {@link CodeActionRequest}.
type CodeActionParams struct { // line 5503
	// The document in which the command was invoked.
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	// The range for which the command was invoked.
	Range Range `json:"range"`
	// Context carrying additional information.
	Context CodeActionContext `json:"context"`
	WorkDoneProgressParams
	PartialResultParams
}

// Registration options for a {@link CodeActionRequest}.
type CodeActionRegistrationOptions struct { // line 5671
	TextDocumentRegistrationOptions
	CodeActionOptions
}

// The reason why code actions were requested.
//
// @since 3.17.0
type CodeActionTriggerKind uint32 // line 14021
// Structure to capture a description for an error code.
//
// @since 3.16.0
type CodeDescription struct { // line 10380
	// An URI to open with more information about the diagnostic error.
	Href URI `json:"href"`
}

// A code lens represents a {@link Command command} that should be shown along with
// source text, like the number of references, a way to run tests, etc.
//
// A code lens is _unresolved_ when no command is associated to it. For performance
// reasons the creation of a code lens and resolving should be done in two stages.
type CodeLens struct { // line 5794
	// The range in which this code lens is valid. Should only span a single line.
	Range Range `json:"range"`
	// The command this code lens represents.
	Command *Command `json:"command,omitempty"`
	// A data entry field that is preserved on a code lens item between
	// a {@link CodeLensRequest} and a [CodeLensResolveRequest]
	// (#CodeLensResolveRequest)
	Data interface{} `json:"data,omitempty"`
}

// The client capabilities  of a {@link CodeLensRequest}.
type CodeLensClientCapabilities struct { // line 12200
	// Whether code lens supports dynamic registration.
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
}

// Code Lens provider options of a {@link CodeLensRequest}.
type CodeLensOptions struct { // line 9445
	// Code lens has a resolve provider as well.
	ResolveProvider bool `json:"resolveProvider,omitempty"`
	WorkDoneProgressOptions
}

// The parameters of a {@link CodeLensRequest}.
type CodeLensParams struct { // line 5770
	// The document to request code lens for.
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	WorkDoneProgressParams
	PartialResultParams
}

// Registration options for a {@link CodeLensRequest}.
type CodeLensRegistrationOptions struct { // line 5826
	TextDocumentRegistrationOptions
	CodeLensOptions
}

// @since 3.16.0
type CodeLensWorkspaceClientCapabilities struct { // line 11358
	// Whether the client implementation supports a refresh request sent from the
	// server to the client.
	//
	// Note that this event is global and will force the client to refresh all
	// code lenses currently shown. It should be used with absolute care and is
	// useful for situation where a server for example detect a project wide
	// change that requires such a calculation.
	RefreshSupport bool `json:"refreshSupport,omitempty"`
}

// Represents a color in RGBA space.
type Color struct { // line 6669
	// The red component of this color in the range [0-1].
	Red float64 `json:"red"`
	// The green component of this color in the range [0-1].
	Green float64 `json:"green"`
	// The blue component of this color in the range [0-1].
	Blue float64 `json:"blue"`
	// The alpha component of this color in the range [0-1].
	Alpha float64 `json:"alpha"`
}

// Represents a color range from a document.
type ColorInformation struct { // line 2312
	// The range in the document where this color appears.
	Range Range `json:"range"`
	// The actual color value for this color range.
	Color Color `json:"color"`
}
type ColorPresentation struct { // line 2394
	// The label of this color presentation. It will be shown on the color
	// picker header. By default this is also the text that is inserted when selecting
	// this color presentation.
	Label string `json:"label"`
	// An {@link TextEdit edit} which is applied to a document when selecting
	// this presentation for the color.  When `falsy` the {@link ColorPresentation.label label}
	// is used.
	TextEdit *TextEdit `json:"textEdit,omitempty"`
	// An optional array of additional {@link TextEdit text edits} that are applied when
	// selecting this color presentation. Edits must not overlap with the main {@link ColorPresentation.textEdit edit} nor with themselves.
	AdditionalTextEdits []TextEdit `json:"additionalTextEdits,omitempty"`
}

// Parameters for a {@link ColorPresentationRequest}.
type ColorPresentationParams struct { // line 2354
	// The text document.
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	// The color to request presentations for.
	Color Color `json:"color"`
	// The range where the color would be inserted. Serves as a context.
	Range Range `json:"range"`
	WorkDoneProgressParams
	PartialResultParams
}

// Represents a reference to a command. Provides a title which
// will be used to represent a command in the UI and, optionally,
// an array of arguments which will be passed to the command handler
// function when invoked.
type Command struct { // line 5543
	// Title of the command, like `save`.
	Title string `json:"title"`
	// The identifier of the actual command handler.
	Command string `json:"command"`
	// Arguments that the command handler should be
	// invoked with.
	Arguments []json.RawMessage `json:"arguments,omitempty"`
}

// Completion client capabilities
type CompletionClientCapabilities struct { // line 11533
	// Whether completion supports dynamic registration.
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
	// The client supports the following `CompletionItem` specific
	// capabilities.
	CompletionItem     PCompletionItemPCompletion      `json:"completionItem,omitempty"`
	CompletionItemKind *PCompletionItemKindPCompletion `json:"completionItemKind,omitempty"`
	// Defines how the client handles whitespace and indentation
	// when accepting a completion item that uses multi line
	// text in either `insertText` or `textEdit`.
	//
	// @since 3.17.0
	InsertTextMode InsertTextMode `json:"insertTextMode,omitempty"`
	// The client supports to send additional context information for a
	// `textDocument/completion` request.
	ContextSupport bool `json:"contextSupport,omitempty"`
	// The client supports the following `CompletionList` specific
	// capabilities.
	//
	// @since 3.17.0
	CompletionList *PCompletionListPCompletion `json:"completionList,omitempty"`
}

// Contains additional information about the context in which a completion request is triggered.
type CompletionContext struct { // line 8946
	// How the completion was triggered.
	TriggerKind CompletionTriggerKind `json:"triggerKind"`
	// The trigger character (a single character) that has trigger code complete.
	// Is undefined if `triggerKind !== CompletionTriggerKind.TriggerCharacter`
	TriggerCharacter string `json:"triggerCharacter,omitempty"`
}

// A completion item represents a text snippet that is
// proposed to complete text that is being typed.
type CompletionItem struct { // line 4723
	// The label of this completion item.
	//
	// The label property is also by default the text that
	// is inserted when selecting this completion.
	//
	// If label details are provided the label itself should
	// be an unqualified name of the completion item.
	Label string `json:"label"`
	// Additional details for the label
	//
	// @since 3.17.0
	LabelDetails *CompletionItemLabelDetails `json:"labelDetails,omitempty"`
	// The kind of this completion item. Based of the kind
	// an icon is chosen by the editor.
	Kind CompletionItemKind `json:"kind,omitempty"`
	// Tags for this completion item.
	//
	// @since 3.15.0
	Tags []CompletionItemTag `json:"tags,omitempty"`
	// A human-readable string with additional information
	// about this item, like type or symbol information.
	Detail string `json:"detail,omitempty"`
	// A human-readable string that represents a doc-comment.
	Documentation *Or_CompletionItem_documentation `json:"documentation,omitempty"`
	// Indicates if this item is deprecated.
	// @deprecated Use `tags` instead.
	Deprecated bool `json:"deprecated,omitempty"`
	// Select this item when showing.
	//
	// *Note* that only one completion item can be selected and that the
	// tool / client decides which item that is. The rule is that the *first*
	// item of those that match best is selected.
	Preselect bool `json:"preselect,omitempty"`
	// A string that should be used when comparing this item
	// with other items. When `falsy` the {@link CompletionItem.label label}
	// is used.
	SortText string `json:"sortText,omitempty"`
	// A string that should be used when filtering a set of
	// completion items. When `falsy` the {@link CompletionItem.label label}
	// is used.
	FilterText string `json:"filterText,omitempty"`
	// A string that should be inserted into a document when selecting
	// this completion. When `falsy` the {@link CompletionItem.label label}
	// is used.
	//
	// The `insertText` is subject to interpretation by the client side.
	// Some tools might not take the string literally. For example
	// VS Code when code complete is requested in this example
	// `con<cursor position>` and a completion item with an `insertText` of
	// `console` is provided it will only insert `sole`. Therefore it is
	// recommended to use `textEdit` instead since it avoids additional client
	// side interpretation.
	InsertText string `json:"insertText,omitempty"`
	// The format of the insert text. The format applies to both the
	// `insertText` property and the `newText` property of a provided
	// `textEdit`. If omitted defaults to `InsertTextFormat.PlainText`.
	//
	// Please note that the insertTextFormat doesn't apply to
	// `additionalTextEdits`.
	InsertTextFormat *InsertTextFormat `json:"insertTextFormat,omitempty"`
	// How whitespace and indentation is handled during completion
	// item insertion. If not provided the clients default value depends on
	// the `textDocument.completion.insertTextMode` client capability.
	//
	// @since 3.16.0
	InsertTextMode *InsertTextMode `json:"insertTextMode,omitempty"`
	// An {@link TextEdit edit} which is applied to a document when selecting
	// this completion. When an edit is provided the value of
	// {@link CompletionItem.insertText insertText} is ignored.
	//
	// Most editors support two different operations when accepting a completion
	// item. One is to insert a completion text and the other is to replace an
	// existing text with a completion text. Since this can usually not be
	// predetermined by a server it can report both ranges. Clients need to
	// signal support for `InsertReplaceEdits` via the
	// `textDocument.completion.insertReplaceSupport` client capability
	// property.
	//
	// *Note 1:* The text edit's range as well as both ranges from an insert
	// replace edit must be a [single line] and they must contain the position
	// at which completion has been requested.
	// *Note 2:* If an `InsertReplaceEdit` is returned the edit's insert range
	// must be a prefix of the edit's replace range, that means it must be
	// contained and starting at the same position.
	//
	// @since 3.16.0 additional type `InsertReplaceEdit`
	TextEdit *TextEdit `json:"textEdit,omitempty"`
	// The edit text used if the completion item is part of a CompletionList and
	// CompletionList defines an item default for the text edit range.
	//
	// Clients will only honor this property if they opt into completion list
	// item defaults using the capability `completionList.itemDefaults`.
	//
	// If not provided and a list's default range is provided the label
	// property is used as a text.
	//
	// @since 3.17.0
	TextEditText string `json:"textEditText,omitempty"`
	// An optional array of additional {@link TextEdit text edits} that are applied when
	// selecting this completion. Edits must not overlap (including the same insert position)
	// with the main {@link CompletionItem.textEdit edit} nor with themselves.
	//
	// Additional text edits should be used to change text unrelated to the current cursor position
	// (for example adding an import statement at the top of the file if the completion item will
	// insert an unqualified type).
	AdditionalTextEdits []TextEdit `json:"additionalTextEdits,omitempty"`
	// An optional set of characters that when pressed while this completion is active will accept it first and
	// then type that character. *Note* that all commit characters should have `length=1` and that superfluous
	// characters will be ignored.
	CommitCharacters []string `json:"commitCharacters,omitempty"`
	// An optional {@link Command command} that is executed *after* inserting this completion. *Note* that
	// additional modifications to the current document should be described with the
	// {@link CompletionItem.additionalTextEdits additionalTextEdits}-property.
	Command *Command `json:"command,omitempty"`
	// A data entry field that is preserved on a completion item between a
	// {@link CompletionRequest} and a {@link CompletionResolveRequest}.
	Data interface{} `json:"data,omitempty"`
}

// The kind of a completion entry.
type CompletionItemKind uint32 // line 13527
// Additional details for a completion item label.
//
// @since 3.17.0
type CompletionItemLabelDetails struct { // line 8969
	// An optional string which is rendered less prominently directly after {@link CompletionItem.label label},
	// without any spacing. Should be used for function signatures and type annotations.
	Detail string `json:"detail,omitempty"`
	// An optional string which is rendered less prominently after {@link CompletionItem.detail}. Should be used
	// for fully qualified names and file paths.
	Description string `json:"description,omitempty"`
}

// Completion item tags are extra annotations that tweak the rendering of a completion
// item.
//
// @since 3.15.0
type CompletionItemTag uint32 // line 13637
// Represents a collection of {@link CompletionItem completion items} to be presented
// in the editor.
type CompletionList struct { // line 4932
	// This list it not complete. Further typing results in recomputing this list.
	//
	// Recomputed lists have all their items replaced (not appended) in the
	// incomplete completion sessions.
	IsIncomplete bool `json:"isIncomplete"`
	// In many cases the items of an actual completion result share the same
	// value for properties like `commitCharacters` or the range of a text
	// edit. A completion list can therefore define item defaults which will
	// be used if a completion item itself doesn't specify the value.
	//
	// If a completion list specifies a default value and a completion item
	// also specifies a corresponding value the one from the item is used.
	//
	// Servers are only allowed to return default values if the client
	// signals support for this via the `completionList.itemDefaults`
	// capability.
	//
	// @since 3.17.0
	ItemDefaults *PItemDefaultsMsg_textDocument_completion `json:"itemDefaults,omitempty"`
	// The completion items.
	Items []CompletionItem `json:"items"`
}

// Completion options.
type CompletionOptions struct { // line 9025
	// Most tools trigger completion request automatically without explicitly requesting
	// it using a keyboard shortcut (e.g. Ctrl+Space). Typically they do so when the user
	// starts to type an identifier. For example if the user types `c` in a JavaScript file
	// code complete will automatically pop up present `console` besides others as a
	// completion item. Characters that make up identifiers don't need to be listed here.
	//
	// If code complete should automatically be trigger on characters not being valid inside
	// an identifier (for example `.` in JavaScript) list them in `triggerCharacters`.
	TriggerCharacters []string `json:"triggerCharacters,omitempty"`
	// The list of all possible characters that commit a completion. This field can be used
	// if clients don't support individual commit characters per completion item. See
	// `ClientCapabilities.textDocument.completion.completionItem.commitCharactersSupport`
	//
	// If a server provides both `allCommitCharacters` and commit characters on an individual
	// completion item the ones on the completion item win.
	//
	// @since 3.2.0
	AllCommitCharacters []string `json:"allCommitCharacters,omitempty"`
	// The server provides support to resolve additional
	// information for a completion item.
	ResolveProvider bool `json:"resolveProvider,omitempty"`
	// The server supports the following `CompletionItem` specific
	// capabilities.
	//
	// @since 3.17.0
	CompletionItem *PCompletionItemPCompletionProvider `json:"completionItem,omitempty"`
	WorkDoneProgressOptions
}

// Completion parameters
type CompletionParams struct { // line 4692
	// The completion context. This is only available it the client specifies
	// to send this using the client capability `textDocument.completion.contextSupport === true`
	Context CompletionContext `json:"context,omitempty"`
	TextDocumentPositionParams
	WorkDoneProgressParams
	PartialResultParams
}

// Registration options for a {@link CompletionRequest}.
type CompletionRegistrationOptions struct { // line 5049
	TextDocumentRegistrationOptions
	CompletionOptions
}

// How a completion was triggered
type CompletionTriggerKind uint32 // line 13970
type ConfigurationItem struct {   // line 6632
	// The scope to get the configuration section for.
	ScopeURI string `json:"scopeUri,omitempty"`
	// The configuration section asked for.
	Section string `json:"section,omitempty"`
}

// The parameters of a configuration request.
type ConfigurationParams struct { // line 2272
	Items []ConfigurationItem `json:"items"`
}

// Create file operation.
type CreateFile struct { // line 6948
	// A create
	Kind string `json:"kind"`
	// The resource to create.
	URI DocumentURI `json:"uri"`
	// Additional options
	Options *CreateFileOptions `json:"options,omitempty"`
	ResourceOperation
}

// Options to create a file.
type CreateFileOptions struct { // line 9747
	// Overwrite existing file. Overwrite wins over `ignoreIfExists`
	Overwrite bool `json:"overwrite,omitempty"`
	// Ignore if exists.
	IgnoreIfExists bool `json:"ignoreIfExists,omitempty"`
}

// The parameters sent in notifications/requests for user-initiated creation of
// files.
//
// @since 3.16.0
type CreateFilesParams struct { // line 3248
	// An array of all files/folders created in this operation.
	Files []FileCreate `json:"files"`
}

// The declaration of a symbol representation as one or many {@link Location locations}.
type Declaration = []Location // (alias) line 14248
// @since 3.14.0
type DeclarationClientCapabilities struct { // line 11874
	// Whether declaration supports dynamic registration. If this is set to `true`
	// the client supports the new `DeclarationRegistrationOptions` return value
	// for the corresponding server capability as well.
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
	// The client supports additional metadata in the form of declaration links.
	LinkSupport bool `json:"linkSupport,omitempty"`
}

// Information about where a symbol is declared.
//
// Provides additional metadata over normal {@link Location location} declarations, including the range of
// the declaring symbol.
//
// Servers should prefer returning `DeclarationLink` over `Declaration` if supported
// by the client.
type DeclarationLink = LocationLink // (alias) line 14268
type DeclarationOptions struct {    // line 6727
	WorkDoneProgressOptions
}
type DeclarationParams struct { // line 2567
	TextDocumentPositionParams
	WorkDoneProgressParams
	PartialResultParams
}
type DeclarationRegistrationOptions struct { // line 2587
	DeclarationOptions
	TextDocumentRegistrationOptions
	StaticRegistrationOptions
}

// The definition of a symbol represented as one or many {@link Location locations}.
// For most programming languages there is only one location at which a symbol is
// defined.
//
// Servers should prefer returning `DefinitionLink` over `Definition` if supported
// by the client.
type Definition = Or_Definition // (alias) line 14166
// Client Capabilities for a {@link DefinitionRequest}.
type DefinitionClientCapabilities struct { // line 11899
	// Whether definition supports dynamic registration.
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
	// The client supports additional metadata in the form of definition links.
	//
	// @since 3.14.0
	LinkSupport bool `json:"linkSupport,omitempty"`
}

// Information about where a symbol is defined.
//
// Provides additional metadata over normal {@link Location location} definitions, including the range of
// the defining symbol
type DefinitionLink = LocationLink // (alias) line 14186
// Server Capabilities for a {@link DefinitionRequest}.
type DefinitionOptions struct { // line 9237
	WorkDoneProgressOptions
}

// Parameters for a {@link DefinitionRequest}.
type DefinitionParams struct { // line 5213
	TextDocumentPositionParams
	WorkDoneProgressParams
	PartialResultParams
}

// Registration options for a {@link DefinitionRequest}.
type DefinitionRegistrationOptions struct { // line 5234
	TextDocumentRegistrationOptions
	DefinitionOptions
}

// Delete file operation
type DeleteFile struct { // line 7030
	// A delete
	Kind string `json:"kind"`
	// The file to delete.
	URI DocumentURI `json:"uri"`
	// Delete options.
	Options *DeleteFileOptions `json:"options,omitempty"`
	ResourceOperation
}

// Delete file options
type DeleteFileOptions struct { // line 9795
	// Delete the content recursively if a folder is denoted.
	Recursive bool `json:"recursive,omitempty"`
	// Ignore the operation if the file doesn't exist.
	IgnoreIfNotExists bool `json:"ignoreIfNotExists,omitempty"`
}

// The parameters sent in notifications/requests for user-initiated deletes of
// files.
//
// @since 3.16.0
type DeleteFilesParams struct { // line 3373
	// An array of all files/folders deleted in this operation.
	Files []FileDelete `json:"files"`
}

// Represents a diagnostic, such as a compiler error or warning. Diagnostic objects
// are only valid in the scope of a resource.
type Diagnostic struct { // line 8843
	// The range at which the message applies
	Range Range `json:"range"`
	// The diagnostic's severity. Can be omitted. If omitted it is up to the
	// client to interpret diagnostics as error, warning, info or hint.
	Severity DiagnosticSeverity `json:"severity,omitempty"`
	// The diagnostic's code, which usually appear in the user interface.
	Code interface{} `json:"code,omitempty"`
	// An optional property to describe the error code.
	// Requires the code field (above) to be present/not null.
	//
	// @since 3.16.0
	CodeDescription *CodeDescription `json:"codeDescription,omitempty"`
	// A human-readable string describing the source of this
	// diagnostic, e.g. 'typescript' or 'super lint'. It usually
	// appears in the user interface.
	Source string `json:"source,omitempty"`
	// The diagnostic's message. It usually appears in the user interface
	Message string `json:"message"`
	// Additional metadata about the diagnostic.
	//
	// @since 3.15.0
	Tags []DiagnosticTag `json:"tags,omitempty"`
	// An array of related diagnostic information, e.g. when symbol-names within
	// a scope collide all definitions can be marked via this property.
	RelatedInformation []DiagnosticRelatedInformation `json:"relatedInformation,omitempty"`
	// A data entry field that is preserved between a `textDocument/publishDiagnostics`
	// notification and `textDocument/codeAction` request.
	//
	// @since 3.16.0
	Data *json.RawMessage `json:"data,omitempty"`
}

// Client capabilities specific to diagnostic pull requests.
//
// @since 3.17.0
type DiagnosticClientCapabilities struct { // line 12784
	// Whether implementation supports dynamic registration. If this is set to `true`
	// the client supports the new `(TextDocumentRegistrationOptions & StaticRegistrationOptions)`
	// return value for the corresponding server capability as well.
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
	// Whether the clients supports related documents for document diagnostic pulls.
	RelatedDocumentSupport bool `json:"relatedDocumentSupport,omitempty"`
}

// Diagnostic options.
//
// @since 3.17.0
type DiagnosticOptions struct { // line 7529
	// An optional identifier under which the diagnostics are
	// managed by the client.
	Identifier string `json:"identifier,omitempty"`
	// Whether the language has inter file dependencies meaning that
	// editing code in one file can result in a different diagnostic
	// set in another file. Inter file dependencies are common for
	// most programming languages and typically uncommon for linters.
	InterFileDependencies bool `json:"interFileDependencies"`
	// The server provides support for workspace diagnostics as well.
	WorkspaceDiagnostics bool `json:"workspaceDiagnostics"`
	WorkDoneProgressOptions
}

// Diagnostic registration options.
//
// @since 3.17.0
type DiagnosticRegistrationOptions struct { // line 3928
	TextDocumentRegistrationOptions
	DiagnosticOptions
	StaticRegistrationOptions
}

// Represents a related message and source code location for a diagnostic. This should be
// used to point to code locations that cause or related to a diagnostics, e.g when duplicating
// a symbol in a scope.
type DiagnosticRelatedInformation struct { // line 10395
	// The location of this related diagnostic information.
	Location Location `json:"location"`
	// The message of this related diagnostic information.
	Message string `json:"message"`
}

// Cancellation data returned from a diagnostic request.
//
// @since 3.17.0
type DiagnosticServerCancellationData struct { // line 3914
	RetriggerRequest bool `json:"retriggerRequest"`
}

// The diagnostic's severity.
type DiagnosticSeverity uint32 // line 13919
// The diagnostic tags.
//
// @since 3.15.0
type DiagnosticTag uint32 // line 13949
// Workspace client capabilities specific to diagnostic pull requests.
//
// @since 3.17.0
type DiagnosticWorkspaceClientCapabilities struct { // line 11476
	// Whether the client implementation supports a refresh request sent from
	// the server to the client.
	//
	// Note that this event is global and will force the client to refresh all
	// pulled diagnostics currently shown. It should be used with absolute care and
	// is useful for situation where a server for example detects a project wide
	// change that requires such a calculation.
	RefreshSupport bool `json:"refreshSupport,omitempty"`
}
type DidChangeConfigurationClientCapabilities struct { // line 11202
	// Did change configuration notification supports dynamic registration.
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
}

// The parameters of a change configuration notification.
type DidChangeConfigurationParams struct { // line 4339
	// The actual changed settings
	Settings interface{} `json:"settings"`
}
type DidChangeConfigurationRegistrationOptions struct { // line 4353
	Section *OrPSection_workspace_didChangeConfiguration `json:"section,omitempty"`
}

// The params sent in a change notebook document notification.
//
// @since 3.17.0
type DidChangeNotebookDocumentParams struct { // line 4047
	// The notebook document that did change. The version number points
	// to the version after all provided changes have been applied. If
	// only the text document content of a cell changes the notebook version
	// doesn't necessarily have to change.
	NotebookDocument VersionedNotebookDocumentIdentifier `json:"notebookDocument"`
	// The actual changes to the notebook document.
	//
	// The changes describe single state changes to the notebook document.
	// So if there are two changes c1 (at array index 0) and c2 (at array
	// index 1) for a notebook in state S then c1 moves the notebook from
	// S to S' and c2 from S' to S''. So c1 is computed on the state S and
	// c2 is computed on the state S'.
	//
	// To mirror the content of a notebook using change events use the following approach:
	//
	//  - start with the same initial content
	//  - apply the 'notebookDocument/didChange' notifications in the order you receive them.
	//  - apply the `NotebookChangeEvent`s in a single notification in the order
	//   you receive them.
	Change NotebookDocumentChangeEvent `json:"change"`
}

// The change text document notification's parameters.
type DidChangeTextDocumentParams struct { // line 4482
	// The document that did change. The version number points
	// to the version after all provided content changes have
	// been applied.
	TextDocument VersionedTextDocumentIdentifier `json:"textDocument"`
	// The actual content changes. The content changes describe single state changes
	// to the document. So if there are two content changes c1 (at array index 0) and
	// c2 (at array index 1) for a document in state S then c1 moves the document from
	// S to S' and c2 from S' to S''. So c1 is computed on the state S and c2 is computed
	// on the state S'.
	//
	// To mirror the content of a document using change events use the following approach:
	//
	//  - start with the same initial content
	//  - apply the 'textDocument/didChange' notifications in the order you receive them.
	//  - apply the `TextDocumentContentChangeEvent`s in a single notification in the order
	//   you receive them.
	ContentChanges []TextDocumentContentChangeEvent `json:"contentChanges"`
}
type DidChangeWatchedFilesClientCapabilities struct { // line 11216
	// Did change watched files notification supports dynamic registration. Please note
	// that the current protocol doesn't support static configuration for file changes
	// from the server side.
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
	// Whether the client has support for {@link  RelativePattern relative pattern}
	// or not.
	//
	// @since 3.17.0
	RelativePatternSupport bool `json:"relativePatternSupport,omitempty"`
}

// The watched files change notification's parameters.
type DidChangeWatchedFilesParams struct { // line 4623
	// The actual file events.
	Changes []FileEvent `json:"changes"`
}

// Describe options to be used when registered for text document change events.
type DidChangeWatchedFilesRegistrationOptions struct { // line 4640
	// The watchers to register.
	Watchers []FileSystemWatcher `json:"watchers"`
}

// The parameters of a `workspace/didChangeWorkspaceFolders` notification.
type DidChangeWorkspaceFoldersParams struct { // line 2258
	// The actual workspace folder change event.
	Event WorkspaceFoldersChangeEvent `json:"event"`
}

// The params sent in a close notebook document notification.
//
// @since 3.17.0
type DidCloseNotebookDocumentParams struct { // line 4085
	// The notebook document that got closed.
	NotebookDocument NotebookDocumentIdentifier `json:"notebookDocument"`
	// The text documents that represent the content
	// of a notebook cell that got closed.
	CellTextDocuments []TextDocumentIdentifier `json:"cellTextDocuments"`
}

// The parameters sent in a close text document notification
type DidCloseTextDocumentParams struct { // line 4527
	// The document that was closed.
	TextDocument TextDocumentIdentifier `json:"textDocument"`
}

// The params sent in an open notebook document notification.
//
// @since 3.17.0
type DidOpenNotebookDocumentParams struct { // line 4021
	// The notebook document that got opened.
	NotebookDocument NotebookDocument `json:"notebookDocument"`
	// The text documents that represent the content
	// of a notebook cell.
	CellTextDocuments []TextDocumentItem `json:"cellTextDocuments"`
}

// The parameters sent in an open text document notification
type DidOpenTextDocumentParams struct { // line 4468
	// The document that was opened.
	TextDocument TextDocumentItem `json:"textDocument"`
}

// The params sent in a save notebook document notification.
//
// @since 3.17.0
type DidSaveNotebookDocumentParams struct { // line 4070
	// The notebook document that got saved.
	NotebookDocument NotebookDocumentIdentifier `json:"notebookDocument"`
}

// The parameters sent in a save text document notification
type DidSaveTextDocumentParams struct { // line 4541
	// The document that was saved.
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	// Optional the content when saved. Depends on the includeText value
	// when the save notification was requested.
	Text *string `json:"text,omitempty"`
}
type DocumentColorClientCapabilities struct { // line 12240
	// Whether implementation supports dynamic registration. If this is set to `true`
	// the client supports the new `DocumentColorRegistrationOptions` return value
	// for the corresponding server capability as well.
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
}
type DocumentColorOptions struct { // line 6707
	WorkDoneProgressOptions
}

// Parameters for a {@link DocumentColorRequest}.
type DocumentColorParams struct { // line 2288
	// The text document.
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	WorkDoneProgressParams
	PartialResultParams
}
type DocumentColorRegistrationOptions struct { // line 2334
	TextDocumentRegistrationOptions
	DocumentColorOptions
	StaticRegistrationOptions
}

// Parameters of the document diagnostic request.
//
// @since 3.17.0
type DocumentDiagnosticParams struct { // line 3841
	// The text document.
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	// The additional identifier  provided during registration.
	Identifier string `json:"identifier,omitempty"`
	// The result id of a previous response if provided.
	PreviousResultID string `json:"previousResultId,omitempty"`
	WorkDoneProgressParams
	PartialResultParams
}
type DocumentDiagnosticReport = Or_DocumentDiagnosticReport // (alias) line 13909
// The document diagnostic report kinds.
//
// @since 3.17.0
type DocumentDiagnosticReportKind string // line 13115
// A partial result for a document diagnostic report.
//
// @since 3.17.0
type DocumentDiagnosticReportPartialResult struct { // line 3884
	RelatedDocuments map[DocumentURI]interface{} `json:"relatedDocuments"`
}

// A document filter describes a top level text document or
// a notebook cell document.
//
// @since 3.17.0 - proposed support for NotebookCellTextDocumentFilter.
type DocumentFilter = Or_DocumentFilter // (alias) line 14508
// Client capabilities of a {@link DocumentFormattingRequest}.
type DocumentFormattingClientCapabilities struct { // line 12254
	// Whether formatting supports dynamic registration.
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
}

// Provider options for a {@link DocumentFormattingRequest}.
type DocumentFormattingOptions struct { // line 9539
	WorkDoneProgressOptions
}

// The parameters of a {@link DocumentFormattingRequest}.
type DocumentFormattingParams struct { // line 5922
	// The document to format.
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	// The format options.
	Options FormattingOptions `json:"options"`
	WorkDoneProgressParams
}

// Registration options for a {@link DocumentFormattingRequest}.
type DocumentFormattingRegistrationOptions struct { // line 5950
	TextDocumentRegistrationOptions
	DocumentFormattingOptions
}

// A document highlight is a range inside a text document which deserves
// special attention. Usually a document highlight is visualized by changing
// the background color of its range.
type DocumentHighlight struct { // line 5314
	// The range this highlight applies to.
	Range Range `json:"range"`
	// The highlight kind, default is {@link DocumentHighlightKind.Text text}.
	Kind DocumentHighlightKind `json:"kind,omitempty"`
}

// Client Capabilities for a {@link DocumentHighlightRequest}.
type DocumentHighlightClientCapabilities struct { // line 11989
	// Whether document highlight supports dynamic registration.
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
}

// A document highlight kind.
type DocumentHighlightKind uint32 // line 13694
// Provider options for a {@link DocumentHighlightRequest}.
type DocumentHighlightOptions struct { // line 9273
	WorkDoneProgressOptions
}

// Parameters for a {@link DocumentHighlightRequest}.
type DocumentHighlightParams struct { // line 5293
	TextDocumentPositionParams
	WorkDoneProgressParams
	PartialResultParams
}

// Registration options for a {@link DocumentHighlightRequest}.
type DocumentHighlightRegistrationOptions struct { // line 5337
	TextDocumentRegistrationOptions
	DocumentHighlightOptions
}

// A document link is a range in a text document that links to an internal or external resource, like another
// text document or a web site.
type DocumentLink struct { // line 5865
	// The range this link applies to.
	Range Range `json:"range"`
	// The uri this link points to. If missing a resolve request is sent later.
	Target *URI `json:"target,omitempty"`
	// The tooltip text when you hover over this link.
	//
	// If a tooltip is provided, is will be displayed in a string that includes instructions on how to
	// trigger the link, such as `{0} (ctrl + click)`. The specific instructions vary depending on OS,
	// user settings, and localization.
	//
	// @since 3.15.0
	Tooltip string `json:"tooltip,omitempty"`
	// A data entry field that is preserved on a document link between a
	// DocumentLinkRequest and a DocumentLinkResolveRequest.
	Data interface{} `json:"data,omitempty"`
}

// The client capabilities of a {@link DocumentLinkRequest}.
type DocumentLinkClientCapabilities struct { // line 12215
	// Whether document link supports dynamic registration.
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
	// Whether the client supports the `tooltip` property on `DocumentLink`.
	//
	// @since 3.15.0
	TooltipSupport bool `json:"tooltipSupport,omitempty"`
}

// Provider options for a {@link DocumentLinkRequest}.
type DocumentLinkOptions struct { // line 9466
	// Document links have a resolve provider as well.
	ResolveProvider bool `json:"resolveProvider,omitempty"`
	WorkDoneProgressOptions
}

// The parameters of a {@link DocumentLinkRequest}.
type DocumentLinkParams struct { // line 5841
	// The document to provide document links for.
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	WorkDoneProgressParams
	PartialResultParams
}

// Registration options for a {@link DocumentLinkRequest}.
type DocumentLinkRegistrationOptions struct { // line 5907
	TextDocumentRegistrationOptions
	DocumentLinkOptions
}

// Client capabilities of a {@link DocumentOnTypeFormattingRequest}.
type DocumentOnTypeFormattingClientCapabilities struct { // line 12295
	// Whether on type formatting supports dynamic registration.
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
}

// Provider options for a {@link DocumentOnTypeFormattingRequest}.
type DocumentOnTypeFormattingOptions struct { // line 9573
	// A character on which formatting should be triggered, like `{`.
	FirstTriggerCharacter string `json:"firstTriggerCharacter"`
	// More trigger characters.
	MoreTriggerCharacter []string `json:"moreTriggerCharacter,omitempty"`
}

// The parameters of a {@link DocumentOnTypeFormattingRequest}.
type DocumentOnTypeFormattingParams struct { // line 6057
	// The document to format.
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	// The position around which the on type formatting should happen.
	// This is not necessarily the exact position where the character denoted
	// by the property `ch` got typed.
	Position Position `json:"position"`
	// The character that has been typed that triggered the formatting
	// on type request. That is not necessarily the last character that
	// got inserted into the document since the client could auto insert
	// characters as well (e.g. like automatic brace completion).
	Ch string `json:"ch"`
	// The formatting options.
	Options FormattingOptions `json:"options"`
}

// Registration options for a {@link DocumentOnTypeFormattingRequest}.
type DocumentOnTypeFormattingRegistrationOptions struct { // line 6095
	TextDocumentRegistrationOptions
	DocumentOnTypeFormattingOptions
}

// Client capabilities of a {@link DocumentRangeFormattingRequest}.
type DocumentRangeFormattingClientCapabilities struct { // line 12269
	// Whether range formatting supports dynamic registration.
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
	// Whether the client supports formatting multiple ranges at once.
	//
	// @since 3.18.0
	// @proposed
	RangesSupport bool `json:"rangesSupport,omitempty"`
}

// Provider options for a {@link DocumentRangeFormattingRequest}.
type DocumentRangeFormattingOptions struct { // line 9550
	// Whether the server supports formatting multiple ranges at once.
	//
	// @since 3.18.0
	// @proposed
	RangesSupport bool `json:"rangesSupport,omitempty"`
	WorkDoneProgressOptions
}

// The parameters of a {@link DocumentRangeFormattingRequest}.
type DocumentRangeFormattingParams struct { // line 5965
	// The document to format.
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	// The range to format
	Range Range `json:"range"`
	// The format options
	Options FormattingOptions `json:"options"`
	WorkDoneProgressParams
}

// Registration options for a {@link DocumentRangeFormattingRequest}.
type DocumentRangeFormattingRegistrationOptions struct { // line 6001
	TextDocumentRegistrationOptions
	DocumentRangeFormattingOptions
}

// The parameters of a {@link DocumentRangesFormattingRequest}.
//
// @since 3.18.0
// @proposed
type DocumentRangesFormattingParams struct { // line 6016
	// The document to format.
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	// The ranges to format
	Ranges []Range `json:"ranges"`
	// The format options
	Options FormattingOptions `json:"options"`
	WorkDoneProgressParams
}

// A document selector is the combination of one or many document filters.
//
// @sample `let sel:DocumentSelector = [{ language: 'typescript' }, { language: 'json', pattern: '**∕tsconfig.json' }]`;
//
// The use of a string as a document filter is deprecated @since 3.16.0.
type DocumentSelector = []DocumentFilter // (alias) line 14363
// Represents programming constructs like variables, classes, interfaces etc.
// that appear in a document. Document symbols can be hierarchical and they
// have two ranges: one that encloses its definition and one that points to
// its most interesting range, e.g. the range of an identifier.
type DocumentSymbol struct { // line 5406
	// The name of this symbol. Will be displayed in the user interface and therefore must not be
	// an empty string or a string only consisting of white spaces.
	Name string `json:"name"`
	// More detail for this symbol, e.g the signature of a function.
	Detail string `json:"detail,omitempty"`
	// The kind of this symbol.
	Kind SymbolKind `json:"kind"`
	// Tags for this document symbol.
	//
	// @since 3.16.0
	Tags []SymbolTag `json:"tags,omitempty"`
	// Indicates if this symbol is deprecated.
	//
	// @deprecated Use tags instead
	Deprecated bool `json:"deprecated,omitempty"`
	// The range enclosing this symbol not including leading/trailing whitespace but everything else
	// like comments. This information is typically used to determine if the clients cursor is
	// inside the symbol to reveal in the symbol in the UI.
	Range Range `json:"range"`
	// The range that should be selected and revealed when this symbol is being picked, e.g the name of a function.
	// Must be contained by the `range`.
	SelectionRange Range `json:"selectionRange"`
	// Children of this symbol, e.g. properties of a class.
	Children []DocumentSymbol `json:"children,omitempty"`
}

// Client Capabilities for a {@link DocumentSymbolRequest}.
type DocumentSymbolClientCapabilities struct { // line 12004
	// Whether document symbol supports dynamic registration.
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
	// Specific capabilities for the `SymbolKind` in the
	// `textDocument/documentSymbol` request.
	SymbolKind *PSymbolKindPDocumentSymbol `json:"symbolKind,omitempty"`
	// The client supports hierarchical document symbols.
	HierarchicalDocumentSymbolSupport bool `json:"hierarchicalDocumentSymbolSupport,omitempty"`
	// The client supports tags on `SymbolInformation`. Tags are supported on
	// `DocumentSymbol` if `hierarchicalDocumentSymbolSupport` is set to true.
	// Clients supporting tags have to handle unknown tags gracefully.
	//
	// @since 3.16.0
	TagSupport *PTagSupportPDocumentSymbol `json:"tagSupport,omitempty"`
	// The client supports an additional label presented in the UI when
	// registering a document symbol provider.
	//
	// @since 3.16.0
	LabelSupport bool `json:"labelSupport,omitempty"`
}

// Provider options for a {@link DocumentSymbolRequest}.
type DocumentSymbolOptions struct { // line 9328
	// A human-readable string that is shown when multiple outlines trees
	// are shown for the same document.
	//
	// @since 3.16.0
	Label string `json:"label,omitempty"`
	WorkDoneProgressOptions
}

// Parameters for a {@link DocumentSymbolRequest}.
type DocumentSymbolParams struct { // line 5352
	// The text document.
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	WorkDoneProgressParams
	PartialResultParams
}

// Registration options for a {@link DocumentSymbolRequest}.
type DocumentSymbolRegistrationOptions struct { // line 5488
	TextDocumentRegistrationOptions
	DocumentSymbolOptions
}
type DocumentURI string

// Predefined error codes.
type ErrorCodes int32 // line 13136
// The client capabilities of a {@link ExecuteCommandRequest}.
type ExecuteCommandClientCapabilities struct { // line 11327
	// Execute command supports dynamic registration.
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
}

// The server capabilities of a {@link ExecuteCommandRequest}.
type ExecuteCommandOptions struct { // line 9621
	// The commands to be executed on the server
	Commands []string `json:"commands"`
	WorkDoneProgressOptions
}

// The parameters of a {@link ExecuteCommandRequest}.
type ExecuteCommandParams struct { // line 6177
	// The identifier of the actual command handler.
	Command string `json:"command"`
	// Arguments that the command should be invoked with.
	Arguments []json.RawMessage `json:"arguments,omitempty"`
	WorkDoneProgressParams
}

// Registration options for a {@link ExecuteCommandRequest}.
type ExecuteCommandRegistrationOptions struct { // line 6209
	ExecuteCommandOptions
}
type ExecutionSummary struct { // line 10516
	// A strict monotonically increasing value
	// indicating the execution order of a cell
	// inside a notebook.
	ExecutionOrder uint32 `json:"executionOrder"`
	// Whether the execution was successful or
	// not if known by the client.
	Success bool `json:"success,omitempty"`
}

// created for Literal (Lit_CodeActionClientCapabilities_codeActionLiteralSupport_codeActionKind)
type FCodeActionKindPCodeActionLiteralSupport struct { // line 12107
	// The code action kind values the client supports. When this
	// property exists the client also guarantees that it will
	// handle values outside its set gracefully and falls back
	// to a default value when unknown.
	ValueSet []CodeActionKind `json:"valueSet"`
}

// created for Literal (Lit_CompletionList_itemDefaults_editRange_Item1)
type FEditRangePItemDefaults struct { // line 4972
	Insert  Range `json:"insert"`
	Replace Range `json:"replace"`
}

// created for Literal (Lit_SemanticTokensClientCapabilities_requests_full_Item1)
type FFullPRequests struct { // line 12581
	// The client will send the `textDocument/semanticTokens/full/delta` request if
	// the server provides a corresponding handler.
	Delta bool `json:"delta"`
}

// created for Literal (Lit_CompletionClientCapabilities_completionItem_insertTextModeSupport)
type FInsertTextModeSupportPCompletionItem struct { // line 11660
	ValueSet []InsertTextMode `json:"valueSet"`
}

// created for Literal (Lit_SignatureHelpClientCapabilities_signatureInformation_parameterInformation)
type FParameterInformationPSignatureInformation struct { // line 11826
	// The client supports processing label offsets instead of a
	// simple label string.
	//
	// @since 3.14.0
	LabelOffsetSupport bool `json:"labelOffsetSupport,omitempty"`
}

// created for Literal (Lit_SemanticTokensClientCapabilities_requests_range_Item1)
type FRangePRequests struct { // line 12561
}

// created for Literal (Lit_CompletionClientCapabilities_completionItem_resolveSupport)
type FResolveSupportPCompletionItem struct { // line 11636
	// The properties that a client can resolve lazily.
	Properties []string `json:"properties"`
}

// created for Literal (Lit_NotebookDocumentChangeEvent_cells_structure)
type FStructurePCells struct { // line 7723
	// The change to the cell array.
	Array NotebookCellArrayChange `json:"array"`
	// Additional opened cell text documents.
	DidOpen []TextDocumentItem `json:"didOpen,omitempty"`
	// Additional closed cell text documents.
	DidClose []TextDocumentIdentifier `json:"didClose,omitempty"`
}

// created for Literal (Lit_CompletionClientCapabilities_completionItem_tagSupport)
type FTagSupportPCompletionItem struct { // line 11602
	// The tags supported by the client.
	ValueSet []CompletionItemTag `json:"valueSet"`
}
type FailureHandlingKind string // line 14108
// The file event type
type FileChangeType uint32 // line 13869
// Represents information on a file/folder create.
//
// @since 3.16.0
type FileCreate struct { // line 6898
	// A file:// URI for the location of the file/folder being created.
	URI string `json:"uri"`
}

// Represents information on a file/folder delete.
//
// @since 3.16.0
type FileDelete struct { // line 7147
	// A file:// URI for the location of the file/folder being deleted.
	URI string `json:"uri"`
}

// An event describing a file change.
type FileEvent struct { // line 8798
	// The file's uri.
	URI DocumentURI `json:"uri"`
	// The change type.
	Type FileChangeType `json:"type"`
}

// Capabilities relating to events from file operations by the user in the client.
//
// These events do not come from the file system, they come from user operations
// like renaming a file in the UI.
//
// @since 3.16.0
type FileOperationClientCapabilities struct { // line 11374
	// Whether the client supports dynamic registration for file requests/notifications.
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
	// The client has support for sending didCreateFiles notifications.
	DidCreate bool `json:"didCreate,omitempty"`
	// The client has support for sending willCreateFiles requests.
	WillCreate bool `json:"willCreate,omitempty"`
	// The client has support for sending didRenameFiles notifications.
	DidRename bool `json:"didRename,omitempty"`
	// The client has support for sending willRenameFiles requests.
	WillRename bool `json:"willRename,omitempty"`
	// The client has support for sending didDeleteFiles notifications.
	DidDelete bool `json:"didDelete,omitempty"`
	// The client has support for sending willDeleteFiles requests.
	WillDelete bool `json:"willDelete,omitempty"`
}

// A filter to describe in which file operation requests or notifications
// the server is interested in receiving.
//
// @since 3.16.0
type FileOperationFilter struct { // line 7100
	// A Uri scheme like `file` or `untitled`.
	Scheme string `json:"scheme,omitempty"`
	// The actual file operation pattern.
	Pattern FileOperationPattern `json:"pattern"`
}

// Options for notifications/requests for user operations on files.
//
// @since 3.16.0
type FileOperationOptions struct { // line 10319
	// The server is interested in receiving didCreateFiles notifications.
	DidCreate *FileOperationRegistrationOptions `json:"didCreate,omitempty"`
	// The server is interested in receiving willCreateFiles requests.
	WillCreate *FileOperationRegistrationOptions `json:"willCreate,omitempty"`
	// The server is interested in receiving didRenameFiles notifications.
	DidRename *FileOperationRegistrationOptions `json:"didRename,omitempty"`
	// The server is interested in receiving willRenameFiles requests.
	WillRename *FileOperationRegistrationOptions `json:"willRename,omitempty"`
	// The server is interested in receiving didDeleteFiles file notifications.
	DidDelete *FileOperationRegistrationOptions `json:"didDelete,omitempty"`
	// The server is interested in receiving willDeleteFiles file requests.
	WillDelete *FileOperationRegistrationOptions `json:"willDelete,omitempty"`
}

// A pattern to describe in which file operation requests or notifications
// the server is interested in receiving.
//
// @since 3.16.0
type FileOperationPattern struct { // line 9819
	// The glob pattern to match. Glob patterns can have the following syntax:
	//
	//  - `*` to match one or more characters in a path segment
	//  - `?` to match on one character in a path segment
	//  - `**` to match any number of path segments, including none
	//  - `{}` to group sub patterns into an OR expression. (e.g. `**​/*.{ts,js}` matches all TypeScript and JavaScript files)
	//  - `[]` to declare a range of characters to match in a path segment (e.g., `example.[0-9]` to match on `example.0`, `example.1`, …)
	//  - `[!...]` to negate a range of characters to match in a path segment (e.g., `example.[!0-9]` to match on `example.a`, `example.b`, but not `example.0`)
	Glob string `json:"glob"`
	// Whether to match files or folders with this pattern.
	//
	// Matches both if undefined.
	Matches *FileOperationPatternKind `json:"matches,omitempty"`
	// Additional options used during matching.
	Options *FileOperationPatternOptions `json:"options,omitempty"`
}

// A pattern kind describing if a glob pattern matches a file a folder or
// both.
//
// @since 3.16.0
type FileOperationPatternKind string // line 14042
// Matching options for the file operation pattern.
//
// @since 3.16.0
type FileOperationPatternOptions struct { // line 10500
	// The pattern should be matched ignoring casing.
	IgnoreCase bool `json:"ignoreCase,omitempty"`
}

// The options to register for file operations.
//
// @since 3.16.0
type FileOperationRegistrationOptions struct { // line 3337
	// The actual filters.
	Filters []FileOperationFilter `json:"filters"`
}

// Represents information on a file/folder rename.
//
// @since 3.16.0
type FileRename struct { // line 7124
	// A file:// URI for the original location of the file/folder being renamed.
	OldURI string `json:"oldUri"`
	// A file:// URI for the new location of the file/folder being renamed.
	NewURI string `json:"newUri"`
}
type FileSystemWatcher struct { // line 8820
	// The glob pattern to watch. See {@link GlobPattern glob pattern} for more detail.
	//
	// @since 3.17.0 support for relative patterns.
	GlobPattern GlobPattern `json:"globPattern"`
	// The kind of events of interest. If omitted it defaults
	// to WatchKind.Create | WatchKind.Change | WatchKind.Delete
	// which is 7.
	Kind *WatchKind `json:"kind,omitempty"`
}

// Represents a folding range. To be valid, start and end line must be bigger than zero and smaller
// than the number of lines in the document. Clients are free to ignore invalid ranges.
type FoldingRange struct { // line 2488
	// The zero-based start line of the range to fold. The folded area starts after the line's last character.
	// To be valid, the end must be zero or larger and smaller than the number of lines in the document.
	StartLine uint32 `json:"startLine"`
	// The zero-based character offset from where the folded range starts. If not defined, defaults to the length of the start line.
	StartCharacter uint32 `json:"startCharacter,omitempty"`
	// The zero-based end line of the range to fold. The folded area ends with the line's last character.
	// To be valid, the end must be zero or larger and smaller than the number of lines in the document.
	EndLine uint32 `json:"endLine"`
	// The zero-based character offset before the folded range ends. If not defined, defaults to the length of the end line.
	EndCharacter uint32 `json:"endCharacter,omitempty"`
	// Describes the kind of the folding range such as `comment' or 'region'. The kind
	// is used to categorize folding ranges and used by commands like 'Fold all comments'.
	// See {@link FoldingRangeKind} for an enumeration of standardized kinds.
	Kind string `json:"kind,omitempty"`
	// The text that the client should show when the specified range is
	// collapsed. If not defined or not supported by the client, a default
	// will be chosen by the client.
	//
	// @since 3.17.0
	CollapsedText string `json:"collapsedText,omitempty"`
}
type FoldingRangeClientCapabilities struct { // line 12354
	// Whether implementation supports dynamic registration for folding range
	// providers. If this is set to `true` the client supports the new
	// `FoldingRangeRegistrationOptions` return value for the corresponding
	// server capability as well.
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
	// The maximum number of folding ranges that the client prefers to receive
	// per document. The value serves as a hint, servers are free to follow the
	// limit.
	RangeLimit uint32 `json:"rangeLimit,omitempty"`
	// If set, the client signals that it only supports folding complete lines.
	// If set, client will ignore specified `startCharacter` and `endCharacter`
	// properties in a FoldingRange.
	LineFoldingOnly bool `json:"lineFoldingOnly,omitempty"`
	// Specific options for the folding range kind.
	//
	// @since 3.17.0
	FoldingRangeKind *PFoldingRangeKindPFoldingRange `json:"foldingRangeKind,omitempty"`
	// Specific options for the folding range.
	//
	// @since 3.17.0
	FoldingRange *PFoldingRangePFoldingRange `json:"foldingRange,omitempty"`
}

// A set of predefined range kinds.
type FoldingRangeKind string      // line 13208
type FoldingRangeOptions struct { // line 6717
	WorkDoneProgressOptions
}

// Parameters for a {@link FoldingRangeRequest}.
type FoldingRangeParams struct { // line 2464
	// The text document.
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	WorkDoneProgressParams
	PartialResultParams
}
type FoldingRangeRegistrationOptions struct { // line 2547
	TextDocumentRegistrationOptions
	FoldingRangeOptions
	StaticRegistrationOptions
}

// Value-object describing what options formatting should use.
type FormattingOptions struct { // line 9487
	// Size of a tab in spaces.
	TabSize uint32 `json:"tabSize"`
	// Prefer spaces over tabs.
	InsertSpaces bool `json:"insertSpaces"`
	// Trim trailing whitespace on a line.
	//
	// @since 3.15.0
	TrimTrailingWhitespace bool `json:"trimTrailingWhitespace,omitempty"`
	// Insert a newline character at the end of the file if one does not exist.
	//
	// @since 3.15.0
	InsertFinalNewline bool `json:"insertFinalNewline,omitempty"`
	// Trim all newlines after the final newline at the end of the file.
	//
	// @since 3.15.0
	TrimFinalNewlines bool `json:"trimFinalNewlines,omitempty"`
}

// A diagnostic report with a full set of problems.
//
// @since 3.17.0
type FullDocumentDiagnosticReport struct { // line 7471
	// A full document diagnostic report.
	Kind string `json:"kind"`
	// An optional result id. If provided it will
	// be sent on the next diagnostic request for the
	// same document.
	ResultID string `json:"resultId,omitempty"`
	// The actual items.
	Items []Diagnostic `json:"items"`
}

// General client capabilities.
//
// @since 3.16.0
type GeneralClientCapabilities struct { // line 11029
	// Client capability that signals how the client
	// handles stale requests (e.g. a request
	// for which the client will not process the response
	// anymore since the information is outdated).
	//
	// @since 3.17.0
	StaleRequestSupport *PStaleRequestSupportPGeneral `json:"staleRequestSupport,omitempty"`
	// Client capabilities specific to regular expressions.
	//
	// @since 3.16.0
	RegularExpressions *RegularExpressionsClientCapabilities `json:"regularExpressions,omitempty"`
	// Client capabilities specific to the client's markdown parser.
	//
	// @since 3.16.0
	Markdown *MarkdownClientCapabilities `json:"markdown,omitempty"`
	// The position encodings supported by the client. Client and server
	// have to agree on the same position encoding to ensure that offsets
	// (e.g. character position in a line) are interpreted the same on both
	// sides.
	//
	// To keep the protocol backwards compatible the following applies: if
	// the value 'utf-16' is missing from the array of position encodings
	// servers can assume that the client supports UTF-16. UTF-16 is
	// therefore a mandatory encoding.
	//
	// If omitted it defaults to ['utf-16'].
	//
	// Implementation considerations: since the conversion from one encoding
	// into another requires the content of the file / line the conversion
	// is best done where the file is read which is usually on the server
	// side.
	//
	// @since 3.17.0
	PositionEncodings []PositionEncodingKind `json:"positionEncodings,omitempty"`
}

// The glob pattern. Either a string pattern or a relative pattern.
//
// @since 3.17.0
type GlobPattern = string // (alias) line 14542
// The result of a hover request.
type Hover struct { // line 5081
	// The hover's content
	Contents MarkupContent `json:"contents"`
	// An optional range inside the text document that is used to
	// visualize the hover, e.g. by changing the background color.
	Range Range `json:"range,omitempty"`
}
type HoverClientCapabilities struct { // line 11767
	// Whether hover supports dynamic registration.
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
	// Client supports the following content formats for the content
	// property. The order describes the preferred format of the client.
	ContentFormat []MarkupKind `json:"contentFormat,omitempty"`
}

// Hover options.
type HoverOptions struct { // line 9094
	WorkDoneProgressOptions
}

// Parameters for a {@link HoverRequest}.
type HoverParams struct { // line 5064
	TextDocumentPositionParams
	WorkDoneProgressParams
}

// Registration options for a {@link HoverRequest}.
type HoverRegistrationOptions struct { // line 5120
	TextDocumentRegistrationOptions
	HoverOptions
}

// @since 3.6.0
type ImplementationClientCapabilities struct { // line 11948
	// Whether implementation supports dynamic registration. If this is set to `true`
	// the client supports the new `ImplementationRegistrationOptions` return value
	// for the corresponding server capability as well.
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
	// The client supports additional metadata in the form of definition links.
	//
	// @since 3.14.0
	LinkSupport bool `json:"linkSupport,omitempty"`
}
type ImplementationOptions struct { // line 6569
	WorkDoneProgressOptions
}
type ImplementationParams struct { // line 2136
	TextDocumentPositionParams
	WorkDoneProgressParams
	PartialResultParams
}
type ImplementationRegistrationOptions struct { // line 2176
	TextDocumentRegistrationOptions
	ImplementationOptions
	StaticRegistrationOptions
}

// The data type of the ResponseError if the
// initialize request fails.
type InitializeError struct { // line 4321
	// Indicates whether the client execute the following retry logic:
	// (1) show the message provided by the ResponseError to the user
	// (2) user selects retry or cancel
	// (3) if user selected retry the initialize method is sent again.
	Retry bool `json:"retry"`
}
type InitializeParams struct { // line 4263
	XInitializeParams
	WorkspaceFoldersInitializeParams
}

// The result returned from an initialize request.
type InitializeResult struct { // line 4277
	// The capabilities the language server provides.
	Capabilities ServerCapabilities `json:"capabilities"`
	// Information about the server.
	//
	// @since 3.15.0
	ServerInfo *PServerInfoMsg_initialize `json:"serverInfo,omitempty"`
}
type InitializedParams struct { // line 4335
}

// Inlay hint information.
//
// @since 3.17.0
type InlayHint struct { // line 3718
	// The position of this hint.
	Position Position `json:"position"`
	// The label of this hint. A human readable string or an array of
	// InlayHintLabelPart label parts.
	//
	// *Note* that neither the string nor the label part can be empty.
	Label []InlayHintLabelPart `json:"label"`
	// The kind of this hint. Can be omitted in which case the client
	// should fall back to a reasonable default.
	Kind InlayHintKind `json:"kind,omitempty"`
	// Optional text edits that are performed when accepting this inlay hint.
	//
	// *Note* that edits are expected to change the document so that the inlay
	// hint (or its nearest variant) is now part of the document and the inlay
	// hint itself is now obsolete.
	TextEdits []TextEdit `json:"textEdits,omitempty"`
	// The tooltip text when you hover over this item.
	Tooltip *OrPTooltip_textDocument_inlayHint `json:"tooltip,omitempty"`
	// Render padding before the hint.
	//
	// Note: Padding should use the editor's background color, not the
	// background color of the hint itself. That means padding can be used
	// to visually align/separate an inlay hint.
	PaddingLeft bool `json:"paddingLeft,omitempty"`
	// Render padding after the hint.
	//
	// Note: Padding should use the editor's background color, not the
	// background color of the hint itself. That means padding can be used
	// to visually align/separate an inlay hint.
	PaddingRight bool `json:"paddingRight,omitempty"`
	// A data entry field that is preserved on an inlay hint between
	// a `textDocument/inlayHint` and a `inlayHint/resolve` request.
	Data interface{} `json:"data,omitempty"`
}

// Inlay hint client capabilities.
//
// @since 3.17.0
type InlayHintClientCapabilities struct { // line 12745
	// Whether inlay hints support dynamic registration.
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
	// Indicates which properties a client can resolve lazily on an inlay
	// hint.
	ResolveSupport *PResolveSupportPInlayHint `json:"resolveSupport,omitempty"`
}

// Inlay hint kinds.
//
// @since 3.17.0
type InlayHintKind uint32 // line 13426
// An inlay hint label part allows for interactive and composite labels
// of inlay hints.
//
// @since 3.17.0
type InlayHintLabelPart struct { // line 7298
	// The value of this label part.
	Value string `json:"value"`
	// The tooltip text when you hover over this label part. Depending on
	// the client capability `inlayHint.resolveSupport` clients might resolve
	// this property late using the resolve request.
	Tooltip *OrPTooltipPLabel `json:"tooltip,omitempty"`
	// An optional source code location that represents this
	// label part.
	//
	// The editor will use this location for the hover and for code navigation
	// features: This part will become a clickable link that resolves to the
	// definition of the symbol at the given location (not necessarily the
	// location itself), it shows the hover that shows at the given location,
	// and it shows a context menu with further code navigation commands.
	//
	// Depending on the client capability `inlayHint.resolveSupport` clients
	// might resolve this property late using the resolve request.
	Location *Location `json:"location,omitempty"`
	// An optional command for this label part.
	//
	// Depending on the client capability `inlayHint.resolveSupport` clients
	// might resolve this property late using the resolve request.
	Command *Command `json:"command,omitempty"`
}

// Inlay hint options used during static registration.
//
// @since 3.17.0
type InlayHintOptions struct { // line 7371
	// The server provides support to resolve additional
	// information for an inlay hint item.
	ResolveProvider bool `json:"resolveProvider,omitempty"`
	WorkDoneProgressOptions
}

// A parameter literal used in inlay hint requests.
//
// @since 3.17.0
type InlayHintParams struct { // line 3689
	// The text document.
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	// The document range for which inlay hints should be computed.
	Range Range `json:"range"`
	WorkDoneProgressParams
}

// Inlay hint options used during static or dynamic registration.
//
// @since 3.17.0
type InlayHintRegistrationOptions struct { // line 3819
	InlayHintOptions
	TextDocumentRegistrationOptions
	StaticRegistrationOptions
}

// Client workspace capabilities specific to inlay hints.
//
// @since 3.17.0
type InlayHintWorkspaceClientCapabilities struct { // line 11460
	// Whether the client implementation supports a refresh request sent from
	// the server to the client.
	//
	// Note that this event is global and will force the client to refresh all
	// inlay hints currently shown. It should be used with absolute care and
	// is useful for situation where a server for example detects a project wide
	// change that requires such a calculation.
	RefreshSupport bool `json:"refreshSupport,omitempty"`
}

// Client capabilities specific to inline completions.
//
// @since 3.18.0
// @proposed
type InlineCompletionClientCapabilities struct { // line 12809
	// Whether implementation supports dynamic registration for inline completion providers.
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
}

// Provides information about the context in which an inline completion was requested.
//
// @since 3.18.0
// @proposed
type InlineCompletionContext struct { // line 7833
	// Describes how the inline completion was triggered.
	TriggerKind InlineCompletionTriggerKind `json:"triggerKind"`
	// Provides information about the currently selected item in the autocomplete widget if it is visible.
	SelectedCompletionInfo *SelectedCompletionInfo `json:"selectedCompletionInfo,omitempty"`
}

// An inline completion item represents a text snippet that is proposed inline to complete text that is being typed.
//
// @since 3.18.0
// @proposed
type InlineCompletionItem struct { // line 4158
	// The text to replace the range with. Must be set.
	InsertText Or_InlineCompletionItem_insertText `json:"insertText"`
	// A text that is used to decide if this inline completion should be shown. When `falsy` the {@link InlineCompletionItem.insertText} is used.
	FilterText string `json:"filterText,omitempty"`
	// The range to replace. Must begin and end on the same line.
	Range *Range `json:"range,omitempty"`
	// An optional {@link Command} that is executed *after* inserting this completion.
	Command *Command `json:"command,omitempty"`
}

// Represents a collection of {@link InlineCompletionItem inline completion items} to be presented in the editor.
//
// @since 3.18.0
// @proposed
type InlineCompletionList struct { // line 4139
	// The inline completion items
	Items []InlineCompletionItem `json:"items"`
}

// Inline completion options used during static registration.
//
// @since 3.18.0
// @proposed
type InlineCompletionOptions struct { // line 7882
	WorkDoneProgressOptions
}

// A parameter literal used in inline completion requests.
//
// @since 3.18.0
// @proposed
type InlineCompletionParams struct { // line 4111
	// Additional information about the context in which inline completions were
	// requested.
	Context InlineCompletionContext `json:"context"`
	TextDocumentPositionParams
	WorkDoneProgressParams
}

// Inline completion options used during static or dynamic registration.
//
// @since 3.18.0
// @proposed
type InlineCompletionRegistrationOptions struct { // line 4210
	InlineCompletionOptions
	TextDocumentRegistrationOptions
	StaticRegistrationOptions
}

// Describes how an {@link InlineCompletionItemProvider inline completion provider} was triggered.
//
// @since 3.18.0
// @proposed
type InlineCompletionTriggerKind uint32 // line 13820
// Inline value information can be provided by different means:
//
//   - directly as a text value (class InlineValueText).
//   - as a name to use for a variable lookup (class InlineValueVariableLookup)
//   - as an evaluatable expression (class InlineValueEvaluatableExpression)
//
// The InlineValue types combines all inline value types into one type.
//
// @since 3.17.0
type InlineValue = Or_InlineValue // (alias) line 14276
// Client capabilities specific to inline values.
//
// @since 3.17.0
type InlineValueClientCapabilities struct { // line 12729
	// Whether implementation supports dynamic registration for inline value providers.
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
}

// @since 3.17.0
type InlineValueContext struct { // line 7184
	// The stack frame (as a DAP Id) where the execution has stopped.
	FrameID int32 `json:"frameId"`
	// The document range where execution has stopped.
	// Typically the end position of the range denotes the line where the inline values are shown.
	StoppedLocation Range `json:"stoppedLocation"`
}

// Provide an inline value through an expression evaluation.
// If only a range is specified, the expression will be extracted from the underlying document.
// An optional expression can be used to override the extracted expression.
//
// @since 3.17.0
type InlineValueEvaluatableExpression struct { // line 7262
	// The document range for which the inline value applies.
	// The range is used to extract the evaluatable expression from the underlying document.
	Range Range `json:"range"`
	// If specified the expression overrides the extracted expression.
	Expression string `json:"expression,omitempty"`
}

// Inline value options used during static registration.
//
// @since 3.17.0
type InlineValueOptions struct { // line 7286
	WorkDoneProgressOptions
}

// A parameter literal used in inline value requests.
//
// @since 3.17.0
type InlineValueParams struct { // line 3630
	// The text document.
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	// The document range for which inline values should be computed.
	Range Range `json:"range"`
	// Additional information about the context in which inline values were
	// requested.
	Context InlineValueContext `json:"context"`
	WorkDoneProgressParams
}

// Inline value options used during static or dynamic registration.
//
// @since 3.17.0
type InlineValueRegistrationOptions struct { // line 3667
	InlineValueOptions
	TextDocumentRegistrationOptions
	StaticRegistrationOptions
}

// Provide inline value as text.
//
// @since 3.17.0
type InlineValueText struct { // line 7207
	// The document range for which the inline value applies.
	Range Range `json:"range"`
	// The text of the inline value.
	Text string `json:"text"`
}

// Provide inline value through a variable lookup.
// If only a range is specified, the variable name will be extracted from the underlying document.
// An optional variable name can be used to override the extracted name.
//
// @since 3.17.0
type InlineValueVariableLookup struct { // line 7230
	// The document range for which the inline value applies.
	// The range is used to extract the variable name from the underlying document.
	Range Range `json:"range"`
	// If specified the name of the variable to look up.
	VariableName string `json:"variableName,omitempty"`
	// How to perform the lookup.
	CaseSensitiveLookup bool `json:"caseSensitiveLookup"`
}

// Client workspace capabilities specific to inline values.
//
// @since 3.17.0
type InlineValueWorkspaceClientCapabilities struct { // line 11444
	// Whether the client implementation supports a refresh request sent from the
	// server to the client.
	//
	// Note that this event is global and will force the client to refresh all
	// inline values currently shown. It should be used with absolute care and is
	// useful for situation where a server for example detects a project wide
	// change that requires such a calculation.
	RefreshSupport bool `json:"refreshSupport,omitempty"`
}

// A special text edit to provide an insert and a replace operation.
//
// @since 3.16.0
type InsertReplaceEdit struct { // line 8994
	// The string to be inserted.
	NewText string `json:"newText"`
	// The range if the insert is requested
	Insert Range `json:"insert"`
	// The range if the replace is requested.
	Replace Range `json:"replace"`
}

// Defines whether the insert text in a completion item should be interpreted as
// plain text or a snippet.
type InsertTextFormat uint32 // line 13653
// How whitespace and indentation is handled during completion
// item insertion.
//
// @since 3.16.0
type InsertTextMode uint32 // line 13673
type LSPAny = interface{}

// LSP arrays.
// @since 3.17.0
type LSPArray = []interface{} // (alias) line 14194
type LSPErrorCodes int32      // line 13176
// LSP object definition.
// @since 3.17.0
type LSPObject = map[string]LSPAny // (alias) line 14526
// Client capabilities for the linked editing range request.
//
// @since 3.16.0
type LinkedEditingRangeClientCapabilities struct { // line 12681
	// Whether implementation supports dynamic registration. If this is set to `true`
	// the client supports the new `(TextDocumentRegistrationOptions & StaticRegistrationOptions)`
	// return value for the corresponding server capability as well.
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
}
type LinkedEditingRangeOptions struct { // line 6888
	WorkDoneProgressOptions
}
type LinkedEditingRangeParams struct { // line 3185
	TextDocumentPositionParams
	WorkDoneProgressParams
}
type LinkedEditingRangeRegistrationOptions struct { // line 3228
	TextDocumentRegistrationOptions
	LinkedEditingRangeOptions
	StaticRegistrationOptions
}

// The result of a linked editing range request.
//
// @since 3.16.0
type LinkedEditingRanges struct { // line 3201
	// A list of ranges that can be edited together. The ranges must have
	// identical length and contain identical text content. The ranges cannot overlap.
	Ranges []Range `json:"ranges"`
	// An optional word pattern (regular expression) that describes valid contents for
	// the given ranges. If no pattern is provided, the client configuration's word
	// pattern will be used.
	WordPattern string `json:"wordPattern,omitempty"`
}

// created for Literal (Lit_NotebookDocumentChangeEvent_cells_textContent_Elem)
type Lit_NotebookDocumentChangeEvent_cells_textContent_Elem struct { // line 7781
	Document VersionedTextDocumentIdentifier  `json:"document"`
	Changes  []TextDocumentContentChangeEvent `json:"changes"`
}

// created for Literal (Lit_NotebookDocumentFilter_Item1)
type Lit_NotebookDocumentFilter_Item1 struct { // line 14708
	// The type of the enclosing notebook.
	NotebookType string `json:"notebookType,omitempty"`
	// A Uri {@link Uri.scheme scheme}, like `file` or `untitled`.
	Scheme string `json:"scheme"`
	// A glob pattern.
	Pattern string `json:"pattern,omitempty"`
}

// created for Literal (Lit_NotebookDocumentFilter_Item2)
type Lit_NotebookDocumentFilter_Item2 struct { // line 14741
	// The type of the enclosing notebook.
	NotebookType string `json:"notebookType,omitempty"`
	// A Uri {@link Uri.scheme scheme}, like `file` or `untitled`.
	Scheme string `json:"scheme,omitempty"`
	// A glob pattern.
	Pattern string `json:"pattern"`
}

// created for Literal (Lit_NotebookDocumentSyncOptions_notebookSelector_Elem_Item0_cells_Elem)
type Lit_NotebookDocumentSyncOptions_notebookSelector_Elem_Item0_cells_Elem struct { // line 10185
	Language string `json:"language"`
}

// created for Literal (Lit_NotebookDocumentSyncOptions_notebookSelector_Elem_Item1)
type Lit_NotebookDocumentSyncOptions_notebookSelector_Elem_Item1 struct { // line 10206
	// The notebook to be synced If a string
	// value is provided it matches against the
	// notebook type. '*' matches every notebook.
	Notebook *Or_NotebookDocumentSyncOptions_notebookSelector_Elem_Item1_notebook `json:"notebook,omitempty"`
	// The cells of the matching notebook to be synced.
	Cells []Lit_NotebookDocumentSyncOptions_notebookSelector_Elem_Item1_cells_Elem `json:"cells"`
}

// created for Literal (Lit_NotebookDocumentSyncOptions_notebookSelector_Elem_Item1_cells_Elem)
type Lit_NotebookDocumentSyncOptions_notebookSelector_Elem_Item1_cells_Elem struct { // line 10232
	Language string `json:"language"`
}

// created for Literal (Lit_PrepareRenameResult_Item2)
type Lit_PrepareRenameResult_Item2 struct { // line 14347
	DefaultBehavior bool `json:"defaultBehavior"`
}

// created for Literal (Lit_TextDocumentContentChangeEvent_Item1)
type Lit_TextDocumentContentChangeEvent_Item1 struct { // line 14455
	// The new text of the whole document.
	Text string `json:"text"`
}

// created for Literal (Lit_TextDocumentFilter_Item2)
type Lit_TextDocumentFilter_Item2 struct { // line 14632
	// A language id, like `typescript`.
	Language string `json:"language,omitempty"`
	// A Uri {@link Uri.scheme scheme}, like `file` or `untitled`.
	Scheme string `json:"scheme,omitempty"`
	// A glob pattern, like `*.{ts,js}`.
	Pattern string `json:"pattern"`
}

// Represents a location inside a resource, such as a line
// inside a text file.
type Location struct { // line 2156
	URI   DocumentURI `json:"uri"`
	Range Range       `json:"range"`
}

// Represents the connection of two locations. Provides additional metadata over normal {@link Location locations},
// including an origin range.
type LocationLink struct { // line 6508
	// Span of the origin of this link.
	//
	// Used as the underlined span for mouse interaction. Defaults to the word range at
	// the definition position.
	OriginSelectionRange *Range `json:"originSelectionRange,omitempty"`
	// The target resource identifier of this link.
	TargetURI DocumentURI `json:"targetUri"`
	// The full target range of this link. If the target for example is a symbol then target range is the
	// range enclosing this symbol not including leading/trailing whitespace but everything else
	// like comments. This information is typically used to highlight the range in the editor.
	TargetRange Range `json:"targetRange"`
	// The range that should be selected and revealed when this link is being followed, e.g the name of a function.
	// Must be contained by the `targetRange`. See also `DocumentSymbol#range`
	TargetSelectionRange Range `json:"targetSelectionRange"`
}

// The log message parameters.
type LogMessageParams struct { // line 4446
	// The message type. See {@link MessageType}
	Type MessageType `json:"type"`
	// The actual message.
	Message string `json:"message"`
}
type LogTraceParams struct { // line 6395
	Message string `json:"message"`
	Verbose string `json:"verbose,omitempty"`
}

// Client capabilities specific to the used markdown parser.
//
// @since 3.16.0
type MarkdownClientCapabilities struct { // line 12917
	// The name of the parser.
	Parser string `json:"parser"`
	// The version of the parser.
	Version string `json:"version,omitempty"`
	// A list of HTML tags that the client allows / supports in
	// Markdown.
	//
	// @since 3.17.0
	AllowedTags []string `json:"allowedTags,omitempty"`
}

// MarkedString can be used to render human readable text. It is either a markdown string
// or a code-block that provides a language and a code snippet. The language identifier
// is semantically equal to the optional language identifier in fenced code blocks in GitHub
// issues. See https://help.github.com/articles/creating-and-highlighting-code-blocks/#syntax-highlighting
//
// The pair of a language and a value is an equivalent to markdown:
// ```${language}
// ${value}
// ```
//
// Note that markdown strings will be sanitized - that means html will be escaped.
// @deprecated use MarkupContent instead.
type MarkedString = Or_MarkedString // (alias) line 14473
// A `MarkupContent` literal represents a string value which content is interpreted base on its
// kind flag. Currently the protocol supports `plaintext` and `markdown` as markup kinds.
//
// If the kind is `markdown` then the value can contain fenced code blocks like in GitHub issues.
// See https://help.github.com/articles/creating-and-highlighting-code-blocks/#syntax-highlighting
//
// Here is an example how such a string can be constructed using JavaScript / TypeScript:
// ```ts
//
//	let markdown: MarkdownContent = {
//	 kind: MarkupKind.Markdown,
//	 value: [
//	   '# Header',
//	   'Some text',
//	   '```typescript',
//	   'someCode();',
//	   '```'
//	 ].join('\n')
//	};
//
// ```
//
// *Please Note* that clients might sanitize the return markdown. A client could decide to
// remove HTML from the markdown to avoid script execution.
type MarkupContent struct { // line 7349
	// The type of the Markup
	Kind MarkupKind `json:"kind"`
	// The content itself
	Value string `json:"value"`
}

// Describes the content type that a client supports in various
// result literals like `Hover`, `ParameterInfo` or `CompletionItem`.
//
// Please note that `MarkupKinds` must not start with a `$`. This kinds
// are reserved for internal usage.
type MarkupKind string          // line 13800
type MessageActionItem struct { // line 4433
	// A short title like 'Retry', 'Open Log' etc.
	Title string `json:"title"`
}

// The message type
type MessageType uint32 // line 13447
// Moniker definition to match LSIF 0.5 moniker definition.
//
// @since 3.16.0
type Moniker struct { // line 3411
	// The scheme of the moniker. For example tsc or .Net
	Scheme string `json:"scheme"`
	// The identifier of the moniker. The value is opaque in LSIF however
	// schema owners are allowed to define the structure if they want.
	Identifier string `json:"identifier"`
	// The scope in which the moniker is unique
	Unique UniquenessLevel `json:"unique"`
	// The moniker kind if known.
	Kind *MonikerKind `json:"kind,omitempty"`
}

// Client capabilities specific to the moniker request.
//
// @since 3.16.0
type MonikerClientCapabilities struct { // line 12697
	// Whether moniker supports dynamic registration. If this is set to `true`
	// the client supports the new `MonikerRegistrationOptions` return value
	// for the corresponding server capability as well.
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
}

// The moniker kind.
//
// @since 3.16.0
type MonikerKind string      // line 13400
type MonikerOptions struct { // line 7162
	WorkDoneProgressOptions
}
type MonikerParams struct { // line 3391
	TextDocumentPositionParams
	WorkDoneProgressParams
	PartialResultParams
}
type MonikerRegistrationOptions struct { // line 3451
	TextDocumentRegistrationOptions
	MonikerOptions
}

// created for Literal (Lit_MarkedString_Item1)
type Msg_MarkedString struct { // line 14483
	Language string `json:"language"`
	Value    string `json:"value"`
}

// created for Literal (Lit_NotebookDocumentFilter_Item0)
type Msg_NotebookDocumentFilter struct { // line 14675
	// The type of the enclosing notebook.
	NotebookType string `json:"notebookType"`
	// A Uri {@link Uri.scheme scheme}, like `file` or `untitled`.
	Scheme string `json:"scheme,omitempty"`
	// A glob pattern.
	Pattern string `json:"pattern,omitempty"`
}

// created for Literal (Lit_PrepareRenameResult_Item1)
type Msg_PrepareRename2Gn struct { // line 14326
	Range       Range  `json:"range"`
	Placeholder string `json:"placeholder"`
}

// created for Literal (Lit_TextDocumentContentChangeEvent_Item0)
type Msg_TextDocumentContentChangeEvent struct { // line 14423
	// The range of the document that changed.
	Range *Range `json:"range"`
	// The optional length of the range that got replaced.
	//
	// @deprecated use range instead.
	RangeLength uint32 `json:"rangeLength,omitempty"`
	// The new text for the provided range.
	Text string `json:"text"`
}

// created for Literal (Lit_TextDocumentFilter_Item1)
type Msg_TextDocumentFilter struct { // line 14599
	// A language id, like `typescript`.
	Language string `json:"language,omitempty"`
	// A Uri {@link Uri.scheme scheme}, like `file` or `untitled`.
	Scheme string `json:"scheme"`
	// A glob pattern, like `*.{ts,js}`.
	Pattern string `json:"pattern,omitempty"`
}

// created for Literal (Lit__InitializeParams_clientInfo)
type Msg_XInitializeParams_clientInfo struct { // line 7971
	// The name of the client as defined by the client.
	Name string `json:"name"`
	// The client's version as defined by the client.
	Version string `json:"version,omitempty"`
}

// A notebook cell.
//
// A cell's document URI must be unique across ALL notebook
// cells and can therefore be used to uniquely identify a
// notebook cell or the cell's text document.
//
// @since 3.17.0
type NotebookCell struct { // line 9928
	// The cell's kind
	Kind NotebookCellKind `json:"kind"`
	// The URI of the cell's text document
	// content.
	Document DocumentURI `json:"document"`
	// Additional metadata stored with the cell.
	//
	// Note: should always be an object literal (e.g. LSPObject)
	Metadata *LSPObject `json:"metadata,omitempty"`
	// Additional execution summary information
	// if supported by the client.
	ExecutionSummary *ExecutionSummary `json:"executionSummary,omitempty"`
}

// A change describing how to move a `NotebookCell`
// array from state S to S'.
//
// @since 3.17.0
type NotebookCellArrayChange struct { // line 9969
	// The start oftest of the cell that changed.
	Start uint32 `json:"start"`
	// The deleted cells
	DeleteCount uint32 `json:"deleteCount"`
	// The new cells, if any
	Cells []NotebookCell `json:"cells,omitempty"`
}

// A notebook cell kind.
//
// @since 3.17.0
type NotebookCellKind uint32 // line 14063
// A notebook cell text document filter denotes a cell text
// document by different properties.
//
// @since 3.17.0
type NotebookCellTextDocumentFilter struct { // line 10467
	// A filter that matches against the notebook
	// containing the notebook cell. If a string
	// value is provided it matches against the
	// notebook type. '*' matches every notebook.
	Notebook Or_NotebookCellTextDocumentFilter_notebook `json:"notebook"`
	// A language id like `python`.
	//
	// Will be matched against the language id of the
	// notebook cell document. '*' matches every language.
	Language string `json:"language,omitempty"`
}

// A notebook document.
//
// @since 3.17.0
type NotebookDocument struct { // line 7590
	// The notebook document's uri.
	URI URI `json:"uri"`
	// The type of the notebook.
	NotebookType string `json:"notebookType"`
	// The version number of this document (it will increase after each
	// change, including undo/redo).
	Version int32 `json:"version"`
	// Additional metadata stored with the notebook
	// document.
	//
	// Note: should always be an object literal (e.g. LSPObject)
	Metadata *LSPObject `json:"metadata,omitempty"`
	// The cells of a notebook.
	Cells []NotebookCell `json:"cells"`
}

// A change event for a notebook document.
//
// @since 3.17.0
type NotebookDocumentChangeEvent struct { // line 7702
	// The changed meta data if any.
	//
	// Note: should always be an object literal (e.g. LSPObject)
	Metadata *LSPObject `json:"metadata,omitempty"`
	// Changes to cells
	Cells *PCellsPChange `json:"cells,omitempty"`
}

// Capabilities specific to the notebook document support.
//
// @since 3.17.0
type NotebookDocumentClientCapabilities struct { // line 10978
	// Capabilities specific to notebook document synchronization
	//
	// @since 3.17.0
	Synchronization NotebookDocumentSyncClientCapabilities `json:"synchronization"`
}

// A notebook document filter denotes a notebook document by
// different properties. The properties will be match
// against the notebook's URI (same as with documents)
//
// @since 3.17.0
type NotebookDocumentFilter = Msg_NotebookDocumentFilter // (alias) line 14669
// A literal to identify a notebook document in the client.
//
// @since 3.17.0
type NotebookDocumentIdentifier struct { // line 7818
	// The notebook document's uri.
	URI URI `json:"uri"`
}

// Notebook specific client capabilities.
//
// @since 3.17.0
type NotebookDocumentSyncClientCapabilities struct { // line 12826
	// Whether implementation supports dynamic registration. If this is
	// set to `true` the client supports the new
	// `(TextDocumentRegistrationOptions & StaticRegistrationOptions)`
	// return value for the corresponding server capability as well.
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
	// The client supports sending execution summary data per cell.
	ExecutionSummarySupport bool `json:"executionSummarySupport,omitempty"`
}

// Options specific to a notebook plus its cells
// to be synced to the server.
//
// If a selector provides a notebook document
// filter but no cell selector all cells of a
// matching notebook document will be synced.
//
// If a selector provides no notebook document
// filter but only a cell selector all notebook
// document that contain at least one matching
// cell will be synced.
//
// @since 3.17.0
type NotebookDocumentSyncOptions struct { // line 10149
	// The notebooks to be synced
	NotebookSelector []PNotebookSelectorPNotebookDocumentSync `json:"notebookSelector"`
	// Whether save notification should be forwarded to
	// the server. Will only be honored if mode === `notebook`.
	Save bool `json:"save,omitempty"`
}

// Registration options specific to a notebook.
//
// @since 3.17.0
type NotebookDocumentSyncRegistrationOptions struct { // line 10269
	NotebookDocumentSyncOptions
	StaticRegistrationOptions
}

// A text document identifier to optionally denote a specific version of a text document.
type OptionalVersionedTextDocumentIdentifier struct { // line 9673
	// The version number of this document. If a versioned text document identifier
	// is sent from the server to the client and the file is not open in the editor
	// (the server has not received an open notification before) the server can send
	// `null` to indicate that the version is unknown and the content on disk is the
	// truth (as specified with document content ownership).
	Version int32 `json:"version"`
	TextDocumentIdentifier
}

// created for Or [FEditRangePItemDefaults Range]
type OrFEditRangePItemDefaults struct { // line 4965
	Value interface{} `json:"value"`
}

// created for Or [NotebookDocumentFilter string]
type OrFNotebookPNotebookSelector struct { // line 10166
	Value interface{} `json:"value"`
}

// created for Or [Location PLocationMsg_workspace_symbol]
type OrPLocation_workspace_symbol struct { // line 5716
	Value interface{} `json:"value"`
}

// created for Or [[]string string]
type OrPSection_workspace_didChangeConfiguration struct { // line 4359
	Value interface{} `json:"value"`
}

// created for Or [MarkupContent string]
type OrPTooltipPLabel struct { // line 7312
	Value interface{} `json:"value"`
}

// created for Or [MarkupContent string]
type OrPTooltip_textDocument_inlayHint struct { // line 3773
	Value interface{} `json:"value"`
}

// created for Or [int32 string]
type Or_CancelParams_id struct { // line 6421
	Value interface{} `json:"value"`
}

// created for Or [MarkupContent string]
type Or_CompletionItem_documentation struct { // line 4778
	Value interface{} `json:"value"`
}

// created for Or [InsertReplaceEdit TextEdit]
type Or_CompletionItem_textEdit struct { // line 4861
	Value interface{} `json:"value"`
}

// created for Or [Location []Location]
type Or_Definition struct { // line 14169
	Value interface{} `json:"value"`
}

// created for Or [int32 string]
type Or_Diagnostic_code struct { // line 8866
	Value interface{} `json:"value"`
}

// created for Or [RelatedFullDocumentDiagnosticReport RelatedUnchangedDocumentDiagnosticReport]
type Or_DocumentDiagnosticReport struct { // line 14301
	Value interface{} `json:"value"`
}

// created for Or [FullDocumentDiagnosticReport UnchangedDocumentDiagnosticReport]
type Or_DocumentDiagnosticReportPartialResult_relatedDocuments_Value struct { // line 3896
	Value interface{} `json:"value"`
}

// created for Or [NotebookCellTextDocumentFilter TextDocumentFilter]
type Or_DocumentFilter struct { // line 14511
	Value interface{} `json:"value"`
}

// created for Or [MarkedString MarkupContent []MarkedString]
type Or_Hover_contents struct { // line 5087
	Value interface{} `json:"value"`
}

// created for Or [[]InlayHintLabelPart string]
type Or_InlayHint_label struct { // line 3732
	Value interface{} `json:"value"`
}

// created for Or [StringValue string]
type Or_InlineCompletionItem_insertText struct { // line 4164
	Value interface{} `json:"value"`
}

// created for Or [InlineValueEvaluatableExpression InlineValueText InlineValueVariableLookup]
type Or_InlineValue struct { // line 14279
	Value interface{} `json:"value"`
}

// created for Or [Msg_MarkedString string]
type Or_MarkedString struct { // line 14476
	Value interface{} `json:"value"`
}

// created for Or [NotebookDocumentFilter string]
type Or_NotebookCellTextDocumentFilter_notebook struct { // line 10473
	Value interface{} `json:"value"`
}

// created for Or [NotebookDocumentFilter string]
type Or_NotebookDocumentSyncOptions_notebookSelector_Elem_Item1_notebook struct { // line 10212
	Value interface{} `json:"value"`
}

// created for Or [FullDocumentDiagnosticReport UnchangedDocumentDiagnosticReport]
type Or_RelatedFullDocumentDiagnosticReport_relatedDocuments_Value struct { // line 7405
	Value interface{} `json:"value"`
}

// created for Or [FullDocumentDiagnosticReport UnchangedDocumentDiagnosticReport]
type Or_RelatedUnchangedDocumentDiagnosticReport_relatedDocuments_Value struct { // line 7444
	Value interface{} `json:"value"`
}

// created for Or [URI WorkspaceFolder]
type Or_RelativePattern_baseUri struct { // line 11107
	Value interface{} `json:"value"`
}

// created for Or [CodeAction Command]
type Or_Result_textDocument_codeAction_Item0_Elem struct { // line 1414
	Value interface{} `json:"value"`
}

// created for Or [InlineCompletionList []InlineCompletionItem]
type Or_Result_textDocument_inlineCompletion struct { // line 981
	Value interface{} `json:"value"`
}

// created for Or [FFullPRequests bool]
type Or_SemanticTokensClientCapabilities_requests_full struct { // line 12574
	Value interface{} `json:"value"`
}

// created for Or [FRangePRequests bool]
type Or_SemanticTokensClientCapabilities_requests_range struct { // line 12554
	Value interface{} `json:"value"`
}

// created for Or [PFullESemanticTokensOptions bool]
type Or_SemanticTokensOptions_full struct { // line 6816
	Value interface{} `json:"value"`
}

// created for Or [PRangeESemanticTokensOptions bool]
type Or_SemanticTokensOptions_range struct { // line 6796
	Value interface{} `json:"value"`
}

// created for Or [CallHierarchyOptions CallHierarchyRegistrationOptions bool]
type Or_ServerCapabilities_callHierarchyProvider struct { // line 8526
	Value interface{} `json:"value"`
}

// created for Or [CodeActionOptions bool]
type Or_ServerCapabilities_codeActionProvider struct { // line 8334
	Value interface{} `json:"value"`
}

// created for Or [DocumentColorOptions DocumentColorRegistrationOptions bool]
type Or_ServerCapabilities_colorProvider struct { // line 8370
	Value interface{} `json:"value"`
}

// created for Or [DeclarationOptions DeclarationRegistrationOptions bool]
type Or_ServerCapabilities_declarationProvider struct { // line 8196
	Value interface{} `json:"value"`
}

// created for Or [DefinitionOptions bool]
type Or_ServerCapabilities_definitionProvider struct { // line 8218
	Value interface{} `json:"value"`
}

// created for Or [DiagnosticOptions DiagnosticRegistrationOptions]
type Or_ServerCapabilities_diagnosticProvider struct { // line 8683
	Value interface{} `json:"value"`
}

// created for Or [DocumentFormattingOptions bool]
type Or_ServerCapabilities_documentFormattingProvider struct { // line 8410
	Value interface{} `json:"value"`
}

// created for Or [DocumentHighlightOptions bool]
type Or_ServerCapabilities_documentHighlightProvider struct { // line 8298
	Value interface{} `json:"value"`
}

// created for Or [DocumentRangeFormattingOptions bool]
type Or_ServerCapabilities_documentRangeFormattingProvider struct { // line 8428
	Value interface{} `json:"value"`
}

// created for Or [DocumentSymbolOptions bool]
type Or_ServerCapabilities_documentSymbolProvider struct { // line 8316
	Value interface{} `json:"value"`
}

// created for Or [FoldingRangeOptions FoldingRangeRegistrationOptions bool]
type Or_ServerCapabilities_foldingRangeProvider struct { // line 8473
	Value interface{} `json:"value"`
}

// created for Or [HoverOptions bool]
type Or_ServerCapabilities_hoverProvider struct { // line 8169
	Value interface{} `json:"value"`
}

// created for Or [ImplementationOptions ImplementationRegistrationOptions bool]
type Or_ServerCapabilities_implementationProvider struct { // line 8258
	Value interface{} `json:"value"`
}

// created for Or [InlayHintOptions InlayHintRegistrationOptions bool]
type Or_ServerCapabilities_inlayHintProvider struct { // line 8660
	Value interface{} `json:"value"`
}

// created for Or [InlineCompletionOptions bool]
type Or_ServerCapabilities_inlineCompletionProvider struct { // line 8702
	Value interface{} `json:"value"`
}

// created for Or [InlineValueOptions InlineValueRegistrationOptions bool]
type Or_ServerCapabilities_inlineValueProvider struct { // line 8637
	Value interface{} `json:"value"`
}

// created for Or [LinkedEditingRangeOptions LinkedEditingRangeRegistrationOptions bool]
type Or_ServerCapabilities_linkedEditingRangeProvider struct { // line 8549
	Value interface{} `json:"value"`
}

// created for Or [MonikerOptions MonikerRegistrationOptions bool]
type Or_ServerCapabilities_monikerProvider struct { // line 8591
	Value interface{} `json:"value"`
}

// created for Or [NotebookDocumentSyncOptions NotebookDocumentSyncRegistrationOptions]
type Or_ServerCapabilities_notebookDocumentSync struct { // line 8141
	Value interface{} `json:"value"`
}

// created for Or [ReferenceOptions bool]
type Or_ServerCapabilities_referencesProvider struct { // line 8280
	Value interface{} `json:"value"`
}

// created for Or [RenameOptions bool]
type Or_ServerCapabilities_renameProvider struct { // line 8455
	Value interface{} `json:"value"`
}

// created for Or [SelectionRangeOptions SelectionRangeRegistrationOptions bool]
type Or_ServerCapabilities_selectionRangeProvider struct { // line 8495
	Value interface{} `json:"value"`
}

// created for Or [SemanticTokensOptions SemanticTokensRegistrationOptions]
type Or_ServerCapabilities_semanticTokensProvider struct { // line 8572
	Value interface{} `json:"value"`
}

// created for Or [TextDocumentSyncKind TextDocumentSyncOptions]
type Or_ServerCapabilities_textDocumentSync struct { // line 8123
	Value interface{} `json:"value"`
}

// created for Or [TypeDefinitionOptions TypeDefinitionRegistrationOptions bool]
type Or_ServerCapabilities_typeDefinitionProvider struct { // line 8236
	Value interface{} `json:"value"`
}

// created for Or [TypeHierarchyOptions TypeHierarchyRegistrationOptions bool]
type Or_ServerCapabilities_typeHierarchyProvider struct { // line 8614
	Value interface{} `json:"value"`
}

// created for Or [WorkspaceSymbolOptions bool]
type Or_ServerCapabilities_workspaceSymbolProvider struct { // line 8392
	Value interface{} `json:"value"`
}

// created for Or [MarkupContent string]
type Or_SignatureInformation_documentation struct { // line 9160
	Value interface{} `json:"value"`
}

// created for Or [AnnotatedTextEdit TextEdit]
type Or_TextDocumentEdit_edits_Elem struct { // line 6929
	Value interface{} `json:"value"`
}

// created for Or [SaveOptions bool]
type Or_TextDocumentSyncOptions_save struct { // line 10132
	Value interface{} `json:"value"`
}

// created for Or [WorkspaceFullDocumentDiagnosticReport WorkspaceUnchangedDocumentDiagnosticReport]
type Or_WorkspaceDocumentDiagnosticReport struct { // line 14402
	Value interface{} `json:"value"`
}

// created for Or [CreateFile DeleteFile RenameFile TextDocumentEdit]
type Or_WorkspaceEdit_documentChanges_Elem struct { // line 3293
	Value interface{} `json:"value"`
}

// created for Or [Declaration []DeclarationLink]
type Or_textDocument_declaration struct { // line 249
	Value interface{} `json:"value"`
}

// created for Literal (Lit_NotebookDocumentChangeEvent_cells)
type PCellsPChange struct { // line 7717
	// Changes to the cell structure to add or
	// remove cells.
	Structure *FStructurePCells `json:"structure,omitempty"`
	// Changes to notebook cells properties like its
	// kind, execution summary or metadata.
	Data []NotebookCell `json:"data,omitempty"`
	// Changes to the text content of notebook cells.
	TextContent []Lit_NotebookDocumentChangeEvent_cells_textContent_Elem `json:"textContent,omitempty"`
}

// created for Literal (Lit_WorkspaceEditClientCapabilities_changeAnnotationSupport)
type PChangeAnnotationSupportPWorkspaceEdit struct { // line 11181
	// Whether the client groups edits with equal labels into tree nodes,
	// for instance all edits labelled with "Changes in Strings" would
	// be a tree node.
	GroupsOnLabel bool `json:"groupsOnLabel,omitempty"`
}

// created for Literal (Lit_CodeActionClientCapabilities_codeActionLiteralSupport)
type PCodeActionLiteralSupportPCodeAction struct { // line 12101
	// The code action kind is support with the following value
	// set.
	CodeActionKind FCodeActionKindPCodeActionLiteralSupport `json:"codeActionKind"`
}

// created for Literal (Lit_CompletionClientCapabilities_completionItemKind)
type PCompletionItemKindPCompletion struct { // line 11699
	// The completion item kind values the client supports. When this
	// property exists the client also guarantees that it will
	// handle values outside its set gracefully and falls back
	// to a default value when unknown.
	//
	// If this property is not present the client only supports
	// the completion items kinds from `Text` to `Reference` as defined in
	// the initial version of the protocol.
	ValueSet []CompletionItemKind `json:"valueSet,omitempty"`
}

// created for Literal (Lit_CompletionClientCapabilities_completionItem)
type PCompletionItemPCompletion struct { // line 11548
	// Client supports snippets as insert text.
	//
	// A snippet can define tab stops and placeholders with `$1`, `$2`
	// and `${3:foo}`. `$0` defines the final tab stop, it defaults to
	// the end of the snippet. Placeholders with equal identifiers are linked,
	// that is typing in one will update others too.
	SnippetSupport bool `json:"snippetSupport,omitempty"`
	// Client supports commit characters on a completion item.
	CommitCharactersSupport bool `json:"commitCharactersSupport,omitempty"`
	// Client supports the following content formats for the documentation
	// property. The order describes the preferred format of the client.
	DocumentationFormat []MarkupKind `json:"documentationFormat,omitempty"`
	// Client supports the deprecated property on a completion item.
	DeprecatedSupport bool `json:"deprecatedSupport,omitempty"`
	// Client supports the preselect property on a completion item.
	PreselectSupport bool `json:"preselectSupport,omitempty"`
	// Client supports the tag property on a completion item. Clients supporting
	// tags have to handle unknown tags gracefully. Clients especially need to
	// preserve unknown tags when sending a completion item back to the server in
	// a resolve call.
	//
	// @since 3.15.0
	TagSupport FTagSupportPCompletionItem `json:"tagSupport"`
	// Client support insert replace edit to control different behavior if a
	// completion item is inserted in the text or should replace text.
	//
	// @since 3.16.0
	InsertReplaceSupport bool `json:"insertReplaceSupport,omitempty"`
	// Indicates which properties a client can resolve lazily on a completion
	// item. Before version 3.16.0 only the predefined properties `documentation`
	// and `details` could be resolved lazily.
	//
	// @since 3.16.0
	ResolveSupport *FResolveSupportPCompletionItem `json:"resolveSupport,omitempty"`
	// The client supports the `insertTextMode` property on
	// a completion item to override the whitespace handling mode
	// as defined by the client (see `insertTextMode`).
	//
	// @since 3.16.0
	InsertTextModeSupport *FInsertTextModeSupportPCompletionItem `json:"insertTextModeSupport,omitempty"`
	// The client has support for completion item label
	// details (see also `CompletionItemLabelDetails`).
	//
	// @since 3.17.0
	LabelDetailsSupport bool `json:"labelDetailsSupport,omitempty"`
}

// created for Literal (Lit_CompletionOptions_completionItem)
type PCompletionItemPCompletionProvider struct { // line 9065
	// The server has support for completion item label
	// details (see also `CompletionItemLabelDetails`) when
	// receiving a completion item in a resolve call.
	//
	// @since 3.17.0
	LabelDetailsSupport bool `json:"labelDetailsSupport,omitempty"`
}

// created for Literal (Lit_CompletionClientCapabilities_completionList)
type PCompletionListPCompletion struct { // line 11741
	// The client supports the following itemDefaults on
	// a completion list.
	//
	// The value lists the supported property names of the
	// `CompletionList.itemDefaults` object. If omitted
	// no properties are supported.
	//
	// @since 3.17.0
	ItemDefaults []string `json:"itemDefaults,omitempty"`
}

// created for Literal (Lit_CodeAction_disabled)
type PDisabledMsg_textDocument_codeAction struct { // line 5622
	// Human readable description of why the code action is currently disabled.
	//
	// This is displayed in the code actions UI.
	Reason string `json:"reason"`
}

// created for Literal (Lit_FoldingRangeClientCapabilities_foldingRangeKind)
type PFoldingRangeKindPFoldingRange struct { // line 12387
	// The folding range kind values the client supports. When this
	// property exists the client also guarantees that it will
	// handle values outside its set gracefully and falls back
	// to a default value when unknown.
	ValueSet []FoldingRangeKind `json:"valueSet,omitempty"`
}

// created for Literal (Lit_FoldingRangeClientCapabilities_foldingRange)
type PFoldingRangePFoldingRange struct { // line 12412
	// If set, the client signals that it supports setting collapsedText on
	// folding ranges to display custom labels instead of the default text.
	//
	// @since 3.17.0
	CollapsedText bool `json:"collapsedText,omitempty"`
}

// created for Literal (Lit_SemanticTokensOptions_full_Item1)
type PFullESemanticTokensOptions struct { // line 6823
	// The server supports deltas for full documents.
	Delta bool `json:"delta"`
}

// created for Literal (Lit_CompletionList_itemDefaults)
type PItemDefaultsMsg_textDocument_completion struct { // line 4946
	// A default commit character set.
	//
	// @since 3.17.0
	CommitCharacters []string `json:"commitCharacters,omitempty"`
	// A default edit range.
	//
	// @since 3.17.0
	EditRange *OrFEditRangePItemDefaults `json:"editRange,omitempty"`
	// A default insert text format.
	//
	// @since 3.17.0
	InsertTextFormat *InsertTextFormat `json:"insertTextFormat,omitempty"`
	// A default insert text mode.
	//
	// @since 3.17.0
	InsertTextMode *InsertTextMode `json:"insertTextMode,omitempty"`
	// A default data value.
	//
	// @since 3.17.0
	Data interface{} `json:"data,omitempty"`
}

// created for Literal (Lit_WorkspaceSymbol_location_Item1)
type PLocationMsg_workspace_symbol struct { // line 5723
	URI DocumentURI `json:"uri"`
}

// created for Literal (Lit_ShowMessageRequestClientCapabilities_messageActionItem)
type PMessageActionItemPShowMessage struct { // line 12857
	// Whether the client supports additional attributes which
	// are preserved and send back to the server in the
	// request's response.
	AdditionalPropertiesSupport bool `json:"additionalPropertiesSupport,omitempty"`
}

// created for Literal (Lit_NotebookDocumentSyncOptions_notebookSelector_Elem_Item0)
type PNotebookSelectorPNotebookDocumentSync struct { // line 10160
	// The notebook to be synced If a string
	// value is provided it matches against the
	// notebook type. '*' matches every notebook.
	Notebook OrFNotebookPNotebookSelector `json:"notebook"`
	// The cells of the matching notebook to be synced.
	Cells []Lit_NotebookDocumentSyncOptions_notebookSelector_Elem_Item0_cells_Elem `json:"cells,omitempty"`
}

// created for Literal (Lit_SemanticTokensOptions_range_Item1)
type PRangeESemanticTokensOptions struct { // line 6803
}

// created for Literal (Lit_SemanticTokensClientCapabilities_requests)
type PRequestsPSemanticTokens struct { // line 12548
	// The client will send the `textDocument/semanticTokens/range` request if
	// the server provides a corresponding handler.
	Range Or_SemanticTokensClientCapabilities_requests_range `json:"range"`
	// The client will send the `textDocument/semanticTokens/full` request if
	// the server provides a corresponding handler.
	Full Or_SemanticTokensClientCapabilities_requests_full `json:"full"`
}

// created for Literal (Lit_CodeActionClientCapabilities_resolveSupport)
type PResolveSupportPCodeAction struct { // line 12166
	// The properties that a client can resolve lazily.
	Properties []string `json:"properties"`
}

// created for Literal (Lit_InlayHintClientCapabilities_resolveSupport)
type PResolveSupportPInlayHint struct { // line 12760
	// The properties that a client can resolve lazily.
	Properties []string `json:"properties"`
}

// created for Literal (Lit_WorkspaceSymbolClientCapabilities_resolveSupport)
type PResolveSupportPSymbol struct { // line 11303
	// The properties that a client can resolve lazily. Usually
	// `location.range`
	Properties []string `json:"properties"`
}

// created for Literal (Lit_InitializeResult_serverInfo)
type PServerInfoMsg_initialize struct { // line 4291
	// The name of the server as defined by the server.
	Name string `json:"name"`
	// The server's version as defined by the server.
	Version string `json:"version,omitempty"`
}

// created for Literal (Lit_SignatureHelpClientCapabilities_signatureInformation)
type PSignatureInformationPSignatureHelp struct { // line 11808
	// Client supports the following content formats for the documentation
	// property. The order describes the preferred format of the client.
	DocumentationFormat []MarkupKind `json:"documentationFormat,omitempty"`
	// Client capabilities specific to parameter information.
	ParameterInformation *FParameterInformationPSignatureInformation `json:"parameterInformation,omitempty"`
	// The client supports the `activeParameter` property on `SignatureInformation`
	// literal.
	//
	// @since 3.16.0
	ActiveParameterSupport bool `json:"activeParameterSupport,omitempty"`
}

// created for Literal (Lit_GeneralClientCapabilities_staleRequestSupport)
type PStaleRequestSupportPGeneral struct { // line 11035
	// The client will actively cancel the request.
	Cancel bool `json:"cancel"`
	// The list of requests for which the client
	// will retry the request if it receives a
	// response with error code `ContentModified`
	RetryOnContentModified []string `json:"retryOnContentModified"`
}

// created for Literal (Lit_DocumentSymbolClientCapabilities_symbolKind)
type PSymbolKindPDocumentSymbol struct { // line 12019
	// The symbol kind values the client supports. When this
	// property exists the client also guarantees that it will
	// handle values outside its set gracefully and falls back
	// to a default value when unknown.
	//
	// If this property is not present the client only supports
	// the symbol kinds from `File` to `Array` as defined in
	// the initial version of the protocol.
	ValueSet []SymbolKind `json:"valueSet,omitempty"`
}

// created for Literal (Lit_WorkspaceSymbolClientCapabilities_symbolKind)
type PSymbolKindPSymbol struct { // line 11255
	// The symbol kind values the client supports. When this
	// property exists the client also guarantees that it will
	// handle values outside its set gracefully and falls back
	// to a default value when unknown.
	//
	// If this property is not present the client only supports
	// the symbol kinds from `File` to `Array` as defined in
	// the initial version of the protocol.
	ValueSet []SymbolKind `json:"valueSet,omitempty"`
}

// created for Literal (Lit_DocumentSymbolClientCapabilities_tagSupport)
type PTagSupportPDocumentSymbol struct { // line 12052
	// The tags supported by the client.
	ValueSet []SymbolTag `json:"valueSet"`
}

// created for Literal (Lit_PublishDiagnosticsClientCapabilities_tagSupport)
type PTagSupportPPublishDiagnostics struct { // line 12463
	// The tags supported by the client.
	ValueSet []DiagnosticTag `json:"valueSet"`
}

// created for Literal (Lit_WorkspaceSymbolClientCapabilities_tagSupport)
type PTagSupportPSymbol struct { // line 11279
	// The tags supported by the client.
	ValueSet []SymbolTag `json:"valueSet"`
}

// The parameters of a configuration request.
type ParamConfiguration struct { // line 2272
	Items []ConfigurationItem `json:"items"`
}
type ParamInitialize struct { // line 4263
	XInitializeParams
	WorkspaceFoldersInitializeParams
}

// Represents a parameter of a callable-signature. A parameter can
// have a label and a doc-comment.
type ParameterInformation struct { // line 10417
	// The label of this parameter information.
	//
	// Either a string or an inclusive start and exclusive end offsets within its containing
	// signature label. (see SignatureInformation.label). The offsets are based on a UTF-16
	// string representation as `Position` and `Range` does.
	//
	// *Note*: a label of type string should be a substring of its containing signature label.
	// Its intended use case is to highlight the parameter label part in the `SignatureInformation.label`.
	Label string `json:"label"`
	// The human-readable doc-comment of this parameter. Will be shown
	// in the UI but can be omitted.
	Documentation string `json:"documentation,omitempty"`
}
type PartialResultParams struct { // line 6494
	// An optional token that a server can use to report partial results (e.g. streaming) to
	// the client.
	PartialResultToken *ProgressToken `json:"partialResultToken,omitempty"`
}

// The glob pattern to watch relative to the base path. Glob patterns can have the following syntax:
//
//   - `*` to match one or more characters in a path segment
//   - `?` to match on one character in a path segment
//   - `**` to match any number of path segments, including none
//   - `{}` to group conditions (e.g. `**​/*.{ts,js}` matches all TypeScript and JavaScript files)
//   - `[]` to declare a range of characters to match in a path segment (e.g., `example.[0-9]` to match on `example.0`, `example.1`, …)
//   - `[!...]` to negate a range of characters to match in a path segment (e.g., `example.[!0-9]` to match on `example.a`, `example.b`, but not `example.0`)
//
// @since 3.17.0
type Pattern = string // (alias) line 14778
// Position in a text document expressed as zero-based line and character
// offset. Prior to 3.17 the offsets were always based on a UTF-16 string
// representation. So a string of the form `a𐐀b` the character offset of the
// character `a` is 0, the character offset of `𐐀` is 1 and the character
// offset of b is 3 since `𐐀` is represented using two code units in UTF-16.
// Since 3.17 clients and servers can agree on a different string encoding
// representation (e.g. UTF-8). The client announces it's supported encoding
// via the client capability [`general.positionEncodings`](#clientCapabilities).
// The value is an array of position encodings the client supports, with
// decreasing preference (e.g. the encoding at index `0` is the most preferred
// one). To stay backwards compatible the only mandatory encoding is UTF-16
// represented via the string `utf-16`. The server can pick one of the
// encodings offered by the client and signals that encoding back to the
// client via the initialize result's property
// [`capabilities.positionEncoding`](#serverCapabilities). If the string value
// `utf-16` is missing from the client's capability `general.positionEncodings`
// servers can safely assume that the client supports UTF-16. If the server
// omits the position encoding in its initialize result the encoding defaults
// to the string value `utf-16`. Implementation considerations: since the
// conversion from one encoding into another requires the content of the
// file / line the conversion is best done where the file is read which is
// usually on the server side.
//
// Positions are line end character agnostic. So you can not specify a position
// that denotes `\r|\n` or `\n|` where `|` represents the character offset.
//
// @since 3.17.0 - support for negotiated position encoding.
type Position struct { // line 6737
	// Line position in a document (zero-based).
	//
	// If a line number is greater than the number of lines in a document, it defaults back to the number of lines in the document.
	// If a line number is negative, it defaults to 0.
	Line uint32 `json:"line"`
	// Character offset on a line in a document (zero-based).
	//
	// The meaning of this offset is determined by the negotiated
	// `PositionEncodingKind`.
	//
	// If the character value is greater than the line length it defaults back to the
	// line length.
	Character uint32 `json:"character"`
}

// A set of predefined position encoding kinds.
//
// @since 3.17.0
type PositionEncodingKind string             // line 13842
type PrepareRename2Gn = Msg_PrepareRename2Gn // (alias) line 13927
type PrepareRenameParams struct {            // line 6161
	TextDocumentPositionParams
	WorkDoneProgressParams
}
type PrepareRenameResult = Msg_PrepareRename2Gn // (alias) line 13927
type PrepareSupportDefaultBehavior uint32       // line 14137
// A previous result id in a workspace pull request.
//
// @since 3.17.0
type PreviousResultID struct { // line 7567
	// The URI for which the client knowns a
	// result id.
	URI DocumentURI `json:"uri"`
	// The value of the previous result id.
	Value string `json:"value"`
}

// A previous result id in a workspace pull request.
//
// @since 3.17.0
type PreviousResultId struct { // line 7567
	// The URI for which the client knowns a
	// result id.
	URI DocumentURI `json:"uri"`
	// The value of the previous result id.
	Value string `json:"value"`
}
type ProgressParams struct { // line 6437
	// The progress token provided by the client or server.
	Token ProgressToken `json:"token"`
	// The progress data.
	Value interface{} `json:"value"`
}
type ProgressToken = interface{} // (alias) line 14375
// The publish diagnostic client capabilities.
type PublishDiagnosticsClientCapabilities struct { // line 12448
	// Whether the clients accepts diagnostics with related information.
	RelatedInformation bool `json:"relatedInformation,omitempty"`
	// Client supports the tag property to provide meta data about a diagnostic.
	// Clients supporting tags have to handle unknown tags gracefully.
	//
	// @since 3.15.0
	TagSupport *PTagSupportPPublishDiagnostics `json:"tagSupport,omitempty"`
	// Whether the client interprets the version property of the
	// `textDocument/publishDiagnostics` notification's parameter.
	//
	// @since 3.15.0
	VersionSupport bool `json:"versionSupport,omitempty"`
	// Client supports a codeDescription property
	//
	// @since 3.16.0
	CodeDescriptionSupport bool `json:"codeDescriptionSupport,omitempty"`
	// Whether code action supports the `data` property which is
	// preserved between a `textDocument/publishDiagnostics` and
	// `textDocument/codeAction` request.
	//
	// @since 3.16.0
	DataSupport bool `json:"dataSupport,omitempty"`
}

// The publish diagnostic notification's parameters.
type PublishDiagnosticsParams struct { // line 4657
	// The URI for which diagnostic information is reported.
	URI DocumentURI `json:"uri"`
	// Optional the version number of the document the diagnostics are published for.
	//
	// @since 3.15.0
	Version int32 `json:"version,omitempty"`
	// An array of diagnostic information items.
	Diagnostics []Diagnostic `json:"diagnostics"`
}

// A range in a text document expressed as (zero-based) start and end positions.
//
// If you want to specify a range that contains a line including the line ending
// character(s) then use an end position denoting the start of the next line.
// For example:
// ```ts
//
//	{
//	    start: { line: 5, character: 23 }
//	    end : { line 6, character : 0 }
//	}
//
// ```
type Range struct { // line 6547
	// The range's start position.
	Start Position `json:"start"`
	// The range's end position.
	End Position `json:"end"`
}

// Client Capabilities for a {@link ReferencesRequest}.
type ReferenceClientCapabilities struct { // line 11974
	// Whether references supports dynamic registration.
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
}

// Value-object that contains additional information when
// requesting references.
type ReferenceContext struct { // line 9248
	// Include the declaration of the current symbol.
	IncludeDeclaration bool `json:"includeDeclaration"`
}

// Reference options.
type ReferenceOptions struct { // line 9262
	WorkDoneProgressOptions
}

// Parameters for a {@link ReferencesRequest}.
type ReferenceParams struct { // line 5249
	Context ReferenceContext `json:"context"`
	TextDocumentPositionParams
	WorkDoneProgressParams
	PartialResultParams
}

// Registration options for a {@link ReferencesRequest}.
type ReferenceRegistrationOptions struct { // line 5278
	TextDocumentRegistrationOptions
	ReferenceOptions
}

// General parameters to register for a notification or to register a provider.
type Registration struct { // line 7895
	// The id used to register the request. The id can be used to deregister
	// the request again.
	ID string `json:"id"`
	// The method / capability to register for.
	Method string `json:"method"`
	// Options necessary for the registration.
	RegisterOptions interface{} `json:"registerOptions,omitempty"`
}
type RegistrationParams struct { // line 4233
	Registrations []Registration `json:"registrations"`
}

// Client capabilities specific to regular expressions.
//
// @since 3.16.0
type RegularExpressionsClientCapabilities struct { // line 12893
	// The engine's name.
	Engine string `json:"engine"`
	// The engine's version.
	Version string `json:"version,omitempty"`
}

// A full diagnostic report with a set of related documents.
//
// @since 3.17.0
type RelatedFullDocumentDiagnosticReport struct { // line 7393
	// Diagnostics of related documents. This information is useful
	// in programming languages where code in a file A can generate
	// diagnostics in a file B which A depends on. An example of
	// such a language is C/C++ where marco definitions in a file
	// a.cpp and result in errors in a header file b.hpp.
	//
	// @since 3.17.0
	RelatedDocuments map[DocumentURI]interface{} `json:"relatedDocuments,omitempty"`
	FullDocumentDiagnosticReport
}

// An unchanged diagnostic report with a set of related documents.
//
// @since 3.17.0
type RelatedUnchangedDocumentDiagnosticReport struct { // line 7432
	// Diagnostics of related documents. This information is useful
	// in programming languages where code in a file A can generate
	// diagnostics in a file B which A depends on. An example of
	// such a language is C/C++ where marco definitions in a file
	// a.cpp and result in errors in a header file b.hpp.
	//
	// @since 3.17.0
	RelatedDocuments map[DocumentURI]interface{} `json:"relatedDocuments,omitempty"`
	UnchangedDocumentDiagnosticReport
}

// A relative pattern is a helper to construct glob patterns that are matched
// relatively to a base URI. The common value for a `baseUri` is a workspace
// folder root, but it can be another absolute URI as well.
//
// @since 3.17.0
type RelativePattern struct { // line 11101
	// A workspace folder or a base URI to which this pattern will be matched
	// against relatively.
	BaseURI Or_RelativePattern_baseUri `json:"baseUri"`
	// The actual glob pattern;
	Pattern Pattern `json:"pattern"`
}
type RenameClientCapabilities struct { // line 12310
	// Whether rename supports dynamic registration.
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
	// Client supports testing for validity of rename operations
	// before execution.
	//
	// @since 3.12.0
	PrepareSupport bool `json:"prepareSupport,omitempty"`
	// Client supports the default behavior result.
	//
	// The value indicates the default behavior used by the
	// client.
	//
	// @since 3.16.0
	PrepareSupportDefaultBehavior *PrepareSupportDefaultBehavior `json:"prepareSupportDefaultBehavior,omitempty"`
	// Whether the client honors the change annotations in
	// text edits and resource operations returned via the
	// rename request's workspace edit by for example presenting
	// the workspace edit in the user interface and asking
	// for confirmation.
	//
	// @since 3.16.0
	HonorsChangeAnnotations bool `json:"honorsChangeAnnotations,omitempty"`
}

// Rename file operation
type RenameFile struct { // line 6985
	// A rename
	Kind string `json:"kind"`
	// The old (existing) location.
	OldURI DocumentURI `json:"oldUri"`
	// The new location.
	NewURI DocumentURI `json:"newUri"`
	// Rename options.
	Options *RenameFileOptions `json:"options,omitempty"`
	ResourceOperation
}

// Rename file options
type RenameFileOptions struct { // line 9771
	// Overwrite target if existing. Overwrite wins over `ignoreIfExists`
	Overwrite bool `json:"overwrite,omitempty"`
	// Ignores if target exists.
	IgnoreIfExists bool `json:"ignoreIfExists,omitempty"`
}

// The parameters sent in notifications/requests for user-initiated renames of
// files.
//
// @since 3.16.0
type RenameFilesParams struct { // line 3355
	// An array of all files/folders renamed in this operation. When a folder is renamed, only
	// the folder will be included, and not its children.
	Files []FileRename `json:"files"`
}

// Provider options for a {@link RenameRequest}.
type RenameOptions struct { // line 9599
	// Renames should be checked and tested before being executed.
	//
	// @since version 3.12.0
	PrepareProvider bool `json:"prepareProvider,omitempty"`
	WorkDoneProgressOptions
}

// The parameters of a {@link RenameRequest}.
type RenameParams struct { // line 6110
	// The document to rename.
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	// The position at which this request was sent.
	Position Position `json:"position"`
	// The new name of the symbol. If the given name is not valid the
	// request must return a {@link ResponseError} with an
	// appropriate message set.
	NewName string `json:"newName"`
	WorkDoneProgressParams
}

// Registration options for a {@link RenameRequest}.
type RenameRegistrationOptions struct { // line 6146
	TextDocumentRegistrationOptions
	RenameOptions
}

// A generic resource operation.
type ResourceOperation struct { // line 9723
	// The resource operation kind.
	Kind string `json:"kind"`
	// An optional annotation identifier describing the operation.
	//
	// @since 3.16.0
	AnnotationID *ChangeAnnotationIdentifier `json:"annotationId,omitempty"`
}
type ResourceOperationKind string // line 14084
// Save options.
type SaveOptions struct { // line 8783
	// The client is supposed to include the content on save.
	IncludeText bool `json:"includeText,omitempty"`
}

// Describes the currently selected completion item.
//
// @since 3.18.0
// @proposed
type SelectedCompletionInfo struct { // line 10004
	// The range that will be replaced if this completion item is accepted.
	Range Range `json:"range"`
	// The text the range will be replaced with if this completion is accepted.
	Text string `json:"text"`
}

// A selection range represents a part of a selection hierarchy. A selection range
// may have a parent selection range that contains it.
type SelectionRange struct { // line 2642
	// The {@link Range range} of this selection range.
	Range Range `json:"range"`
	// The parent selection range containing this range. Therefore `parent.range` must contain `this.range`.
	Parent *SelectionRange `json:"parent,omitempty"`
}
type SelectionRangeClientCapabilities struct { // line 12434
	// Whether implementation supports dynamic registration for selection range providers. If this is set to `true`
	// the client supports the new `SelectionRangeRegistrationOptions` return value for the corresponding server
	// capability as well.
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
}
type SelectionRangeOptions struct { // line 6760
	WorkDoneProgressOptions
}

// A parameter literal used in selection range requests.
type SelectionRangeParams struct { // line 2607
	// The text document.
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	// The positions inside the text document.
	Positions []Position `json:"positions"`
	WorkDoneProgressParams
	PartialResultParams
}
type SelectionRangeRegistrationOptions struct { // line 2665
	SelectionRangeOptions
	TextDocumentRegistrationOptions
	StaticRegistrationOptions
}

// A set of predefined token modifiers. This set is not fixed
// an clients can specify additional token types via the
// corresponding client capabilities.
//
// @since 3.16.0
type SemanticTokenModifiers string // line 13063
// A set of predefined token types. This set is not fixed
// an clients can specify additional token types via the
// corresponding client capabilities.
//
// @since 3.16.0
type SemanticTokenTypes string // line 12956
// @since 3.16.0
type SemanticTokens struct { // line 2953
	// An optional result id. If provided and clients support delta updating
	// the client will include the result id in the next semantic token request.
	// A server can then instead of computing all semantic tokens again simply
	// send a delta.
	ResultID string `json:"resultId,omitempty"`
	// The actual tokens.
	Data []uint32 `json:"data"`
}

// @since 3.16.0
type SemanticTokensClientCapabilities struct { // line 12533
	// Whether implementation supports dynamic registration. If this is set to `true`
	// the client supports the new `(TextDocumentRegistrationOptions & StaticRegistrationOptions)`
	// return value for the corresponding server capability as well.
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
	// Which requests the client supports and might send to the server
	// depending on the server's capability. Please note that clients might not
	// show semantic tokens or degrade some of the user experience if a range
	// or full request is advertised by the client but not provided by the
	// server. If for example the client capability `requests.full` and
	// `request.range` are both set to true but the server only provides a
	// range provider the client might not render a minimap correctly or might
	// even decide to not show any semantic tokens at all.
	Requests PRequestsPSemanticTokens `json:"requests"`
	// The token types that the client supports.
	TokenTypes []string `json:"tokenTypes"`
	// The token modifiers that the client supports.
	TokenModifiers []string `json:"tokenModifiers"`
	// The token formats the clients supports.
	Formats []TokenFormat `json:"formats"`
	// Whether the client supports tokens that can overlap each other.
	OverlappingTokenSupport bool `json:"overlappingTokenSupport,omitempty"`
	// Whether the client supports tokens that can span multiple lines.
	MultilineTokenSupport bool `json:"multilineTokenSupport,omitempty"`
	// Whether the client allows the server to actively cancel a
	// semantic token request, e.g. supports returning
	// LSPErrorCodes.ServerCancelled. If a server does the client
	// needs to retrigger the request.
	//
	// @since 3.17.0
	ServerCancelSupport bool `json:"serverCancelSupport,omitempty"`
	// Whether the client uses semantic tokens to augment existing
	// syntax tokens. If set to `true` client side created syntax
	// tokens and semantic tokens are both used for colorization. If
	// set to `false` the client only uses the returned semantic tokens
	// for colorization.
	//
	// If the value is `undefined` then the client behavior is not
	// specified.
	//
	// @since 3.17.0
	AugmentsSyntaxTokens bool `json:"augmentsSyntaxTokens,omitempty"`
}

// @since 3.16.0
type SemanticTokensDelta struct { // line 3052
	ResultID string `json:"resultId,omitempty"`
	// The semantic token edits to transform a previous result into a new result.
	Edits []SemanticTokensEdit `json:"edits"`
}

// @since 3.16.0
type SemanticTokensDeltaParams struct { // line 3019
	// The text document.
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	// The result id of a previous response. The result Id can either point to a full response
	// or a delta response depending on what was received last.
	PreviousResultID string `json:"previousResultId"`
	WorkDoneProgressParams
	PartialResultParams
}

// @since 3.16.0
type SemanticTokensDeltaPartialResult struct { // line 3078
	Edits []SemanticTokensEdit `json:"edits"`
}

// @since 3.16.0
type SemanticTokensEdit struct { // line 6853
	// The start offset of the edit.
	Start uint32 `json:"start"`
	// The count of elements to remove.
	DeleteCount uint32 `json:"deleteCount"`
	// The elements to insert.
	Data []uint32 `json:"data,omitempty"`
}

// @since 3.16.0
type SemanticTokensLegend struct { // line 9644
	// The token types a server uses.
	TokenTypes []string `json:"tokenTypes"`
	// The token modifiers a server uses.
	TokenModifiers []string `json:"tokenModifiers"`
}

// @since 3.16.0
type SemanticTokensOptions struct { // line 6782
	// The legend used by the server
	Legend SemanticTokensLegend `json:"legend"`
	// Server supports providing semantic tokens for a specific range
	// of a document.
	Range *Or_SemanticTokensOptions_range `json:"range,omitempty"`
	// Server supports providing semantic tokens for a full document.
	Full *Or_SemanticTokensOptions_full `json:"full,omitempty"`
	WorkDoneProgressOptions
}

// @since 3.16.0
type SemanticTokensParams struct { // line 2928
	// The text document.
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	WorkDoneProgressParams
	PartialResultParams
}

// @since 3.16.0
type SemanticTokensPartialResult struct { // line 2980
	Data []uint32 `json:"data"`
}

// @since 3.16.0
type SemanticTokensRangeParams struct { // line 3095
	// The text document.
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	// The range the semantic tokens are requested for.
	Range Range `json:"range"`
	WorkDoneProgressParams
	PartialResultParams
}

// @since 3.16.0
type SemanticTokensRegistrationOptions struct { // line 2997
	TextDocumentRegistrationOptions
	SemanticTokensOptions
	StaticRegistrationOptions
}

// @since 3.16.0
type SemanticTokensWorkspaceClientCapabilities struct { // line 11342
	// Whether the client implementation supports a refresh request sent from
	// the server to the client.
	//
	// Note that this event is global and will force the client to refresh all
	// semantic tokens currently shown. It should be used with absolute care
	// and is useful for situation where a server for example detects a project
	// wide change that requires such a calculation.
	RefreshSupport bool `json:"refreshSupport,omitempty"`
}

// Defines the capabilities provided by a language
// server.
type ServerCapabilities struct { // line 8107
	// The position encoding the server picked from the encodings offered
	// by the client via the client capability `general.positionEncodings`.
	//
	// If the client didn't provide any position encodings the only valid
	// value that a server can return is 'utf-16'.
	//
	// If omitted it defaults to 'utf-16'.
	//
	// @since 3.17.0
	PositionEncoding *PositionEncodingKind `json:"positionEncoding,omitempty"`
	// Defines how text documents are synced. Is either a detailed structure
	// defining each notification or for backwards compatibility the
	// TextDocumentSyncKind number.
	TextDocumentSync interface{} `json:"textDocumentSync,omitempty"`
	// Defines how notebook documents are synced.
	//
	// @since 3.17.0
	NotebookDocumentSync *Or_ServerCapabilities_notebookDocumentSync `json:"notebookDocumentSync,omitempty"`
	// The server provides completion support.
	CompletionProvider *CompletionOptions `json:"completionProvider,omitempty"`
	// The server provides hover support.
	HoverProvider *Or_ServerCapabilities_hoverProvider `json:"hoverProvider,omitempty"`
	// The server provides signature help support.
	SignatureHelpProvider *SignatureHelpOptions `json:"signatureHelpProvider,omitempty"`
	// The server provides Goto Declaration support.
	DeclarationProvider *Or_ServerCapabilities_declarationProvider `json:"declarationProvider,omitempty"`
	// The server provides goto definition support.
	DefinitionProvider *Or_ServerCapabilities_definitionProvider `json:"definitionProvider,omitempty"`
	// The server provides Goto Type Definition support.
	TypeDefinitionProvider *Or_ServerCapabilities_typeDefinitionProvider `json:"typeDefinitionProvider,omitempty"`
	// The server provides Goto Implementation support.
	ImplementationProvider *Or_ServerCapabilities_implementationProvider `json:"implementationProvider,omitempty"`
	// The server provides find references support.
	ReferencesProvider *Or_ServerCapabilities_referencesProvider `json:"referencesProvider,omitempty"`
	// The server provides document highlight support.
	DocumentHighlightProvider *Or_ServerCapabilities_documentHighlightProvider `json:"documentHighlightProvider,omitempty"`
	// The server provides document symbol support.
	DocumentSymbolProvider *Or_ServerCapabilities_documentSymbolProvider `json:"documentSymbolProvider,omitempty"`
	// The server provides code actions. CodeActionOptions may only be
	// specified if the client states that it supports
	// `codeActionLiteralSupport` in its initial `initialize` request.
	CodeActionProvider interface{} `json:"codeActionProvider,omitempty"`
	// The server provides code lens.
	CodeLensProvider *CodeLensOptions `json:"codeLensProvider,omitempty"`
	// The server provides document link support.
	DocumentLinkProvider *DocumentLinkOptions `json:"documentLinkProvider,omitempty"`
	// The server provides color provider support.
	ColorProvider *Or_ServerCapabilities_colorProvider `json:"colorProvider,omitempty"`
	// The server provides workspace symbol support.
	WorkspaceSymbolProvider *Or_ServerCapabilities_workspaceSymbolProvider `json:"workspaceSymbolProvider,omitempty"`
	// The server provides document formatting.
	DocumentFormattingProvider *Or_ServerCapabilities_documentFormattingProvider `json:"documentFormattingProvider,omitempty"`
	// The server provides document range formatting.
	DocumentRangeFormattingProvider *Or_ServerCapabilities_documentRangeFormattingProvider `json:"documentRangeFormattingProvider,omitempty"`
	// The server provides document formatting on typing.
	DocumentOnTypeFormattingProvider *DocumentOnTypeFormattingOptions `json:"documentOnTypeFormattingProvider,omitempty"`
	// The server provides rename support. RenameOptions may only be
	// specified if the client states that it supports
	// `prepareSupport` in its initial `initialize` request.
	RenameProvider interface{} `json:"renameProvider,omitempty"`
	// The server provides folding provider support.
	FoldingRangeProvider *Or_ServerCapabilities_foldingRangeProvider `json:"foldingRangeProvider,omitempty"`
	// The server provides selection range support.
	SelectionRangeProvider *Or_ServerCapabilities_selectionRangeProvider `json:"selectionRangeProvider,omitempty"`
	// The server provides execute command support.
	ExecuteCommandProvider *ExecuteCommandOptions `json:"executeCommandProvider,omitempty"`
	// The server provides call hierarchy support.
	//
	// @since 3.16.0
	CallHierarchyProvider *Or_ServerCapabilities_callHierarchyProvider `json:"callHierarchyProvider,omitempty"`
	// The server provides linked editing range support.
	//
	// @since 3.16.0
	LinkedEditingRangeProvider *Or_ServerCapabilities_linkedEditingRangeProvider `json:"linkedEditingRangeProvider,omitempty"`
	// The server provides semantic tokens support.
	//
	// @since 3.16.0
	SemanticTokensProvider interface{} `json:"semanticTokensProvider,omitempty"`
	// The server provides moniker support.
	//
	// @since 3.16.0
	MonikerProvider *Or_ServerCapabilities_monikerProvider `json:"monikerProvider,omitempty"`
	// The server provides type hierarchy support.
	//
	// @since 3.17.0
	TypeHierarchyProvider *Or_ServerCapabilities_typeHierarchyProvider `json:"typeHierarchyProvider,omitempty"`
	// The server provides inline values.
	//
	// @since 3.17.0
	InlineValueProvider *Or_ServerCapabilities_inlineValueProvider `json:"inlineValueProvider,omitempty"`
	// The server provides inlay hints.
	//
	// @since 3.17.0
	InlayHintProvider interface{} `json:"inlayHintProvider,omitempty"`
	// The server has support for pull model diagnostics.
	//
	// @since 3.17.0
	DiagnosticProvider *Or_ServerCapabilities_diagnosticProvider `json:"diagnosticProvider,omitempty"`
	// Inline completion options used during static registration.
	//
	// @since 3.18.0
	// @proposed
	InlineCompletionProvider *Or_ServerCapabilities_inlineCompletionProvider `json:"inlineCompletionProvider,omitempty"`
	// Workspace specific server capabilities.
	Workspace *Workspace6Gn `json:"workspace,omitempty"`
	// Experimental server capabilities.
	Experimental interface{} `json:"experimental,omitempty"`
}
type SetTraceParams struct { // line 6383
	Value TraceValues `json:"value"`
}

// Client capabilities for the showDocument request.
//
// @since 3.16.0
type ShowDocumentClientCapabilities struct { // line 12878
	// The client has support for the showDocument
	// request.
	Support bool `json:"support"`
}

// Params to show a resource in the UI.
//
// @since 3.16.0
type ShowDocumentParams struct { // line 3128
	// The uri to show.
	URI URI `json:"uri"`
	// Indicates to show the resource in an external program.
	// To show, for example, `https://code.visualstudio.com/`
	// in the default WEB browser set `external` to `true`.
	External bool `json:"external,omitempty"`
	// An optional property to indicate whether the editor
	// showing the document should take focus or not.
	// Clients might ignore this property if an external
	// program is started.
	TakeFocus bool `json:"takeFocus,omitempty"`
	// An optional selection range if the document is a text
	// document. Clients might ignore the property if an
	// external program is started or the file is not a text
	// file.
	Selection *Range `json:"selection,omitempty"`
}

// The result of a showDocument request.
//
// @since 3.16.0
type ShowDocumentResult struct { // line 3170
	// A boolean indicating if the show was successful.
	Success bool `json:"success"`
}

// The parameters of a notification message.
type ShowMessageParams struct { // line 4378
	// The message type. See {@link MessageType}
	Type MessageType `json:"type"`
	// The actual message.
	Message string `json:"message"`
}

// Show message request client capabilities
type ShowMessageRequestClientCapabilities struct { // line 12851
	// Capabilities specific to the `MessageActionItem` type.
	MessageActionItem *PMessageActionItemPShowMessage `json:"messageActionItem,omitempty"`
}
type ShowMessageRequestParams struct { // line 4400
	// The message type. See {@link MessageType}
	Type MessageType `json:"type"`
	// The actual message.
	Message string `json:"message"`
	// The message action items to present.
	Actions []MessageActionItem `json:"actions,omitempty"`
}

// Signature help represents the signature of something
// callable. There can be multiple signature but only one
// active and only one active parameter.
type SignatureHelp struct { // line 5163
	// One or more signatures.
	Signatures []SignatureInformation `json:"signatures"`
	// The active signature. If omitted or the value lies outside the
	// range of `signatures` the value defaults to zero or is ignored if
	// the `SignatureHelp` has no signatures.
	//
	// Whenever possible implementors should make an active decision about
	// the active signature and shouldn't rely on a default value.
	//
	// In future version of the protocol this property might become
	// mandatory to better express this.
	ActiveSignature uint32 `json:"activeSignature,omitempty"`
	// The active parameter of the active signature. If omitted or the value
	// lies outside the range of `signatures[activeSignature].parameters`
	// defaults to 0 if the active signature has parameters. If
	// the active signature has no parameters it is ignored.
	// In future version of the protocol this property might become
	// mandatory to better express the active parameter if the
	// active signature does have any.
	ActiveParameter uint32 `json:"activeParameter,omitempty"`
}

// Client Capabilities for a {@link SignatureHelpRequest}.
type SignatureHelpClientCapabilities struct { // line 11793
	// Whether signature help supports dynamic registration.
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
	// The client supports the following `SignatureInformation`
	// specific properties.
	SignatureInformation *PSignatureInformationPSignatureHelp `json:"signatureInformation,omitempty"`
	// The client supports to send additional context information for a
	// `textDocument/signatureHelp` request. A client that opts into
	// contextSupport will also support the `retriggerCharacters` on
	// `SignatureHelpOptions`.
	//
	// @since 3.15.0
	ContextSupport bool `json:"contextSupport,omitempty"`
}

// Additional information about the context in which a signature help request was triggered.
//
// @since 3.15.0
type SignatureHelpContext struct { // line 9105
	// Action that caused signature help to be triggered.
	TriggerKind SignatureHelpTriggerKind `json:"triggerKind"`
	// Character that caused signature help to be triggered.
	//
	// This is undefined when `triggerKind !== SignatureHelpTriggerKind.TriggerCharacter`
	TriggerCharacter string `json:"triggerCharacter,omitempty"`
	// `true` if signature help was already showing when it was triggered.
	//
	// Retriggers occurs when the signature help is already active and can be caused by actions such as
	// typing a trigger character, a cursor move, or document content changes.
	IsRetrigger bool `json:"isRetrigger"`
	// The currently active `SignatureHelp`.
	//
	// The `activeSignatureHelp` has its `SignatureHelp.activeSignature` field updated based on
	// the user navigating through available signatures.
	ActiveSignatureHelp *SignatureHelp `json:"activeSignatureHelp,omitempty"`
}

// Server Capabilities for a {@link SignatureHelpRequest}.
type SignatureHelpOptions struct { // line 9200
	// List of characters that trigger signature help automatically.
	TriggerCharacters []string `json:"triggerCharacters,omitempty"`
	// List of characters that re-trigger signature help.
	//
	// These trigger characters are only active when signature help is already showing. All trigger characters
	// are also counted as re-trigger characters.
	//
	// @since 3.15.0
	RetriggerCharacters []string `json:"retriggerCharacters,omitempty"`
	WorkDoneProgressOptions
}

// Parameters for a {@link SignatureHelpRequest}.
type SignatureHelpParams struct { // line 5135
	// The signature help context. This is only available if the client specifies
	// to send this using the client capability `textDocument.signatureHelp.contextSupport === true`
	//
	// @since 3.15.0
	Context *SignatureHelpContext `json:"context,omitempty"`
	TextDocumentPositionParams
	WorkDoneProgressParams
}

// Registration options for a {@link SignatureHelpRequest}.
type SignatureHelpRegistrationOptions struct { // line 5198
	TextDocumentRegistrationOptions
	SignatureHelpOptions
}

// How a signature help was triggered.
//
// @since 3.15.0
type SignatureHelpTriggerKind uint32 // line 13995
// Represents the signature of something callable. A signature
// can have a label, like a function-name, a doc-comment, and
// a set of parameters.
type SignatureInformation struct { // line 9146
	// The label of this signature. Will be shown in
	// the UI.
	Label string `json:"label"`
	// The human-readable doc-comment of this signature. Will be shown
	// in the UI but can be omitted.
	Documentation *Or_SignatureInformation_documentation `json:"documentation,omitempty"`
	// The parameters of this signature.
	Parameters []ParameterInformation `json:"parameters,omitempty"`
	// The index of the active parameter.
	//
	// If provided, this is used in place of `SignatureHelp.activeParameter`.
	//
	// @since 3.16.0
	ActiveParameter uint32 `json:"activeParameter,omitempty"`
}

// Static registration options to be returned in the initialize
// request.
type StaticRegistrationOptions struct { // line 6579
	// The id used to register the request. The id can be used to deregister
	// the request again. See also Registration#id.
	ID string `json:"id,omitempty"`
}

// A string value used as a snippet is a template which allows to insert text
// and to control the editor cursor when insertion happens.
//
// A snippet can define tab stops and placeholders with `$1`, `$2`
// and `${3:foo}`. `$0` defines the final tab stop, it defaults to
// the end of the snippet. Variables are defined with `$name` and
// `${name:default value}`.
//
// @since 3.18.0
// @proposed
type StringValue struct { // line 7858
	// The kind of string value.
	Kind string `json:"kind"`
	// The snippet string.
	Value string `json:"value"`
}

// Represents information about programming constructs like variables, classes,
// interfaces etc.
type SymbolInformation struct { // line 5376
	// extends BaseSymbolInformation
	// Indicates if this symbol is deprecated.
	//
	// @deprecated Use tags instead
	Deprecated bool `json:"deprecated,omitempty"`
	// The location of this symbol. The location's range is used by a tool
	// to reveal the location in the editor. If the symbol is selected in the
	// tool the range's start information is used to position the cursor. So
	// the range usually spans more than the actual symbol's name and does
	// normally include things like visibility modifiers.
	//
	// The range doesn't have to denote a node range in the sense of an abstract
	// syntax tree. It can therefore not be used to re-construct a hierarchy of
	// the symbols.
	Location Location `json:"location"`
	// The name of this symbol.
	Name string `json:"name"`
	// The kind of this symbol.
	Kind SymbolKind `json:"kind"`
	// Tags for this symbol.
	//
	// @since 3.16.0
	Tags []SymbolTag `json:"tags,omitempty"`
	// The name of the symbol containing this symbol. This information is for
	// user interface purposes (e.g. to render a qualifier in the user interface
	// if necessary). It can't be used to re-infer a hierarchy for the document
	// symbols.
	ContainerName string `json:"containerName,omitempty"`
}

// A symbol kind.
type SymbolKind uint32 // line 13234
// Symbol tags are extra annotations that tweak the rendering of a symbol.
//
// @since 3.16
type SymbolTag uint32 // line 13348
// Describe options to be used when registered for text document change events.
type TextDocumentChangeRegistrationOptions struct { // line 4507
	// How documents are synced to the server.
	SyncKind TextDocumentSyncKind `json:"syncKind"`
	TextDocumentRegistrationOptions
}

// Text document specific client capabilities.
type TextDocumentClientCapabilities struct { // line 10677
	// Defines which synchronization capabilities the client supports.
	Synchronization *TextDocumentSyncClientCapabilities `json:"synchronization,omitempty"`
	// Capabilities specific to the `textDocument/completion` request.
	Completion CompletionClientCapabilities `json:"completion,omitempty"`
	// Capabilities specific to the `textDocument/hover` request.
	Hover *HoverClientCapabilities `json:"hover,omitempty"`
	// Capabilities specific to the `textDocument/signatureHelp` request.
	SignatureHelp *SignatureHelpClientCapabilities `json:"signatureHelp,omitempty"`
	// Capabilities specific to the `textDocument/declaration` request.
	//
	// @since 3.14.0
	Declaration *DeclarationClientCapabilities `json:"declaration,omitempty"`
	// Capabilities specific to the `textDocument/definition` request.
	Definition *DefinitionClientCapabilities `json:"definition,omitempty"`
	// Capabilities specific to the `textDocument/typeDefinition` request.
	//
	// @since 3.6.0
	TypeDefinition *TypeDefinitionClientCapabilities `json:"typeDefinition,omitempty"`
	// Capabilities specific to the `textDocument/implementation` request.
	//
	// @since 3.6.0
	Implementation *ImplementationClientCapabilities `json:"implementation,omitempty"`
	// Capabilities specific to the `textDocument/references` request.
	References *ReferenceClientCapabilities `json:"references,omitempty"`
	// Capabilities specific to the `textDocument/documentHighlight` request.
	DocumentHighlight *DocumentHighlightClientCapabilities `json:"documentHighlight,omitempty"`
	// Capabilities specific to the `textDocument/documentSymbol` request.
	DocumentSymbol DocumentSymbolClientCapabilities `json:"documentSymbol,omitempty"`
	// Capabilities specific to the `textDocument/codeAction` request.
	CodeAction CodeActionClientCapabilities `json:"codeAction,omitempty"`
	// Capabilities specific to the `textDocument/codeLens` request.
	CodeLens *CodeLensClientCapabilities `json:"codeLens,omitempty"`
	// Capabilities specific to the `textDocument/documentLink` request.
	DocumentLink *DocumentLinkClientCapabilities `json:"documentLink,omitempty"`
	// Capabilities specific to the `textDocument/documentColor` and the
	// `textDocument/colorPresentation` request.
	//
	// @since 3.6.0
	ColorProvider *DocumentColorClientCapabilities `json:"colorProvider,omitempty"`
	// Capabilities specific to the `textDocument/formatting` request.
	Formatting *DocumentFormattingClientCapabilities `json:"formatting,omitempty"`
	// Capabilities specific to the `textDocument/rangeFormatting` request.
	RangeFormatting *DocumentRangeFormattingClientCapabilities `json:"rangeFormatting,omitempty"`
	// Capabilities specific to the `textDocument/onTypeFormatting` request.
	OnTypeFormatting *DocumentOnTypeFormattingClientCapabilities `json:"onTypeFormatting,omitempty"`
	// Capabilities specific to the `textDocument/rename` request.
	Rename *RenameClientCapabilities `json:"rename,omitempty"`
	// Capabilities specific to the `textDocument/foldingRange` request.
	//
	// @since 3.10.0
	FoldingRange *FoldingRangeClientCapabilities `json:"foldingRange,omitempty"`
	// Capabilities specific to the `textDocument/selectionRange` request.
	//
	// @since 3.15.0
	SelectionRange *SelectionRangeClientCapabilities `json:"selectionRange,omitempty"`
	// Capabilities specific to the `textDocument/publishDiagnostics` notification.
	PublishDiagnostics PublishDiagnosticsClientCapabilities `json:"publishDiagnostics,omitempty"`
	// Capabilities specific to the various call hierarchy requests.
	//
	// @since 3.16.0
	CallHierarchy *CallHierarchyClientCapabilities `json:"callHierarchy,omitempty"`
	// Capabilities specific to the various semantic token request.
	//
	// @since 3.16.0
	SemanticTokens SemanticTokensClientCapabilities `json:"semanticTokens,omitempty"`
	// Capabilities specific to the `textDocument/linkedEditingRange` request.
	//
	// @since 3.16.0
	LinkedEditingRange *LinkedEditingRangeClientCapabilities `json:"linkedEditingRange,omitempty"`
	// Client capabilities specific to the `textDocument/moniker` request.
	//
	// @since 3.16.0
	Moniker *MonikerClientCapabilities `json:"moniker,omitempty"`
	// Capabilities specific to the various type hierarchy requests.
	//
	// @since 3.17.0
	TypeHierarchy *TypeHierarchyClientCapabilities `json:"typeHierarchy,omitempty"`
	// Capabilities specific to the `textDocument/inlineValue` request.
	//
	// @since 3.17.0
	InlineValue *InlineValueClientCapabilities `json:"inlineValue,omitempty"`
	// Capabilities specific to the `textDocument/inlayHint` request.
	//
	// @since 3.17.0
	InlayHint *InlayHintClientCapabilities `json:"inlayHint,omitempty"`
	// Capabilities specific to the diagnostic pull model.
	//
	// @since 3.17.0
	Diagnostic *DiagnosticClientCapabilities `json:"diagnostic,omitempty"`
	// Client capabilities specific to inline completions.
	//
	// @since 3.18.0
	// @proposed
	InlineCompletion *InlineCompletionClientCapabilities `json:"inlineCompletion,omitempty"`
}

// An event describing a change to a text document. If only a text is provided
// it is considered to be the full content of the document.
type TextDocumentContentChangeEvent = Msg_TextDocumentContentChangeEvent // (alias) line 14417
// Describes textual changes on a text document. A TextDocumentEdit describes all changes
// on a document version Si and after they are applied move the document to version Si+1.
// So the creator of a TextDocumentEdit doesn't need to sort the array of edits or do any
// kind of ordering. However the edits must be non overlapping.
type TextDocumentEdit struct { // line 6913
	// The text document to change.
	TextDocument OptionalVersionedTextDocumentIdentifier `json:"textDocument"`
	// The edits to be applied.
	//
	// @since 3.16.0 - support for AnnotatedTextEdit. This is guarded using a
	// client capability.
	Edits []TextEdit `json:"edits"`
}

// A document filter denotes a document by different properties like
// the {@link TextDocument.languageId language}, the {@link Uri.scheme scheme} of
// its resource, or a glob-pattern that is applied to the {@link TextDocument.fileName path}.
//
// Glob patterns can have the following syntax:
//
//   - `*` to match one or more characters in a path segment
//   - `?` to match on one character in a path segment
//   - `**` to match any number of path segments, including none
//   - `{}` to group sub patterns into an OR expression. (e.g. `**​/*.{ts,js}` matches all TypeScript and JavaScript files)
//   - `[]` to declare a range of characters to match in a path segment (e.g., `example.[0-9]` to match on `example.0`, `example.1`, …)
//   - `[!...]` to negate a range of characters to match in a path segment (e.g., `example.[!0-9]` to match on `example.a`, `example.b`, but not `example.0`)
//
// @sample A language filter that applies to typescript files on disk: `{ language: 'typescript', scheme: 'file' }`
// @sample A language filter that applies to all package.json paths: `{ language: 'json', pattern: '**package.json' }`
//
// @since 3.17.0
type TextDocumentFilter = Msg_TextDocumentFilter // (alias) line 14560
// A literal to identify a text document in the client.
type TextDocumentIdentifier struct { // line 6655
	// The text document's uri.
	URI DocumentURI `json:"uri"`
}

// An item to transfer a text document from the client to the
// server.
type TextDocumentItem struct { // line 7641
	// The text document's uri.
	URI DocumentURI `json:"uri"`
	// The text document's language identifier.
	LanguageID string `json:"languageId"`
	// The version number of this document (it will increase after each
	// change, including undo/redo).
	Version int32 `json:"version"`
	// The content of the opened text document.
	Text string `json:"text"`
}

// A parameter literal used in requests to pass a text document and a position inside that
// document.
type TextDocumentPositionParams struct { // line 6458
	// The text document.
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	// The position inside the text document.
	Position Position `json:"position"`
}

// General text document registration options.
type TextDocumentRegistrationOptions struct { // line 2441
	// A document selector to identify the scope of the registration. If set to null
	// the document selector provided on the client side will be used.
	DocumentSelector DocumentSelector `json:"documentSelector"`
}

// Represents reasons why a text document is saved.
type TextDocumentSaveReason uint32 // line 13502
// Save registration options.
type TextDocumentSaveRegistrationOptions struct { // line 4564
	TextDocumentRegistrationOptions
	SaveOptions
}
type TextDocumentSyncClientCapabilities struct { // line 11492
	// Whether text document synchronization supports dynamic registration.
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
	// The client supports sending will save notifications.
	WillSave bool `json:"willSave,omitempty"`
	// The client supports sending a will save request and
	// waits for a response providing text edits which will
	// be applied to the document before it is saved.
	WillSaveWaitUntil bool `json:"willSaveWaitUntil,omitempty"`
	// The client supports did save notifications.
	DidSave bool `json:"didSave,omitempty"`
}

// Defines how the host (editor) should sync
// document changes to the language server.
type TextDocumentSyncKind uint32      // line 13477
type TextDocumentSyncOptions struct { // line 10090
	// Open and close notifications are sent to the server. If omitted open close notification should not
	// be sent.
	OpenClose bool `json:"openClose,omitempty"`
	// Change notifications are sent to the server. See TextDocumentSyncKind.None, TextDocumentSyncKind.Full
	// and TextDocumentSyncKind.Incremental. If omitted it defaults to TextDocumentSyncKind.None.
	Change TextDocumentSyncKind `json:"change,omitempty"`
	// If present will save notifications are sent to the server. If omitted the notification should not be
	// sent.
	WillSave bool `json:"willSave,omitempty"`
	// If present will save wait until requests are sent to the server. If omitted the request should not be
	// sent.
	WillSaveWaitUntil bool `json:"willSaveWaitUntil,omitempty"`
	// If present save notifications are sent to the server. If omitted the notification should not be
	// sent.
	Save *SaveOptions `json:"save,omitempty"`
}

// A text edit applicable to a text document.
type TextEdit struct { // line 4601
	// The range of the text document to be manipulated. To insert
	// text into a document create a range where start === end.
	Range Range `json:"range"`
	// The string to be inserted. For delete operations use an
	// empty string.
	NewText string `json:"newText"`
}
type TokenFormat string // line 14151
type TraceValues string // line 13776
// Since 3.6.0
type TypeDefinitionClientCapabilities struct { // line 11924
	// Whether implementation supports dynamic registration. If this is set to `true`
	// the client supports the new `TypeDefinitionRegistrationOptions` return value
	// for the corresponding server capability as well.
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
	// The client supports additional metadata in the form of definition links.
	//
	// Since 3.14.0
	LinkSupport bool `json:"linkSupport,omitempty"`
}
type TypeDefinitionOptions struct { // line 6594
	WorkDoneProgressOptions
}
type TypeDefinitionParams struct { // line 2196
	TextDocumentPositionParams
	WorkDoneProgressParams
	PartialResultParams
}
type TypeDefinitionRegistrationOptions struct { // line 2216
	TextDocumentRegistrationOptions
	TypeDefinitionOptions
	StaticRegistrationOptions
}

// @since 3.17.0
type TypeHierarchyClientCapabilities struct { // line 12713
	// Whether implementation supports dynamic registration. If this is set to `true`
	// the client supports the new `(TextDocumentRegistrationOptions & StaticRegistrationOptions)`
	// return value for the corresponding server capability as well.
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
}

// @since 3.17.0
type TypeHierarchyItem struct { // line 3483
	// The name of this item.
	Name string `json:"name"`
	// The kind of this item.
	Kind SymbolKind `json:"kind"`
	// Tags for this item.
	Tags []SymbolTag `json:"tags,omitempty"`
	// More detail for this item, e.g. the signature of a function.
	Detail string `json:"detail,omitempty"`
	// The resource identifier of this item.
	URI DocumentURI `json:"uri"`
	// The range enclosing this symbol not including leading/trailing whitespace
	// but everything else, e.g. comments and code.
	Range Range `json:"range"`
	// The range that should be selected and revealed when this symbol is being
	// picked, e.g. the name of a function. Must be contained by the
	// {@link TypeHierarchyItem.range `range`}.
	SelectionRange Range `json:"selectionRange"`
	// A data entry field that is preserved between a type hierarchy prepare and
	// supertypes or subtypes requests. It could also be used to identify the
	// type hierarchy in the server, helping improve the performance on
	// resolving supertypes and subtypes.
	Data interface{} `json:"data,omitempty"`
}

// Type hierarchy options used during static registration.
//
// @since 3.17.0
type TypeHierarchyOptions struct { // line 7172
	WorkDoneProgressOptions
}

// The parameter of a `textDocument/prepareTypeHierarchy` request.
//
// @since 3.17.0
type TypeHierarchyPrepareParams struct { // line 3465
	TextDocumentPositionParams
	WorkDoneProgressParams
}

// Type hierarchy options used during static or dynamic registration.
//
// @since 3.17.0
type TypeHierarchyRegistrationOptions struct { // line 3560
	TextDocumentRegistrationOptions
	TypeHierarchyOptions
	StaticRegistrationOptions
}

// The parameter of a `typeHierarchy/subtypes` request.
//
// @since 3.17.0
type TypeHierarchySubtypesParams struct { // line 3606
	Item TypeHierarchyItem `json:"item"`
	WorkDoneProgressParams
	PartialResultParams
}

// The parameter of a `typeHierarchy/supertypes` request.
//
// @since 3.17.0
type TypeHierarchySupertypesParams struct { // line 3582
	Item TypeHierarchyItem `json:"item"`
	WorkDoneProgressParams
	PartialResultParams
}

// created for Tuple
type UIntCommaUInt struct { // line 10430
	Fld0 uint32 `json:"fld0"`
	Fld1 uint32 `json:"fld1"`
}
type URI = string

// A diagnostic report indicating that the last returned
// report is still accurate.
//
// @since 3.17.0
type UnchangedDocumentDiagnosticReport struct { // line 7506
	// A document diagnostic report indicating
	// no changes to the last result. A server can
	// only return `unchanged` if result ids are
	// provided.
	Kind string `json:"kind"`
	// A result id which will be sent on the next
	// diagnostic request for the same document.
	ResultID string `json:"resultId"`
}

// Moniker uniqueness level to define scope of the moniker.
//
// @since 3.16.0
type UniquenessLevel string // line 13364
// General parameters to unregister a request or notification.
type Unregistration struct { // line 7926
	// The id used to unregister the request or notification. Usually an id
	// provided during the register request.
	ID string `json:"id"`
	// The method to unregister for.
	Method string `json:"method"`
}
type UnregistrationParams struct { // line 4248
	Unregisterations []Unregistration `json:"unregisterations"`
}

// A versioned notebook document identifier.
//
// @since 3.17.0
type VersionedNotebookDocumentIdentifier struct { // line 7679
	// The version number of this notebook document.
	Version int32 `json:"version"`
	// The notebook document's uri.
	URI URI `json:"uri"`
}

// A text document identifier to denote a specific version of a text document.
type VersionedTextDocumentIdentifier struct { // line 8763
	// The version number of this document.
	Version int32 `json:"version"`
	TextDocumentIdentifier
}
type WatchKind = uint32                  // line 13505// The parameters sent in a will save text document notification.
type WillSaveTextDocumentParams struct { // line 4579
	// The document that will be saved.
	TextDocument TextDocumentIdentifier `json:"textDocument"`
	// The 'TextDocumentSaveReason'.
	Reason TextDocumentSaveReason `json:"reason"`
}
type WindowClientCapabilities struct { // line 10994
	// It indicates whether the client supports server initiated
	// progress using the `window/workDoneProgress/create` request.
	//
	// The capability also controls Whether client supports handling
	// of progress notifications. If set servers are allowed to report a
	// `workDoneProgress` property in the request specific server
	// capabilities.
	//
	// @since 3.15.0
	WorkDoneProgress bool `json:"workDoneProgress,omitempty"`
	// Capabilities specific to the showMessage request.
	//
	// @since 3.16.0
	ShowMessage *ShowMessageRequestClientCapabilities `json:"showMessage,omitempty"`
	// Capabilities specific to the showDocument request.
	//
	// @since 3.16.0
	ShowDocument *ShowDocumentClientCapabilities `json:"showDocument,omitempty"`
}
type WorkDoneProgressBegin struct { // line 6276
	Kind string `json:"kind"`
	// Mandatory title of the progress operation. Used to briefly inform about
	// the kind of operation being performed.
	//
	// Examples: "Indexing" or "Linking dependencies".
	Title string `json:"title"`
	// Controls if a cancel button should show to allow the user to cancel the
	// long running operation. Clients that don't support cancellation are allowed
	// to ignore the setting.
	Cancellable bool `json:"cancellable,omitempty"`
	// Optional, more detailed associated progress message. Contains
	// complementary information to the `title`.
	//
	// Examples: "3/25 files", "project/src/module2", "node_modules/some_dep".
	// If unset, the previous progress message (if any) is still valid.
	Message string `json:"message,omitempty"`
	// Optional progress percentage to display (value 100 is considered 100%).
	// If not provided infinite progress is assumed and clients are allowed
	// to ignore the `percentage` value in subsequent in report notifications.
	//
	// The value should be steadily rising. Clients are free to ignore values
	// that are not following this rule. The value range is [0, 100].
	Percentage uint32 `json:"percentage,omitempty"`
}
type WorkDoneProgressCancelParams struct { // line 2698
	// The token to be used to report progress.
	Token ProgressToken `json:"token"`
}
type WorkDoneProgressCreateParams struct { // line 2685
	// The token to be used to report progress.
	Token ProgressToken `json:"token"`
}
type WorkDoneProgressEnd struct { // line 6362
	Kind string `json:"kind"`
	// Optional, a final message indicating to for example indicate the outcome
	// of the operation.
	Message string `json:"message,omitempty"`
}
type WorkDoneProgressOptions struct { // line 2428
	WorkDoneProgress bool `json:"workDoneProgress,omitempty"`
}

// created for And
type WorkDoneProgressOptionsAndTextDocumentRegistrationOptions struct { // line 196
	WorkDoneProgressOptions
	TextDocumentRegistrationOptions
}
type WorkDoneProgressParams struct { // line 6480
	// An optional token that a server can use to report work done progress.
	WorkDoneToken ProgressToken `json:"workDoneToken,omitempty"`
}
type WorkDoneProgressReport struct { // line 6323
	Kind string `json:"kind"`
	// Controls enablement state of a cancel button.
	//
	// Clients that don't support cancellation or don't support controlling the button's
	// enablement state are allowed to ignore the property.
	Cancellable bool `json:"cancellable,omitempty"`
	// Optional, more detailed associated progress message. Contains
	// complementary information to the `title`.
	//
	// Examples: "3/25 files", "project/src/module2", "node_modules/some_dep".
	// If unset, the previous progress message (if any) is still valid.
	Message string `json:"message,omitempty"`
	// Optional progress percentage to display (value 100 is considered 100%).
	// If not provided infinite progress is assumed and clients are allowed
	// to ignore the `percentage` value in subsequent in report notifications.
	//
	// The value should be steadily rising. Clients are free to ignore values
	// that are not following this rule. The value range is [0, 100]
	Percentage uint32 `json:"percentage,omitempty"`
}

// created for Literal (Lit_ServerCapabilities_workspace)
type Workspace6Gn struct { // line 8722
	// The server supports workspace folder.
	//
	// @since 3.6.0
	WorkspaceFolders *WorkspaceFolders5Gn `json:"workspaceFolders,omitempty"`
	// The server is interested in notifications/requests for operations on files.
	//
	// @since 3.16.0
	FileOperations *FileOperationOptions `json:"fileOperations,omitempty"`
}

// Workspace specific client capabilities.
type WorkspaceClientCapabilities struct { // line 10538
	// The client supports applying batch edits
	// to the workspace by supporting the request
	// 'workspace/applyEdit'
	ApplyEdit bool `json:"applyEdit,omitempty"`
	// Capabilities specific to `WorkspaceEdit`s.
	WorkspaceEdit *WorkspaceEditClientCapabilities `json:"workspaceEdit,omitempty"`
	// Capabilities specific to the `workspace/didChangeConfiguration` notification.
	DidChangeConfiguration DidChangeConfigurationClientCapabilities `json:"didChangeConfiguration,omitempty"`
	// Capabilities specific to the `workspace/didChangeWatchedFiles` notification.
	DidChangeWatchedFiles DidChangeWatchedFilesClientCapabilities `json:"didChangeWatchedFiles,omitempty"`
	// Capabilities specific to the `workspace/symbol` request.
	Symbol *WorkspaceSymbolClientCapabilities `json:"symbol,omitempty"`
	// Capabilities specific to the `workspace/executeCommand` request.
	ExecuteCommand *ExecuteCommandClientCapabilities `json:"executeCommand,omitempty"`
	// The client has support for workspace folders.
	//
	// @since 3.6.0
	WorkspaceFolders bool `json:"workspaceFolders,omitempty"`
	// The client supports `workspace/configuration` requests.
	//
	// @since 3.6.0
	Configuration bool `json:"configuration,omitempty"`
	// Capabilities specific to the semantic token requests scoped to the
	// workspace.
	//
	// @since 3.16.0.
	SemanticTokens *SemanticTokensWorkspaceClientCapabilities `json:"semanticTokens,omitempty"`
	// Capabilities specific to the code lens requests scoped to the
	// workspace.
	//
	// @since 3.16.0.
	CodeLens *CodeLensWorkspaceClientCapabilities `json:"codeLens,omitempty"`
	// The client has support for file notifications/requests for user operations on files.
	//
	// Since 3.16.0
	FileOperations *FileOperationClientCapabilities `json:"fileOperations,omitempty"`
	// Capabilities specific to the inline values requests scoped to the
	// workspace.
	//
	// @since 3.17.0.
	InlineValue *InlineValueWorkspaceClientCapabilities `json:"inlineValue,omitempty"`
	// Capabilities specific to the inlay hint requests scoped to the
	// workspace.
	//
	// @since 3.17.0.
	InlayHint *InlayHintWorkspaceClientCapabilities `json:"inlayHint,omitempty"`
	// Capabilities specific to the diagnostic requests scoped to the
	// workspace.
	//
	// @since 3.17.0.
	Diagnostics *DiagnosticWorkspaceClientCapabilities `json:"diagnostics,omitempty"`
}

// Parameters of the workspace diagnostic request.
//
// @since 3.17.0
type WorkspaceDiagnosticParams struct { // line 3950
	// The additional identifier provided during registration.
	Identifier string `json:"identifier,omitempty"`
	// The currently known diagnostic reports with their
	// previous result ids.
	PreviousResultIds []PreviousResultID `json:"previousResultIds"`
	WorkDoneProgressParams
	PartialResultParams
}

// A workspace diagnostic report.
//
// @since 3.17.0
type WorkspaceDiagnosticReport struct { // line 3987
	Items []WorkspaceDocumentDiagnosticReport `json:"items"`
}

// A partial result for a workspace diagnostic report.
//
// @since 3.17.0
type WorkspaceDiagnosticReportPartialResult struct { // line 4004
	Items []WorkspaceDocumentDiagnosticReport `json:"items"`
}

// A workspace diagnostic document report.
//
// @since 3.17.0
type WorkspaceDocumentDiagnosticReport = Or_WorkspaceDocumentDiagnosticReport // (alias) line 14399
// A workspace edit represents changes to many resources managed in the workspace. The edit
// should either provide `changes` or `documentChanges`. If documentChanges are present
// they are preferred over `changes` if the client can handle versioned document edits.
//
// Since version 3.13.0 a workspace edit can contain resource operations as well. If resource
// operations are present clients need to execute the operations in the order in which they
// are provided. So a workspace edit for example can consist of the following two changes:
// (1) a create file a.txt and (2) a text document edit which insert text into file a.txt.
//
// An invalid sequence (e.g. (1) delete file a.txt and (2) insert text into file a.txt) will
// cause failure of the operation. How the client recovers from the failure is described by
// the client capability: `workspace.workspaceEdit.failureHandling`
type WorkspaceEdit struct { // line 3266
	// Holds changes to existing resources.
	Changes map[DocumentURI][]TextEdit `json:"changes,omitempty"`
	// Depending on the client capability `workspace.workspaceEdit.resourceOperations` document changes
	// are either an array of `TextDocumentEdit`s to express changes to n different text documents
	// where each text document edit addresses a specific version of a text document. Or it can contain
	// above `TextDocumentEdit`s mixed with create, rename and delete file / folder operations.
	//
	// Whether a client supports versioned document edits is expressed via
	// `workspace.workspaceEdit.documentChanges` client capability.
	//
	// If a client neither supports `documentChanges` nor `workspace.workspaceEdit.resourceOperations` then
	// only plain `TextEdit`s using the `changes` property are supported.
	DocumentChanges []DocumentChanges `json:"documentChanges,omitempty"`
	// A map of change annotations that can be referenced in `AnnotatedTextEdit`s or create, rename and
	// delete file / folder operations.
	//
	// Whether clients honor this property depends on the client capability `workspace.changeAnnotationSupport`.
	//
	// @since 3.16.0
	ChangeAnnotations map[ChangeAnnotationIdentifier]ChangeAnnotation `json:"changeAnnotations,omitempty"`
}
type WorkspaceEditClientCapabilities struct { // line 11133
	// The client supports versioned document changes in `WorkspaceEdit`s
	DocumentChanges bool `json:"documentChanges,omitempty"`
	// The resource operations the client supports. Clients should at least
	// support 'create', 'rename' and 'delete' files and folders.
	//
	// @since 3.13.0
	ResourceOperations []ResourceOperationKind `json:"resourceOperations,omitempty"`
	// The failure handling strategy of a client if applying the workspace edit
	// fails.
	//
	// @since 3.13.0
	FailureHandling *FailureHandlingKind `json:"failureHandling,omitempty"`
	// Whether the client normalizes line endings to the client specific
	// setting.
	// If set to `true` the client will normalize line ending characters
	// in a workspace edit to the client-specified new line
	// character.
	//
	// @since 3.16.0
	NormalizesLineEndings bool `json:"normalizesLineEndings,omitempty"`
	// Whether the client in general supports change annotations on text edits,
	// create file, rename file and delete file changes.
	//
	// @since 3.16.0
	ChangeAnnotationSupport *PChangeAnnotationSupportPWorkspaceEdit `json:"changeAnnotationSupport,omitempty"`
}

// A workspace folder inside a client.
type WorkspaceFolder struct { // line 2236
	// The associated URI for this workspace folder.
	URI URI `json:"uri"`
	// The name of the workspace folder. Used to refer to this
	// workspace folder in the user interface.
	Name string `json:"name"`
}
type WorkspaceFolders5Gn struct { // line 10287
	// The server has support for workspace folders
	Supported bool `json:"supported,omitempty"`
	// Whether the server wants to receive workspace folder
	// change notifications.
	//
	// If a string is provided the string is treated as an ID
	// under which the notification is registered on the client
	// side. The ID can be used to unregister for these events
	// using the `client/unregisterCapability` request.
	ChangeNotifications string `json:"changeNotifications,omitempty"`
}

// The workspace folder change event.
type WorkspaceFoldersChangeEvent struct { // line 6604
	// The array of added workspace folders
	Added []WorkspaceFolder `json:"added"`
	// The array of the removed workspace folders
	Removed []WorkspaceFolder `json:"removed"`
}
type WorkspaceFoldersInitializeParams struct { // line 8080
	// The workspace folders configured in the client when the server starts.
	//
	// This property is only available if the client supports workspace folders.
	// It can be `null` if the client supports workspace folders but none are
	// configured.
	//
	// @since 3.6.0
	WorkspaceFolders []WorkspaceFolder `json:"workspaceFolders,omitempty"`
}
type WorkspaceFoldersServerCapabilities struct { // line 10287
	// The server has support for workspace folders
	Supported bool `json:"supported,omitempty"`
	// Whether the server wants to receive workspace folder
	// change notifications.
	//
	// If a string is provided the string is treated as an ID
	// under which the notification is registered on the client
	// side. The ID can be used to unregister for these events
	// using the `client/unregisterCapability` request.
	ChangeNotifications string `json:"changeNotifications,omitempty"`
}

// A full document diagnostic report for a workspace diagnostic result.
//
// @since 3.17.0
type WorkspaceFullDocumentDiagnosticReport struct { // line 9852
	// The URI for which diagnostic information is reported.
	URI DocumentURI `json:"uri"`
	// The version number for which the diagnostics are reported.
	// If the document is not marked as open `null` can be provided.
	Version int32 `json:"version"`
	FullDocumentDiagnosticReport
}

// A special workspace symbol that supports locations without a range.
//
// See also SymbolInformation.
//
// @since 3.17.0
type WorkspaceSymbol struct { // line 5710
	// The location of the symbol. Whether a server is allowed to
	// return a location without a range depends on the client
	// capability `workspace.symbol.resolveSupport`.
	//
	// See SymbolInformation#location for more details.
	Location OrPLocation_workspace_symbol `json:"location"`
	// A data entry field that is preserved on a workspace symbol between a
	// workspace symbol request and a workspace symbol resolve request.
	Data interface{} `json:"data,omitempty"`
	BaseSymbolInformation
}

// Client capabilities for a {@link WorkspaceSymbolRequest}.
type WorkspaceSymbolClientCapabilities struct { // line 11240
	// Symbol request supports dynamic registration.
	DynamicRegistration bool `json:"dynamicRegistration,omitempty"`
	// Specific capabilities for the `SymbolKind` in the `workspace/symbol` request.
	SymbolKind *PSymbolKindPSymbol `json:"symbolKind,omitempty"`
	// The client supports tags on `SymbolInformation`.
	// Clients supporting tags have to handle unknown tags gracefully.
	//
	// @since 3.16.0
	TagSupport *PTagSupportPSymbol `json:"tagSupport,omitempty"`
	// The client support partial workspace symbols. The client will send the
	// request `workspaceSymbol/resolve` to the server to resolve additional
	// properties.
	//
	// @since 3.17.0
	ResolveSupport *PResolveSupportPSymbol `json:"resolveSupport,omitempty"`
}

// Server capabilities for a {@link WorkspaceSymbolRequest}.
type WorkspaceSymbolOptions struct { // line 9423
	// The server provides support to resolve additional
	// information for a workspace symbol.
	//
	// @since 3.17.0
	ResolveProvider bool `json:"resolveProvider,omitempty"`
	WorkDoneProgressOptions
}

// The parameters of a {@link WorkspaceSymbolRequest}.
type WorkspaceSymbolParams struct { // line 5686
	// A query string to filter symbols by. Clients may send an empty
	// string here to request all symbols.
	Query string `json:"query"`
	WorkDoneProgressParams
	PartialResultParams
}

// Registration options for a {@link WorkspaceSymbolRequest}.
type WorkspaceSymbolRegistrationOptions struct { // line 5759
	WorkspaceSymbolOptions
}

// An unchanged document diagnostic report for a workspace diagnostic result.
//
// @since 3.17.0
type WorkspaceUnchangedDocumentDiagnosticReport struct { // line 9890
	// The URI for which diagnostic information is reported.
	URI DocumentURI `json:"uri"`
	// The version number for which the diagnostics are reported.
	// If the document is not marked as open `null` can be provided.
	Version int32 `json:"version"`
	UnchangedDocumentDiagnosticReport
}

// The initialize parameters
type XInitializeParams struct { // line 7948
	// The process Id of the parent process that started
	// the server.
	//
	// Is `null` if the process has not been started by another process.
	// If the parent process is not alive then the server should exit.
	ProcessID int32 `json:"processId"`
	// Information about the client
	//
	// @since 3.15.0
	ClientInfo *Msg_XInitializeParams_clientInfo `json:"clientInfo,omitempty"`
	// The locale the client is currently showing the user interface
	// in. This must not necessarily be the locale of the operating
	// system.
	//
	// Uses IETF language tags as the value's syntax
	// (See https://en.wikipedia.org/wiki/IETF_language_tag)
	//
	// @since 3.16.0
	Locale string `json:"locale,omitempty"`
	// The rootPath of the workspace. Is null
	// if no folder is open.
	//
	// @deprecated in favour of rootUri.
	RootPath string `json:"rootPath,omitempty"`
	// The rootUri of the workspace. Is null if no
	// folder is open. If both `rootPath` and `rootUri` are set
	// `rootUri` wins.
	//
	// @deprecated in favour of workspaceFolders.
	RootURI DocumentURI `json:"rootUri"`
	// The capabilities provided by the client (editor or tool)
	Capabilities ClientCapabilities `json:"capabilities"`
	// User provided initialization options.
	InitializationOptions interface{} `json:"initializationOptions,omitempty"`
	// The initial trace setting. If omitted trace is disabled ('off').
	Trace *TraceValues `json:"trace,omitempty"`
	WorkDoneProgressParams
}

// The initialize parameters
type _InitializeParams struct { // line 7948
	// The process Id of the parent process that started
	// the server.
	//
	// Is `null` if the process has not been started by another process.
	// If the parent process is not alive then the server should exit.
	ProcessID int32 `json:"processId"`
	// Information about the client
	//
	// @since 3.15.0
	ClientInfo *Msg_XInitializeParams_clientInfo `json:"clientInfo,omitempty"`
	// The locale the client is currently showing the user interface
	// in. This must not necessarily be the locale of the operating
	// system.
	//
	// Uses IETF language tags as the value's syntax
	// (See https://en.wikipedia.org/wiki/IETF_language_tag)
	//
	// @since 3.16.0
	Locale string `json:"locale,omitempty"`
	// The rootPath of the workspace. Is null
	// if no folder is open.
	//
	// @deprecated in favour of rootUri.
	RootPath string `json:"rootPath,omitempty"`
	// The rootUri of the workspace. Is null if no
	// folder is open. If both `rootPath` and `rootUri` are set
	// `rootUri` wins.
	//
	// @deprecated in favour of workspaceFolders.
	RootURI DocumentURI `json:"rootUri"`
	// The capabilities provided by the client (editor or tool)
	Capabilities ClientCapabilities `json:"capabilities"`
	// User provided initialization options.
	InitializationOptions interface{} `json:"initializationOptions,omitempty"`
	// The initial trace setting. If omitted trace is disabled ('off').
	Trace *TraceValues `json:"trace,omitempty"`
	WorkDoneProgressParams
}

const (
	// A set of predefined code action kinds
	// Empty kind.
	Empty CodeActionKind = "" // line 13726
	// Base kind for quickfix actions: 'quickfix'
	QuickFix CodeActionKind = "quickfix" // line 13731
	// Base kind for refactoring actions: 'refactor'
	Refactor CodeActionKind = "refactor" // line 13736
	// Base kind for refactoring extraction actions: 'refactor.extract'
	//
	// Example extract actions:
	//
	//
	//  - Extract method
	//  - Extract function
	//  - Extract variable
	//  - Extract interface from class
	//  - ...
	RefactorExtract CodeActionKind = "refactor.extract" // line 13741
	// Base kind for refactoring inline actions: 'refactor.inline'
	//
	// Example inline actions:
	//
	//
	//  - Inline function
	//  - Inline variable
	//  - Inline constant
	//  - ...
	RefactorInline CodeActionKind = "refactor.inline" // line 13746
	// Base kind for refactoring rewrite actions: 'refactor.rewrite'
	//
	// Example rewrite actions:
	//
	//
	//  - Convert JavaScript function to class
	//  - Add or remove parameter
	//  - Encapsulate field
	//  - Make method static
	//  - Move method to base class
	//  - ...
	RefactorRewrite CodeActionKind = "refactor.rewrite" // line 13751
	// Base kind for source actions: `source`
	//
	// Source code actions apply to the entire file.
	Source CodeActionKind = "source" // line 13756
	// Base kind for an organize imports source action: `source.organizeImports`
	SourceOrganizeImports CodeActionKind = "source.organizeImports" // line 13761
	// Base kind for auto-fix source actions: `source.fixAll`.
	//
	// Fix all actions automatically fix errors that have a clear fix that do not require user input.
	// They should not suppress errors or perform unsafe fixes such as generating new types or classes.
	//
	// @since 3.15.0
	SourceFixAll CodeActionKind = "source.fixAll" // line 13766
	// The reason why code actions were requested.
	//
	// @since 3.17.0
	// Code actions were explicitly requested by the user or by an extension.
	CodeActionInvoked CodeActionTriggerKind = 1 // line 14028
	// Code actions were requested automatically.
	//
	// This typically happens when current selection in a file changes, but can
	// also be triggered when file content changes.
	CodeActionAutomatic CodeActionTriggerKind = 2 // line 14033
	// The kind of a completion entry.
	TextCompletion          CompletionItemKind = 1  // line 13534
	MethodCompletion        CompletionItemKind = 2  // line 13538
	FunctionCompletion      CompletionItemKind = 3  // line 13542
	ConstructorCompletion   CompletionItemKind = 4  // line 13546
	FieldCompletion         CompletionItemKind = 5  // line 13550
	VariableCompletion      CompletionItemKind = 6  // line 13554
	ClassCompletion         CompletionItemKind = 7  // line 13558
	InterfaceCompletion     CompletionItemKind = 8  // line 13562
	ModuleCompletion        CompletionItemKind = 9  // line 13566
	PropertyCompletion      CompletionItemKind = 10 // line 13570
	UnitCompletion          CompletionItemKind = 11 // line 13574
	ValueCompletion         CompletionItemKind = 12 // line 13578
	EnumCompletion          CompletionItemKind = 13 // line 13582
	KeywordCompletion       CompletionItemKind = 14 // line 13586
	SnippetCompletion       CompletionItemKind = 15 // line 13590
	ColorCompletion         CompletionItemKind = 16 // line 13594
	FileCompletion          CompletionItemKind = 17 // line 13598
	ReferenceCompletion     CompletionItemKind = 18 // line 13602
	FolderCompletion        CompletionItemKind = 19 // line 13606
	EnumMemberCompletion    CompletionItemKind = 20 // line 13610
	ConstantCompletion      CompletionItemKind = 21 // line 13614
	StructCompletion        CompletionItemKind = 22 // line 13618
	EventCompletion         CompletionItemKind = 23 // line 13622
	OperatorCompletion      CompletionItemKind = 24 // line 13626
	TypeParameterCompletion CompletionItemKind = 25 // line 13630
	// Completion item tags are extra annotations that tweak the rendering of a completion
	// item.
	//
	// @since 3.15.0
	// Render a completion as obsolete, usually using a strike-out.
	ComplDeprecated CompletionItemTag = 1 // line 13644
	// How a completion was triggered
	// Completion was triggered by typing an identifier (24x7 code
	// complete), manual invocation (e.g Ctrl+Space) or via API.
	Invoked CompletionTriggerKind = 1 // line 13977
	// Completion was triggered by a trigger character specified by
	// the `triggerCharacters` properties of the `CompletionRegistrationOptions`.
	TriggerCharacter CompletionTriggerKind = 2 // line 13982
	// Completion was re-triggered as current completion list is incomplete
	TriggerForIncompleteCompletions CompletionTriggerKind = 3 // line 13987
	// The diagnostic's severity.
	// Reports an error.
	SeverityError DiagnosticSeverity = 1 // line 13926
	// Reports a warning.
	SeverityWarning DiagnosticSeverity = 2 // line 13931
	// Reports an information.
	SeverityInformation DiagnosticSeverity = 3 // line 13936
	// Reports a hint.
	SeverityHint DiagnosticSeverity = 4 // line 13941
	// The diagnostic tags.
	//
	// @since 3.15.0
	// Unused or unnecessary code.
	//
	// Clients are allowed to render diagnostics with this tag faded out instead of having
	// an error squiggle.
	Unnecessary DiagnosticTag = 1 // line 13956
	// Deprecated or obsolete code.
	//
	// Clients are allowed to rendered diagnostics with this tag strike through.
	Deprecated DiagnosticTag = 2 // line 13961
	// The document diagnostic report kinds.
	//
	// @since 3.17.0
	// A diagnostic report with a full
	// set of problems.
	DiagnosticFull DocumentDiagnosticReportKind = "full" // line 13122
	// A report indicating that the last
	// returned report is still accurate.
	DiagnosticUnchanged DocumentDiagnosticReportKind = "unchanged" // line 13127
	// A document highlight kind.
	// A textual occurrence.
	Text DocumentHighlightKind = 1 // line 13701
	// Read-access of a symbol, like reading a variable.
	Read DocumentHighlightKind = 2 // line 13706
	// Write-access of a symbol, like writing to a variable.
	Write DocumentHighlightKind = 3 // line 13711
	// Predefined error codes.
	ParseError     ErrorCodes = -32700 // line 13143
	InvalidRequest ErrorCodes = -32600 // line 13147
	MethodNotFound ErrorCodes = -32601 // line 13151
	InvalidParams  ErrorCodes = -32602 // line 13155
	InternalError  ErrorCodes = -32603 // line 13159
	// Error code indicating that a server received a notification or
	// request before the server has received the `initialize` request.
	ServerNotInitialized ErrorCodes = -32002 // line 13163
	UnknownErrorCode     ErrorCodes = -32001 // line 13168
	// Applying the workspace change is simply aborted if one of the changes provided
	// fails. All operations executed before the failing operation stay executed.
	Abort FailureHandlingKind = "abort" // line 14115
	// All operations are executed transactional. That means they either all
	// succeed or no changes at all are applied to the workspace.
	Transactional FailureHandlingKind = "transactional" // line 14120
	// If the workspace edit contains only textual file changes they are executed transactional.
	// If resource changes (create, rename or delete file) are part of the change the failure
	// handling strategy is abort.
	TextOnlyTransactional FailureHandlingKind = "textOnlyTransactional" // line 14125
	// The client tries to undo the operations already executed. But there is no
	// guarantee that this is succeeding.
	Undo FailureHandlingKind = "undo" // line 14130
	// The file event type
	// The file got created.
	Created FileChangeType = 1 // line 13876
	// The file got changed.
	Changed FileChangeType = 2 // line 13881
	// The file got deleted.
	Deleted FileChangeType = 3 // line 13886
	// A pattern kind describing if a glob pattern matches a file a folder or
	// both.
	//
	// @since 3.16.0
	// The pattern matches a file only.
	FilePattern FileOperationPatternKind = "file" // line 14049
	// The pattern matches a folder only.
	FolderPattern FileOperationPatternKind = "folder" // line 14054
	// A set of predefined range kinds.
	// Folding range for a comment
	Comment FoldingRangeKind = "comment" // line 13215
	// Folding range for an import or include
	Imports FoldingRangeKind = "imports" // line 13220
	// Folding range for a region (e.g. `#region`)
	Region FoldingRangeKind = "region" // line 13225
	// Inlay hint kinds.
	//
	// @since 3.17.0
	// An inlay hint that for a type annotation.
	Type InlayHintKind = 1 // line 13433
	// An inlay hint that is for a parameter.
	Parameter InlayHintKind = 2 // line 13438
	// Describes how an {@link InlineCompletionItemProvider inline completion provider} was triggered.
	//
	// @since 3.18.0
	// @proposed
	// Completion was triggered explicitly by a user gesture.
	InlineInvoked InlineCompletionTriggerKind = 0 // line 13827
	// Completion was triggered automatically while editing.
	InlineAutomatic InlineCompletionTriggerKind = 1 // line 13832
	// Defines whether the insert text in a completion item should be interpreted as
	// plain text or a snippet.
	// The primary text to be inserted is treated as a plain string.
	PlainTextTextFormat InsertTextFormat = 1 // line 13660
	// The primary text to be inserted is treated as a snippet.
	//
	// A snippet can define tab stops and placeholders with `$1`, `$2`
	// and `${3:foo}`. `$0` defines the final tab stop, it defaults to
	// the end of the snippet. Placeholders with equal identifiers are linked,
	// that is typing in one will update others too.
	//
	// See also: https://microsoft.github.io/language-server-protocol/specifications/specification-current/#snippet_syntax
	SnippetTextFormat InsertTextFormat = 2 // line 13665
	// How whitespace and indentation is handled during completion
	// item insertion.
	//
	// @since 3.16.0
	// The insertion or replace strings is taken as it is. If the
	// value is multi line the lines below the cursor will be
	// inserted using the indentation defined in the string value.
	// The client will not apply any kind of adjustments to the
	// string.
	AsIs InsertTextMode = 1 // line 13680
	// The editor adjusts leading whitespace of new lines so that
	// they match the indentation up to the cursor of the line for
	// which the item is accepted.
	//
	// Consider a line like this: <2tabs><cursor><3tabs>foo. Accepting a
	// multi line completion item is indented using 2 tabs and all
	// following lines inserted will be indented using 2 tabs as well.
	AdjustIndentation InsertTextMode = 2 // line 13685
	// A request failed but it was syntactically correct, e.g the
	// method name was known and the parameters were valid. The error
	// message should contain human readable information about why
	// the request failed.
	//
	// @since 3.17.0
	RequestFailed LSPErrorCodes = -32803 // line 13183
	// The server cancelled the request. This error code should
	// only be used for requests that explicitly support being
	// server cancellable.
	//
	// @since 3.17.0
	ServerCancelled LSPErrorCodes = -32802 // line 13189
	// The server detected that the content of a document got
	// modified outside normal conditions. A server should
	// NOT send this error code if it detects a content change
	// in it unprocessed messages. The result even computed
	// on an older state might still be useful for the client.
	//
	// If a client decides that a result is not of any use anymore
	// the client should cancel the request.
	ContentModified LSPErrorCodes = -32801 // line 13195
	// The client has canceled a request and a server as detected
	// the cancel.
	RequestCancelled LSPErrorCodes = -32800 // line 13200
	// Describes the content type that a client supports in various
	// result literals like `Hover`, `ParameterInfo` or `CompletionItem`.
	//
	// Please note that `MarkupKinds` must not start with a `$`. This kinds
	// are reserved for internal usage.
	// Plain text is supported as a content format
	PlainText MarkupKind = "plaintext" // line 13807
	// Markdown is supported as a content format
	Markdown MarkupKind = "markdown" // line 13812
	// The message type
	// An error message.
	Error MessageType = 1 // line 13454
	// A warning message.
	Warning MessageType = 2 // line 13459
	// An information message.
	Info MessageType = 3 // line 13464
	// A log message.
	Log MessageType = 4 // line 13469
	// The moniker kind.
	//
	// @since 3.16.0
	// The moniker represent a symbol that is imported into a project
	Import MonikerKind = "import" // line 13407
	// The moniker represents a symbol that is exported from a project
	Export MonikerKind = "export" // line 13412
	// The moniker represents a symbol that is local to a project (e.g. a local
	// variable of a function, a class not visible outside the project, ...)
	Local MonikerKind = "local" // line 13417
	// A notebook cell kind.
	//
	// @since 3.17.0
	// A markup-cell is formatted source that is used for display.
	Markup NotebookCellKind = 1 // line 14070
	// A code-cell is source code.
	Code NotebookCellKind = 2 // line 14075
	// A set of predefined position encoding kinds.
	//
	// @since 3.17.0
	// Character offsets count UTF-8 code units (e.g. bytes).
	UTF8 PositionEncodingKind = "utf-8" // line 13849
	// Character offsets count UTF-16 code units.
	//
	// This is the default and must always be supported
	// by servers
	UTF16 PositionEncodingKind = "utf-16" // line 13854
	// Character offsets count UTF-32 code units.
	//
	// Implementation note: these are the same as Unicode codepoints,
	// so this `PositionEncodingKind` may also be used for an
	// encoding-agnostic representation of character offsets.
	UTF32 PositionEncodingKind = "utf-32" // line 13859
	// The client's default behavior is to select the identifier
	// according the to language's syntax rule.
	Identifier PrepareSupportDefaultBehavior = 1 // line 14144
	// Supports creating new files and folders.
	Create ResourceOperationKind = "create" // line 14091
	// Supports renaming existing files and folders.
	Rename ResourceOperationKind = "rename" // line 14096
	// Supports deleting existing files and folders.
	Delete ResourceOperationKind = "delete" // line 14101
	// A set of predefined token modifiers. This set is not fixed
	// an clients can specify additional token types via the
	// corresponding client capabilities.
	//
	// @since 3.16.0
	ModDeclaration    SemanticTokenModifiers = "declaration"    // line 13070
	ModDefinition     SemanticTokenModifiers = "definition"     // line 13074
	ModReadonly       SemanticTokenModifiers = "readonly"       // line 13078
	ModStatic         SemanticTokenModifiers = "static"         // line 13082
	ModDeprecated     SemanticTokenModifiers = "deprecated"     // line 13086
	ModAbstract       SemanticTokenModifiers = "abstract"       // line 13090
	ModAsync          SemanticTokenModifiers = "async"          // line 13094
	ModModification   SemanticTokenModifiers = "modification"   // line 13098
	ModDocumentation  SemanticTokenModifiers = "documentation"  // line 13102
	ModDefaultLibrary SemanticTokenModifiers = "defaultLibrary" // line 13106
	// A set of predefined token types. This set is not fixed
	// an clients can specify additional token types via the
	// corresponding client capabilities.
	//
	// @since 3.16.0
	NamespaceType SemanticTokenTypes = "namespace" // line 12963
	// Represents a generic type. Acts as a fallback for types which can't be mapped to
	// a specific type like class or enum.
	TypeType          SemanticTokenTypes = "type"          // line 12967
	ClassType         SemanticTokenTypes = "class"         // line 12972
	EnumType          SemanticTokenTypes = "enum"          // line 12976
	InterfaceType     SemanticTokenTypes = "interface"     // line 12980
	StructType        SemanticTokenTypes = "struct"        // line 12984
	TypeParameterType SemanticTokenTypes = "typeParameter" // line 12988
	ParameterType     SemanticTokenTypes = "parameter"     // line 12992
	VariableType      SemanticTokenTypes = "variable"      // line 12996
	PropertyType      SemanticTokenTypes = "property"      // line 13000
	EnumMemberType    SemanticTokenTypes = "enumMember"    // line 13004
	EventType         SemanticTokenTypes = "event"         // line 13008
	FunctionType      SemanticTokenTypes = "function"      // line 13012
	MethodType        SemanticTokenTypes = "method"        // line 13016
	MacroType         SemanticTokenTypes = "macro"         // line 13020
	KeywordType       SemanticTokenTypes = "keyword"       // line 13024
	ModifierType      SemanticTokenTypes = "modifier"      // line 13028
	CommentType       SemanticTokenTypes = "comment"       // line 13032
	StringType        SemanticTokenTypes = "string"        // line 13036
	NumberType        SemanticTokenTypes = "number"        // line 13040
	RegexpType        SemanticTokenTypes = "regexp"        // line 13044
	OperatorType      SemanticTokenTypes = "operator"      // line 13048
	// @since 3.17.0
	DecoratorType SemanticTokenTypes = "decorator" // line 13052
	// How a signature help was triggered.
	//
	// @since 3.15.0
	// Signature help was invoked manually by the user or by a command.
	SigInvoked SignatureHelpTriggerKind = 1 // line 14002
	// Signature help was triggered by a trigger character.
	SigTriggerCharacter SignatureHelpTriggerKind = 2 // line 14007
	// Signature help was triggered by the cursor moving or by the document content changing.
	SigContentChange SignatureHelpTriggerKind = 3 // line 14012
	// A symbol kind.
	File          SymbolKind = 1  // line 13241
	Module        SymbolKind = 2  // line 13245
	Namespace     SymbolKind = 3  // line 13249
	Package       SymbolKind = 4  // line 13253
	Class         SymbolKind = 5  // line 13257
	Method        SymbolKind = 6  // line 13261
	Property      SymbolKind = 7  // line 13265
	Field         SymbolKind = 8  // line 13269
	Constructor   SymbolKind = 9  // line 13273
	Enum          SymbolKind = 10 // line 13277
	Interface     SymbolKind = 11 // line 13281
	Function      SymbolKind = 12 // line 13285
	Variable      SymbolKind = 13 // line 13289
	Constant      SymbolKind = 14 // line 13293
	String        SymbolKind = 15 // line 13297
	Number        SymbolKind = 16 // line 13301
	Boolean       SymbolKind = 17 // line 13305
	Array         SymbolKind = 18 // line 13309
	Object        SymbolKind = 19 // line 13313
	Key           SymbolKind = 20 // line 13317
	Null          SymbolKind = 21 // line 13321
	EnumMember    SymbolKind = 22 // line 13325
	Struct        SymbolKind = 23 // line 13329
	Event         SymbolKind = 24 // line 13333
	Operator      SymbolKind = 25 // line 13337
	TypeParameter SymbolKind = 26 // line 13341
	// Symbol tags are extra annotations that tweak the rendering of a symbol.
	//
	// @since 3.16
	// Render a symbol as obsolete, usually using a strike-out.
	DeprecatedSymbol SymbolTag = 1 // line 13355
	// Represents reasons why a text document is saved.
	// Manually triggered, e.g. by the user pressing save, by starting debugging,
	// or by an API call.
	Manual TextDocumentSaveReason = 1 // line 13509
	// Automatic after a delay.
	AfterDelay TextDocumentSaveReason = 2 // line 13514
	// When the editor lost focus.
	FocusOut TextDocumentSaveReason = 3 // line 13519
	// Defines how the host (editor) should sync
	// document changes to the language server.
	// Documents should not be synced at all.
	None TextDocumentSyncKind = 0 // line 13484
	// Documents are synced by always sending the full content
	// of the document.
	Full TextDocumentSyncKind = 1 // line 13489
	// Documents are synced by sending the full content on open.
	// After that only incremental updates to the document are
	// send.
	Incremental TextDocumentSyncKind = 2          // line 13494
	Relative    TokenFormat          = "relative" // line 14158
	// Turn tracing off.
	Off TraceValues = "off" // line 13783
	// Trace messages only.
	Messages TraceValues = "messages" // line 13788
	// Verbose message tracing.
	Verbose TraceValues = "verbose" // line 13793
	// Moniker uniqueness level to define scope of the moniker.
	//
	// @since 3.16.0
	// The moniker is only unique inside a document
	Document UniquenessLevel = "document" // line 13371
	// The moniker is unique inside a project for which a dump got created
	Project UniquenessLevel = "project" // line 13376
	// The moniker is unique inside the group to which a project belongs
	Group UniquenessLevel = "group" // line 13381
	// The moniker is unique inside the moniker scheme.
	Scheme UniquenessLevel = "scheme" // line 13386
	// The moniker is globally unique
	Global UniquenessLevel = "global" // line 13391
	// Interested in create events.
	WatchCreate WatchKind = 1 // line 13901
	// Interested in change events
	WatchChange WatchKind = 2 // line 13906
	// Interested in delete events
	WatchDelete WatchKind = 4 // line 13911
)
