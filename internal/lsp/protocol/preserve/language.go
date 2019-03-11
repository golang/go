// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains the corresponding structures to the
// "Language Features" part of the LSP specification.

package protocol

type CompletionParams struct {
	TextDocumentPositionParams

	/**
	 * The completion context. This is only available if the client specifies
	 * to send this using `ClientCapabilities.textDocument.completion.contextSupport === true`
	 */
	Context CompletionContext `json:"context,omitempty"`
}

/**
 * How a completion was triggered
 */
type CompletionTriggerKind float64

const (
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
	 * Completion was re-triggered as the current completion list is incomplete.
	 */
	TriggerForIncompleteCompletions CompletionTriggerKind = 3
)

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
 * Represents a collection of [completion items](#CompletionItem) to be presented
 * in the editor.
 */
type CompletionList struct {
	/**
	 * This list it not complete. Further typing should result in recomputing
	 * this list.
	 */
	IsIncomplete bool `json:"isIncomplete"`

	/**
	 * The completion items.
	 */
	Items []CompletionItem `json:"items"`
}

/**
 * Defines whether the insert text in a completion item should be interpreted as
 * plain text or a snippet.
 */
type InsertTextFormat float64

const (
	/**
	 * The primary text to be inserted is treated as a plain string.
	 */
	PlainTextFormat InsertTextFormat = 1

	/**
	 * The primary text to be inserted is treated as a snippet.
	 *
	 * A snippet can define tab stops and placeholders with `$1`, `$2`
	 * and `${3:foo}`. `$0` defines the final tab stop, it defaults to
	 * the end of the snippet. Placeholders with equal identifiers are linked,
	 * that is typing in one will update others too.
	 */
	SnippetTextFormat InsertTextFormat = 2
)

type CompletionItem struct {
	/**
	 * The label of this completion item. By default
	 * also the text that is inserted when selecting
	 * this completion.
	 */
	Label string `json:"label"`

	/**
	 * The kind of this completion item. Based of the kind
	 * an icon is chosen by the editor.
	 */
	Kind float64 `json:"kind,omitempty"`

	/**
	 * A human-readable string with additional information
	 * about this item, like type or symbol information.
	 */
	Detail string `json:"detail,omitempty"`

	/**
	 * A human-readable string that represents a doc-comment.
	 */
	Documentation interface{} `json:"documentation,omitempty"` // string | MarkupContent

	/**
	 * Indicates if this item is deprecated.
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
	 * with other items. When `falsy` the label is used.
	 */
	SortText string `json:"sortText,omitempty"`

	/**
	 * A string that should be used when filtering a set of
	 * completion items. When `falsy` the label is used.
	 */
	FilterText string `json:"filterText,omitempty"`

	/**
	 * A string that should be inserted into a document when selecting
	 * this completion. When `falsy` the label is used.
	 *
	 * The `insertText` is subject to interpretation by the client side.
	 * Some tools might not take the string literally. For example
	 * VS Code when code complete is requested in this example `con<cursor position>`
	 * and a completion item with an `insertText` of `console` is provided it
	 * will only insert `sole`. Therefore it is recommended to use `textEdit` instead
	 * since it avoids additional client side interpretation.
	 *
	 * @deprecated Use textEdit instead.
	 */
	InsertText string `json:"insertText,omitempty"`

	/**
	 * The format of the insert text. The format applies to both the `insertText` property
	 * and the `newText` property of a provided `textEdit`.
	 */
	InsertTextFormat InsertTextFormat `json:"insertTextFormat,omitempty"`

	/**
	 * An edit which is applied to a document when selecting this completion. When an edit is provided the value of
	 * `insertText` is ignored.
	 *
	 * *Note:* The range of the edit must be a single line range and it must contain the position at which completion
	 * has been requested.
	 */
	TextEdit *TextEdit `json:"textEdit,omitempty"`

	/**
	 * An optional array of additional text edits that are applied when
	 * selecting this completion. Edits must not overlap (including the same insert position)
	 * with the main edit nor with themselves.
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
	 * An optional command that is executed *after* inserting this completion. *Note* that
	 * additional modifications to the current document should be described with the
	 * additionalTextEdits-property.
	 */
	Command *Command `json:"command,omitempty"`

	/**
	 * An data entry field that is preserved on a completion item between
	 * a completion and a completion resolve request.
	 */
	Data interface{} `json:"data"`
}

/**
 * The kind of a completion entry.
 */
type CompletionItemKind float64

const (
	TextCompletion          CompletionItemKind = 1
	MethodCompletion        CompletionItemKind = 2
	FunctionCompletion      CompletionItemKind = 3
	ConstructorCompletion   CompletionItemKind = 4
	FieldCompletion         CompletionItemKind = 5
	VariableCompletion      CompletionItemKind = 6
	ClassCompletion         CompletionItemKind = 7
	InterfaceCompletion     CompletionItemKind = 8
	ModuleCompletion        CompletionItemKind = 9
	PropertyCompletion      CompletionItemKind = 10
	UnitCompletion          CompletionItemKind = 11
	ValueCompletion         CompletionItemKind = 12
	EnumCompletion          CompletionItemKind = 13
	KeywordCompletion       CompletionItemKind = 14
	SnippetCompletion       CompletionItemKind = 15
	ColorCompletion         CompletionItemKind = 16
	FileCompletion          CompletionItemKind = 17
	ReferenceCompletion     CompletionItemKind = 18
	FolderCompletion        CompletionItemKind = 19
	EnumMemberCompletion    CompletionItemKind = 20
	ConstantCompletion      CompletionItemKind = 21
	StructCompletion        CompletionItemKind = 22
	EventCompletion         CompletionItemKind = 23
	OperatorCompletion      CompletionItemKind = 24
	TypeParameterCompletion CompletionItemKind = 25
)

type CompletionRegistrationOptions struct {
	TextDocumentRegistrationOptions
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
	 * The server provides support to resolve additional
	 * information for a completion item.
	 */
	ResolveProvider bool `json:"resolveProvider,omitempty"`
}

/**
 * The result of a hover request.
 */
type Hover struct {
	/**
	 * The hover's content
	 */
	Contents MarkupContent `json:"contents"`

	/**
	 * An optional range is a range inside a text document
	 * that is used to visualize a hover, e.g. by changing the background color.
	 */
	Range Range `json:"range,omitempty"`
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
	 * `signatures.length === 0`. Whenever possible implementors should
	 * make an active decision about the active signature and shouldn't
	 * rely on a default value.
	 * In future version of the protocol this property might become
	 * mandatory to better express this.
	 */
	ActiveSignature float64 `json:"activeSignature,omitempty"`

	/**
	 * The active parameter of the active signature. If omitted or the value
	 * lies outside the range of `signatures[activeSignature].parameters`
	 * defaults to 0 if the active signature has parameters. If
	 * the active signature has no parameters it is ignored.
	 * In future version of the protocol this property might become
	 * mandatory to better express the active parameter if the
	 * active signature does have any.
	 */
	ActiveParameter float64 `json:"activeParameter,omitempty"`
}

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
	Documentation interface{} `json:"documentation,omitempty"` // string | MarkupContent

	/**
	 * The parameters of this signature.
	 */
	Parameters []ParameterInformation `json:"parameters,omitempty"`
}

/**
 * Represents a parameter of a callable-signature. A parameter can
 * have a label and a doc-comment.
 */
type ParameterInformation struct {
	/**
	 * The label of this parameter. Will be shown in
	 * the UI.
	 */
	Label string `json:"label"`

	/**
	 * The human-readable doc-comment of this parameter. Will be shown
	 * in the UI but can be omitted.
	 */
	Documentation interface{} `json:"documentation,omitempty"` // string | MarkupContent
}

type SignatureHelpRegistrationOptions struct {
	TextDocumentRegistrationOptions
	/**
	 * The characters that trigger signature help
	 * automatically.
	 */
	TriggerCharacters []string `json:"triggerCharacters,omitempty"`
}

type ReferenceParams struct {
	TextDocumentPositionParams
	Context ReferenceContext
}

type ReferenceContext struct {
	/**
	 * Include the declaration of the current symbol.
	 */
	IncludeDeclaration bool `json:"includeDeclaration"`
}

/**
 * A document highlight is a range inside a text document which deserves
 * special attention. Usually a document highlight is visualized by changing
 * the background color of its range.
 *
 */
type DocumentHighlight struct {
	/**
	 * The range this highlight applies to.
	 */
	Range Range `json:"range"`

	/**
	 * The highlight kind, default is DocumentHighlightKind.Text.
	 */
	Kind float64 `json:"kind,omitempty"`
}

/**
 * A document highlight kind.
 */
type DocumentHighlightKind float64

const (
	/**
	 * A textual occurrence.
	 */
	TextHighlight DocumentHighlightKind = 1

	/**
	 * Read-access of a symbol, like reading a variable.
	 */
	ReadHighlight DocumentHighlightKind = 2

	/**
	 * Write-access of a symbol, like writing to a variable.
	 */
	WriteHighlight DocumentHighlightKind = 3
)

type DocumentSymbolParams struct {
	/**
	 * The text document.
	 */
	TextDocument TextDocumentIdentifier `json:"textDocument"`
}

/**
 * A symbol kind.
 */
type SymbolKind float64

const (
	FileSymbol          SymbolKind = 1
	ModuleSymbol        SymbolKind = 2
	NamespaceSymbol     SymbolKind = 3
	PackageSymbol       SymbolKind = 4
	ClassSymbol         SymbolKind = 5
	MethodSymbol        SymbolKind = 6
	PropertySymbol      SymbolKind = 7
	FieldSymbol         SymbolKind = 8
	ConstructorSymbol   SymbolKind = 9
	EnumSymbol          SymbolKind = 10
	InterfaceSymbol     SymbolKind = 11
	FunctionSymbol      SymbolKind = 12
	VariableSymbol      SymbolKind = 13
	ConstantSymbol      SymbolKind = 14
	StringSymbol        SymbolKind = 15
	NumberSymbol        SymbolKind = 16
	BooleanSymbol       SymbolKind = 17
	ArraySymbol         SymbolKind = 18
	ObjectSymbol        SymbolKind = 19
	KeySymbol           SymbolKind = 20
	NullSymbol          SymbolKind = 21
	EnumMemberSymbol    SymbolKind = 22
	StructSymbol        SymbolKind = 23
	EventSymbol         SymbolKind = 24
	OperatorSymbol      SymbolKind = 25
	TypeParameterSymbol SymbolKind = 26
)

/**
 * Represents programming constructs like variables, classes, interfaces etc. that appear in a document. Document symbols can be
 * hierarchical and they have two ranges: one that encloses its definition and one that points to its most interesting range,
 * e.g. the range of an identifier.
 */
type DocumentSymbol struct {

	/**
	 * The name of this symbol.
	 */
	Name string `json:"name"`

	/**
	 * More detail for this symbol, e.g the signature of a function. If not provided the
	 * name is used.
	 */
	Detail string `json:"detail,omitempty"`

	/**
	 * The kind of this symbol.
	 */
	Kind SymbolKind `json:"kind"`

	/**
	 * Indicates if this symbol is deprecated.
	 */
	Deprecated bool `json:"deprecated,omitempty"`

	/**
	 * The range enclosing this symbol not including leading/trailing whitespace but everything else
	 * like comments. This information is typically used to determine if the clients cursor is
	 * inside the symbol to reveal in the symbol in the UI.
	 */
	Range Range `json:"range"`

	/**
	 * The range that should be selected and revealed when this symbol is being picked, e.g the name of a function.
	 * Must be contained by the `range`.
	 */
	SelectionRange Range `json:"selectionRange"`

	/**
	 * Children of this symbol, e.g. properties of a class.
	 */
	Children []DocumentSymbol `json:"children,omitempty"`
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
	Kind float64 `json:"kind"`

	/**
	 * Indicates if this symbol is deprecated.
	 */
	Deprecated bool `json:"deprecated,omitempty"`

	/**
	 * The location of this symbol. The location's range is used by a tool
	 * to reveal the location in the editor. If the symbol is selected in the
	 * tool the range's start information is used to position the cursor. So
	 * the range usually spans more then the actual symbol's name and does
	 * normally include things like visibility modifiers.
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
 * Params for the CodeActionRequest
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
}

/**
 * The kind of a code action.
 *
 * Kinds are a hierarchical list of identifiers separated by `.`, e.g. `"refactor.extract.function"`.
 *
 * The set of kinds is open and client needs to announce the kinds it supports to the server during
 * initialization.
 */
type CodeActionKind string

/**
 * A set of predefined code action kinds
 */
const (
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
)

/**
 * Contains additional diagnostic information about the context in which
 * a code action is run.
 */
type CodeActionContext struct {
	/**
	 * An array of diagnostics.
	 */
	Diagnostics []Diagnostic `json:"diagnostics"`

	/**
	 * Requested kind of actions to return.
	 *
	 * Actions not of this kind are filtered out by the client before being shown. So servers
	 * can omit computing them.
	 */
	Only []CodeActionKind `json:"only,omitempty"`
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
	 * The workspace edit this code action performs.
	 */
	Edit WorkspaceEdit `json:"edit,omitempty"`

	/**
	 * A command this code action executes. If a code action
	 * provides an edit and a command, first the edit is
	 * executed and then the command.
	 */
	Command Command `json:"command,omitempty"`
}

type CodeLensParams struct {
	/**
	 * The document to request code lens for.
	 */
	TextDocument TextDocumentIdentifier `json:"textDocument"`
}

/**
 * A code lens represents a command that should be shown along with
 * source text, like the number of references, a way to run tests, etc.
 *
 * A code lens is _unresolved_ when no command is associated to it. For performance
 * reasons the creation of a code lens and resolving should be done in two stages.
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
	 * a code lens and a code lens resolve request.
	 */
	Data interface{} `json:"data"`
}

type CodeLensRegistrationOptions struct {
	TextDocumentRegistrationOptions
	/**
	 * Code lens has a resolve provider as well.
	 */
	ResolveProvider bool `json:"resolveProvider,omitempty"`
}

type DocumentLinkParams struct {
	/**
	 * The document to provide document links for.
	 */
	TextDocument TextDocumentIdentifier `json:"textDocument"`
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
	 * The uri this link points to. If missing a resolve request is sent later.
	 */
	Target DocumentURI `json:"target,omitempty"`
	/**
	 * A data entry field that is preserved on a document link between a
	 * DocumentLinkRequest and a DocumentLinkResolveRequest.
	 */
	Data interface{} `json:"data,omitempty"`
}

type DocumentLinkRegistrationOptions struct {
	TextDocumentRegistrationOptions
	/**
	 * Document links have a resolve provider as well.
	 */
	ResolveProvider bool `json:"resolveProvider,omitempty"`
}

type DocumentColorParams struct {
	/**
	 * The text document.
	 */
	TextDocument TextDocumentIdentifier `json:"textDocument"`
}

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

/**
 * Represents a color in RGBA space.
 */
type Color struct {

	/**
	 * The red component of this color in the range [0-1].
	 */
	Red float64 `json:"red"`

	/**
	 * The green component of this color in the range [0-1].
	 */
	Green float64 `json:"green"`

	/**
	 * The blue component of this color in the range [0-1].
	 */
	Blue float64 `json:"blue"`

	/**
	 * The alpha component of this color in the range [0-1].
	 */
	Alpha float64 `json:"alpha"`
}

type ColorPresentationParams struct {
	/**
	 * The text document.
	 */
	TextDocument TextDocumentIdentifier `json:"textDocument"`

	/**
	 * The color information to request presentations for.
	 */
	Color Color `json:"color"`

	/**
	 * The range where the color would be inserted. Serves as a context.
	 */
	Range Range `json:"range"`
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

type DocumentFormattingParams struct {
	/**
	 * The document to format.
	 */
	TextDocument TextDocumentIdentifier `json:"textDocument"`

	/**
	 * The format options.
	 */
	Options FormattingOptions `json:"options"`
}

/**
 * Value-object describing what options formatting should use.
 */
type FormattingOptions struct {
	/**
	 * Size of a tab in spaces.
	 */
	TabSize float64 `json:"tabSize"`

	/**
	 * Prefer spaces over tabs.
	 */
	InsertSpaces bool `json:"insertSpaces"`

	/**
	 * Signature for further properties.
	 */
	// TODO: [key: string]: boolean | number | string;
}

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
}

type DocumentOnTypeFormattingParams struct {
	/**
	 * The document to format.
	 */
	TextDocument TextDocumentIdentifier `json:"textDocument"`

	/**
	 * The position at which this request was sent.
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

type DocumentOnTypeFormattingRegistrationOptions struct {
	TextDocumentRegistrationOptions
	/**
	 * A character on which formatting should be triggered, like `}`.
	 */
	FirstTriggerCharacter string `json:"firstTriggerCharacter"`
	/**
	 * More trigger characters.
	 */
	MoreTriggerCharacter []string `json:"moreTriggerCharacter"`
}

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
}

type FoldingRangeRequestParam struct {
	/**
	 * The text document.
	 */
	TextDocument TextDocumentIdentifier `json:"textDocument"`
}

/**
 * Enum of known range kinds
 */
type FoldingRangeKind string

const (
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
)

/**
 * Represents a folding range.
 */
type FoldingRange struct {

	/**
	 * The zero-based line number from where the folded range starts.
	 */
	StartLine float64 `json:"startLine"`

	/**
	 * The zero-based character offset from where the folded range starts. If not defined, defaults to the length of the start line.
	 */
	StartCharacter float64 `json:"startCharacter,omitempty"`

	/**
	 * The zero-based line number where the folded range ends.
	 */
	EndLine float64 `json:"endLine"`

	/**
	 * The zero-based character offset before the folded range ends. If not defined, defaults to the length of the end line.
	 */
	EndCharacter float64 `json:"endCharacter,omitempty"`

	/**
	 * Describes the kind of the folding range such as `comment' or 'region'. The kind
	 * is used to categorize folding ranges and used by commands like 'Fold all comments'. See
	 * [FoldingRangeKind](#FoldingRangeKind) for an enumeration of standardized kinds.
	 */
	Kind string `json:"kind,omitempty"`
}
