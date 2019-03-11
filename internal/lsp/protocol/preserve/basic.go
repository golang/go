// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains the corresponding structures to the
// "Basic JSON Structures" part of the LSP specification.

package protocol

const (
	// CodeRequestCancelled is the error code that is returned when a request is
	// cancelled early.
	CodeRequestCancelled = -32800
)

// DocumentURI represents the URI of a document.
// Many of the interfaces contain fields that correspond to the URI of a document.
// For clarity, the type of such a field is declared as a DocumentURI.
// Over the wire, it will still be transferred as a string, but this guarantees
// that the contents of that string can be parsed as a valid URI.
type DocumentURI string

// Position in a text document expressed as zero-based line and zero-based character offset.
// A position is between two characters like an ‘insert’ cursor in a editor.
type Position struct {
	/**
	 * Line position in a document (zero-based).
	 */
	Line float64 `json:"line"`

	/**
	 * Character offset on a line in a document (zero-based). Assuming that the line is
	 * represented as a string, the `character` value represents the gap between the
	 * `character` and `character + 1`.
	 *
	 * If the character value is greater than the line length it defaults back to the
	 * line length.
	 */
	Character float64 `json:"character"`
}

// Range in a text document expressed as (zero-based) start and end positions.
// A range is comparable to a selection in an editor.
// Therefore the end position is exclusive.
// If you want to specify a range that contains a line including the line
// ending character(s) then use an end position denoting the start of the next
// line.
type Range struct {
	/**
	 * The range's start position.
	 */
	Start Position `json:"start"`

	/**
	 * The range's end position.
	 */
	End Position `json:"end"`
}

// Location represents a location inside a resource, such as a line inside a text file.
type Location struct {
	URI   DocumentURI `json:"uri"`
	Range Range       `json:"range"`
}

// LocationLink rerpesents a link betwee a source and a target location.AfterDelay
type LocationLink struct {
	/**
	 * Span of the origin of this link.
	 *
	 * Used as the underlined span for mouse interaction. Defaults to the word range at
	 * the mouse position.
	 */
	OriginSelectionRange *Range `json:"originSelectionRange,omitempty"`

	/**
	 * The target resource identifier of this link.
	 */
	TargetURI string `json:"targetUri"`

	/**
	 * The full target range of this link.
	 */
	TargetRange Range `json:"targetRange"`

	/**
	 * The span of this link.
	 */
	TargetSelectionRange *Range `json:"targetSeletionRange,omitempty"`
}

// Diagnostic represents a diagnostic, such as a compiler error or warning.
// Diagnostic objects are only valid in the scope of a resource.
type Diagnostic struct {
	/**
	 * The range at which the message applies.
	 */
	Range Range `json:"range"`

	/**
	 * The diagnostic's severity. Can be omitted. If omitted it is up to the
	 * client to interpret diagnostics as error, warning, info or hint.
	 */
	Severity DiagnosticSeverity `json:"severity,omitempty"`

	/**
	 * The diagnostic's code, which might appear in the user interface.
	 */
	Code interface{} `json:"code,omitempty"` // number | string

	/**
	 * A human-readable string describing the source of this
	 * diagnostic, e.g. 'typescript' or 'super lint'.
	 */
	Source string `json:"source,omitempty"`

	/**
	 * The diagnostic's message.
	 */
	Message string `json:"message"`

	/**
	 * An array of related diagnostic information, e.g. when symbol-names within
	 * a scope collide all definitions can be marked via this property.
	 */
	Related []DiagnosticRelatedInformation `json:"relatedInformation,omitempty"`
}

// DiagnosticSeverity indicates the severity of a Diagnostic message.
type DiagnosticSeverity float64

const (
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
)

// DiagnosticRelatedInformation represents a related message and source code
// location for a diagnostic.
// This should be used to point to code locations that cause or related to a
// diagnostics, e.g when duplicating a symbol in a scope.
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

// Command represents a reference to a command.
// Provides a title which will be used to represent a command in the UI.
// Commands are identified by a string identifier.
// The protocol currently doesn’t specify a set of well-known commands.
// So executing a command requires some tool extension code.
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
	Arguments []interface{} `json:"arguments,omitempty"`
}

// TextEdit is a textual edit applicable to a text document.
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

// TextDocumentEdit describes textual changes on a single text document.
// The text document is referred to as a VersionedTextDocumentIdentifier to
// allow clients to check the text document version before an edit is applied.
// A TextDocumentEdit describes all changes on a version Si and after they are
// applied move the document to version Si+1.
// So the creator of a TextDocumentEdit doesn’t need to sort the array or do
// any kind of ordering.
// However the edits must be non overlapping.
type TextDocumentEdit struct {
	/**
	 * The text document to change.
	 */
	TextDocument VersionedTextDocumentIdentifier `json:"textDocument"`

	/**
	 * The edits to be applied.
	 */
	Edits []TextEdit `json:"edits"`
}

// WorkspaceEdit represents changes to many resources managed in the workspace.
// The edit should either provide Changes or DocumentChanges.
// If the client can handle versioned document edits and if DocumentChanges are
// present, the latter are preferred over Changes.
type WorkspaceEdit struct {
	/**
	 * Holds changes to existing resources.
	 */
	Changes map[DocumentURI][]TextEdit `json:"changes,omitempty"`

	/**
	 * An array of `TextDocumentEdit`s to express changes to n different text documents
	 * where each text document edit addresses a specific version of a text document.
	 * Whether a client supports versioned document edits is expressed via
	 * `WorkspaceClientCapabilities.workspaceEdit.documentChanges`.
	 */
	DocumentChanges []TextDocumentEdit `json:"documentChanges,omitempty"`
}

// TextDocumentIdentifier identifies a document using a URI.
// On the protocol level, URIs are passed as strings.
// The corresponding JSON structure looks like this.
type TextDocumentIdentifier struct {
	/**
	 * The text document's URI.
	 */
	URI DocumentURI `json:"uri"`
}

// TextDocumentItem is an item to transfer a text document from the client to
// the server.
type TextDocumentItem struct {
	/**
	 * The text document's URI.
	 */
	URI DocumentURI `json:"uri"`

	/**
	 * The text document's language identifier.
	 */
	LanguageID string `json:"languageId"`

	/**
	 * The version number of this document (it will increase after each
	 * change, including undo/redo).
	 */
	Version float64 `json:"version"`

	/**
	 * The content of the opened text document.
	 */
	Text string `json:"text"`
}

// VersionedTextDocumentIdentifier is an identifier to denote a specific version of a text document.
type VersionedTextDocumentIdentifier struct {
	TextDocumentIdentifier

	/**
	 * The version number of this document. If a versioned text document identifier
	 * is sent from the server to the client and the file is not open in the editor
	 * (the server has not received an open notification before) the server can send
	 * `null` to indicate that the version is known and the content on disk is the
	 * truth (as speced with document content ownership)
	 */
	Version *uint64 `json:"version"`
}

// TextDocumentPositionParams is a parameter literal used in requests to pass
// a text document and a position inside that document.
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

// DocumentFilter is a document filter denotes a document through properties
// like language, scheme or pattern.
// An example is a filter that applies to TypeScript files on disk.
// Another example is a filter the applies to JSON files with name package.json:
//     { language: 'typescript', scheme: 'file' }
//     { language: 'json', pattern: '**/package.json' }
type DocumentFilter struct {
	/**
	 * A language id, like `typescript`.
	 */
	Language string `json:"language,omitempty"`

	/**
	 * A URI [scheme](#URI.scheme), like `file` or `untitled`.
	 */
	Scheme string `json:"scheme,omitempty"`

	/**
	 * A glob pattern, like `*.{ts,js}`.
	 */
	Pattern string `json:"pattern,omitempty"`
}

// A document selector is the combination of one or more document filters.
type DocumentSelector []DocumentFilter

/**
 * Describes the content type that a client supports in various
 * result literals like `Hover`, `ParameterInfo` or `CompletionItem`.
 *
 * Please note that `MarkupKinds` must not start with a `$`. This kinds
 * are reserved for internal usage.
 */
type MarkupKind string

const (
	/**
	 * Plain text is supported as a content format
	 */
	PlainText MarkupKind = "plaintext"

	/**
	 * Markdown is supported as a content format
	 */
	Markdown MarkupKind = "markdown"
)

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
