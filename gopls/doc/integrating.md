# Documentation for plugin authors

If you are integrating `gopls` into an editor by writing an editor plugin, there are quite a few semantics of the communication between the editor and `gopls` that are not specified by the [LSP specification].

We attempt to document those details along with any other information that has been helpful to other plugin authors here.

If you are implementing a plugin yourself and have questions this page does not answer, please reach out to us to ask, and then also contribute your findings back to this page.

## Supported features

For the most part you should look at the [list](status.md#supported-features) in the current status document to know if gopls supports a feature.
For a truly authoritative answer you should check the [result][InitializeResult] of the [initialize] request, where gopls enumerates its support in the [ServerCapabilities].


## Positions and ranges

Many LSP requests pass position or range information. This is described in the [LSP specification][lsp-text-documents]:

> A position inside a document (see Position definition below) is expressed as a zero-based line and character offset. The offsets are based on a UTF-16 string representation. So a string of the form a𐐀b the character offset of the character a is 0, the character offset of 𐐀 is 1 and the character offset of b is 3 since 𐐀 is represented using two code units in UTF-16.

This means that integrators will need to calculate UTF-16 based column offsets.

[`golang.org/x/tools/internal/span`] has the code to do this in go.
[#31080] tracks making `span` and other useful packages non-internal.

## Edits

In order to deliver changes from gopls to the editor, the LSP supports arrays of [`TextEdit`][lsp-textedit]s in responses.
The spec specifies exactly how these should be applied:

> All text edits ranges refer to positions in the original document. Text edits ranges must never overlap, that means no part of the original document must be manipulated by more than one edit. However, it is possible that multiple edits have the same start position: multiple inserts, or any number of inserts followed by a single remove or replace edit. If multiple inserts have the same position, the order in the array defines the order in which the inserted strings appear in the resulting text.

All `[]TextEdit` are sorted such that applying the array of deltas received in reverse order achieves the desired result that holds with the spec.

## Errors

Various error codes are described in the [LSP specification][lsp-response]. We are still determining what it means for a method to return an error; are errors only for low-level LSP/transport issues or can other conditions cause errors to be returned? See some of this discussion on [#31526].

The method chosen is currently influenced by the exact treatment in the currently popular editor integrations. It may well change, and ideally would become more coherent across requests.

* [`textDocument/codeAction`]: Return error if there was an error computing code actions.
* [`textDocument/completion`]: Log errors, return empty result list.
* [`textDocument/definition`]: Return error if there was an error computing the definition for the position.
* [`textDocument/typeDefinition`]: Return error if there was an error computing the type definition for the position.
* [`textDocument/formatting`]: Return error if there was an error formatting the file.
* [`textDocument/highlight`]: Log errors, return empty result.
* [`textDocument/hover`]: Return empty result.
* [`textDocument/documentLink`]: Log errors, return nil result.
* [`textDocument/publishDiagnostics`]: Log errors if there were any while computing diagnostics.
* [`textDocument/references`]: Log errors, return empty result.
* [`textDocument/rename`]: Return error if there was an error computing renames.
* [`textDocument/signatureHelp`]: Log errors, return nil result.
* [`textDocument/documentSymbols`]: Return error if there was an error computing document symbols.

## Watching files

It is fairly normal for files that affect `gopls` to be modified outside of the editor it is associated with.

For instance, files that are needed to do correct type checking are modified by switching branches in git, or updated by a code generator.

Monitoring files inside gopls directly has a lot of awkward problems, but the [LSP specification] has methods that allow gopls to request that the client notify it of file system changes, specifically [`workspace/didChangeWatchedFiles`].
This is currently being added to gopls by a community member, and tracked in [#31553]

[InitializeResult]: https://godoc.org/golang.org/x/tools/internal/lsp/protocol#InitializeResult
[ServerCapabilities]: https://godoc.org/golang.org/x/tools/internal/lsp/protocol#ServerCapabilities
[`golang.org/x/tools/internal/span`]: https://godoc.org/golang.org/x/tools/internal/span#NewPoint

[LSP specification]: https://microsoft.github.io/language-server-protocol/specifications/specification-3-14/
[lsp-response]: https://github.com/Microsoft/language-server-protocol/blob/gh-pages/_specifications/specification-3-14.md#response-message
[initialize]: https://microsoft.github.io/language-server-protocol/specifications/specification-3-14/#initialize
[lsp-text-documents]: https://github.com/Microsoft/language-server-protocol/blob/gh-pages/_specifications/specification-3-14.md#text-documents
[lsp-textedit]: https://github.com/Microsoft/language-server-protocol/blob/gh-pages/_specifications/specification-3-14.md#textedit

[`textDocument/codeAction`]: https://github.com/Microsoft/language-server-protocol/blob/gh-pages/_specifications/specification-3-14.md#textDocument_codeAction
[`textDocument/completion`]: https://github.com/Microsoft/language-server-protocol/blob/gh-pages/_specifications/specification-3-14.md#textDocument_completion
[`textDocument/definition`]: https://github.com/Microsoft/language-server-protocol/blob/gh-pages/_specifications/specification-3-14.md#textDocument_definition
[`textDocument/typeDefinition`]: https://github.com/Microsoft/language-server-protocol/blob/gh-pages/_specifications/specification-3-14.md#textDocument_typeDefinition
[`textDocument/formatting`]: https://github.com/Microsoft/language-server-protocol/blob/gh-pages/_specifications/specification-3-14.md#textDocument_formatting
[`textDocument/highlight`]: https://github.com/Microsoft/language-server-protocol/blob/gh-pages/_specifications/specification-3-14.md#textDocument_highlight
[`textDocument/hover`]: https://github.com/Microsoft/language-server-protocol/blob/gh-pages/_specifications/specification-3-14.md#textDocument_hover
[`textDocument/documentLink`]: https://github.com/Microsoft/language-server-protocol/blob/gh-pages/_specifications/specification-3-14.md#textDocument_documentLink
[`textDocument/publishDiagnostics`]: https://github.com/Microsoft/language-server-protocol/blob/gh-pages/_specifications/specification-3-14.md#textDocument_publishDiagnostics
[`textDocument/references`]: https://github.com/Microsoft/language-server-protocol/blob/gh-pages/_specifications/specification-3-14.md#textDocument_references
[`textDocument/rename`]: https://github.com/Microsoft/language-server-protocol/blob/gh-pages/_specifications/specification-3-14.md#textDocument_rename
[`textDocument/signatureHelp`]: https://github.com/Microsoft/language-server-protocol/blob/gh-pages/_specifications/specification-3-14.md#textDocument_signatureHelp
[`textDocument/documentSymbols`]: https://github.com/Microsoft/language-server-protocol/blob/gh-pages/_specifications/specification-3-14.md#textDocument_documentSymbols
[`workspace/didChangeWatchedFiles`]: https://github.com/Microsoft/language-server-protocol/blob/gh-pages/_specifications/specification-3-14.md#workspace_didChangeWatchedFiles

[#31080]: https://github.com/golang/go/issues/31080
[#31553]: https://github.com/golang/go/issues/31553
[#31526]: https://github.com/golang/go/issues/31526