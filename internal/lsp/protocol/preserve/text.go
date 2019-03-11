// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains the corresponding structures to the
// "Text Synchronization" part of the LSP specification.

package protocol

type DidOpenTextDocumentParams struct {
	/**
	 * The document that was opened.
	 */
	TextDocument TextDocumentItem `json:"textDocument"`
}

type DidChangeTextDocumentParams struct {
	/**
	 * The document that did change. The version number points
	 * to the version after all provided content changes have
	 * been applied.
	 */
	TextDocument VersionedTextDocumentIdentifier `json:"textDocument"`

	/**
	 * The actual content changes. The content changes describe single state changes
	 * to the document. So if there are two content changes c1 and c2 for a document
	 * in state S10 then c1 move the document to S11 and c2 to S12.
	 */
	ContentChanges []TextDocumentContentChangeEvent `json:"contentChanges"`
}

/**
 * An event describing a change to a text document. If range and rangeLength are omitted
 * the new text is considered to be the full content of the document.
 */
type TextDocumentContentChangeEvent struct {
	/**
	 * The range of the document that changed.
	 */
	Range *Range `json:"range,omitempty"`

	/**
	 * The length of the range that got replaced.
	 */
	RangeLength float64 `json:"rangeLength,omitempty"`

	/**
	 * The new text of the range/document.
	 */
	Text string `json:"text"`
}

/**
 * Describe options to be used when registering for text document change events.
 */
type TextDocumentChangeRegistrationOptions struct {
	TextDocumentRegistrationOptions
	/**
	 * How documents are synced to the server. See TextDocumentSyncKind.Full
	 * and TextDocumentSyncKind.Incremental.
	 */
	SyncKind float64 `json:"syncKind"`
}

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

/**
 * Represents reasons why a text document is saved.
 */
type TextDocumentSaveReason float64

const (
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
)

type DidSaveTextDocumentParams struct {
	/**
	 * The document that was saved.
	 */
	TextDocument TextDocumentIdentifier `json:"textDocument"`

	/**
	 * Optional the content when saved. Depends on the includeText value
	 * when the save notification was requested.
	 */
	Text string `json:"text,omitempty"`
}

type TextDocumentSaveRegistrationOptions struct {
	TextDocumentRegistrationOptions
	/**
	 * The client is supposed to include the content on save.
	 */
	IncludeText bool `json:"includeText,omitempty"`
}

type DidCloseTextDocumentParams struct {
	/**
	 * The document that was closed.
	 */
	TextDocument TextDocumentIdentifier `json:"textDocument"`
}
