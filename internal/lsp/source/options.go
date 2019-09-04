// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import "golang.org/x/tools/internal/lsp/protocol"

var (
	DefaultSessionOptions = SessionOptions{
		TextDocumentSyncKind: protocol.Incremental,
		HoverKind:            SynopsisDocumentation,
		InsertTextFormat:     protocol.PlainTextTextFormat,
		SupportedCodeActions: map[FileKind]map[protocol.CodeActionKind]bool{
			Go: {
				protocol.SourceOrganizeImports: true,
				protocol.QuickFix:              true,
			},
			Mod: {},
			Sum: {},
		},
		Completion: CompletionOptions{
			Documentation: true,
			Deep:          true,
			FuzzyMatching: true,
		},
	}
	DefaultViewOptions = ViewOptions{}
)

type SessionOptions struct {
	Env              []string
	BuildFlags       []string
	HoverKind        HoverKind
	DisabledAnalyses map[string]struct{}

	WatchFileChanges              bool
	InsertTextFormat              protocol.InsertTextFormat
	ConfigurationSupported        bool
	DynamicConfigurationSupported bool
	DynamicWatchedFilesSupported  bool
	PreferredContentFormat        protocol.MarkupKind
	LineFoldingOnly               bool

	SupportedCodeActions map[FileKind]map[protocol.CodeActionKind]bool

	TextDocumentSyncKind protocol.TextDocumentSyncKind

	Completion CompletionOptions
}

type ViewOptions struct {
}

type CompletionOptions struct {
	Deep              bool
	FuzzyMatching     bool
	Unimported        bool
	Documentation     bool
	FullDocumentation bool
	Placeholders      bool
}

type HoverKind int

const (
	SingleLine = HoverKind(iota)
	NoDocumentation
	SynopsisDocumentation
	FullDocumentation

	// structured is an experimental setting that returns a structured hover format.
	// This format separates the signature from the documentation, so that the client
	// can do more manipulation of these fields.
	//
	// This should only be used by clients that support this behavior.
	Structured
)
