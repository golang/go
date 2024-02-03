// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package analysis

import "go/token"

// A Diagnostic is a message associated with a source location or range.
//
// An Analyzer may return a variety of diagnostics; the optional Category,
// which should be a constant, may be used to classify them.
// It is primarily intended to make it easy to look up documentation.
//
// If End is provided, the diagnostic is specified to apply to the range between
// Pos and End.
type Diagnostic struct {
	Pos      token.Pos
	End      token.Pos // optional
	Category string    // optional
	Message  string

	// URL is the optional location of a web page that provides
	// additional documentation for this diagnostic.
	//
	// If URL is empty but a Category is specified, then the
	// Analysis driver should treat the URL as "#"+Category.
	//
	// The URL may be relative. If so, the base URL is that of the
	// Analyzer that produced the diagnostic;
	// see https://pkg.go.dev/net/url#URL.ResolveReference.
	URL string

	// SuggestedFixes is an optional list of fixes to address the
	// problem described by the diagnostic, each one representing
	// an alternative strategy; at most one may be applied.
	SuggestedFixes []SuggestedFix

	// Related contains optional secondary positions and messages
	// related to the primary diagnostic.
	Related []RelatedInformation
}

// RelatedInformation contains information related to a diagnostic.
// For example, a diagnostic that flags duplicated declarations of a
// variable may include one RelatedInformation per existing
// declaration.
type RelatedInformation struct {
	Pos     token.Pos
	End     token.Pos // optional
	Message string
}

// A SuggestedFix is a code change associated with a Diagnostic that a
// user can choose to apply to their code. Usually the SuggestedFix is
// meant to fix the issue flagged by the diagnostic.
//
// The TextEdits must not overlap, nor contain edits for other packages.
type SuggestedFix struct {
	// A description for this suggested fix to be shown to a user deciding
	// whether to accept it.
	Message   string
	TextEdits []TextEdit
}

// A TextEdit represents the replacement of the code between Pos and End with the new text.
// Each TextEdit should apply to a single file. End should not be earlier in the file than Pos.
type TextEdit struct {
	// For a pure insertion, End can either be set to Pos or token.NoPos.
	Pos     token.Pos
	End     token.Pos
	NewText []byte
}
