// +build experimental

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

	// TODO(matloob): Should multiple SuggestedFixes be allowed for a diagnostic?
	SuggestedFixes []SuggestedFix // optional
}

// A SuggestedFix is a code change associated with a Diagnostic that a user can choose
// to apply to their code. Usually the SuggestedFix is meant to fix the issue flagged
// by the diagnostic.
type SuggestedFix struct {
	// A description for this suggested fix to be shown to a user deciding
	// whether to accept it.
	Message   string
	TextEdits []TextEdit
}

// A TextEdit represents the replacement of the code between Pos and End with the new text.
type TextEdit struct {
	// For a pure insertion, End can either be set to Pos or token.NoPos.
	Pos     token.Pos
	End     token.Pos
	NewText []byte
}
