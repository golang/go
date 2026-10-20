// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hints

// This file holds the protocol types the renderers produce and read. Only
// the fields the three features use are present; every position counts a
// 0-based line and a 0-based UTF-16 character.

// Position is a protocol position.
type Position struct {
	Line      int `json:"line"`
	Character int `json:"character"`
}

// before reports whether p comes strictly before q.
func (p Position) before(q Position) bool {
	return p.Line < q.Line || (p.Line == q.Line && p.Character < q.Character)
}

// Range is a protocol range; End is exclusive.
type Range struct {
	Start Position `json:"start"`
	End   Position `json:"end"`
}

// holds reports whether p lies in the half-open range.
func (r Range) holds(p Position) bool {
	return !p.before(r.Start) && p.before(r.End)
}

// touches reports whether p lies in the range or at its end.
func (r Range) touches(p Position) bool {
	return !p.before(r.Start) && !r.End.before(p)
}

// intersects reports whether the two ranges share a position, an end point
// included.
func (r Range) intersects(o Range) bool {
	return !o.End.before(r.Start) && !r.End.before(o.Start)
}

// Kinds of InlayHint.
const (
	KindType = 1
)

// InlayHint is a protocol inlay hint.
type InlayHint struct {
	Position     Position `json:"position"`
	Label        string   `json:"label"`
	Kind         int      `json:"kind,omitempty"`
	Tooltip      string   `json:"tooltip,omitempty"`
	PaddingLeft  bool     `json:"paddingLeft,omitempty"`
	PaddingRight bool     `json:"paddingRight,omitempty"`
}

// MarkupContent is protocol markup text.
type MarkupContent struct {
	Kind  string `json:"kind"`
	Value string `json:"value"`
}

// Hover is a protocol hover.
type Hover struct {
	Contents MarkupContent `json:"contents"`
	Range    *Range        `json:"range,omitempty"`
}

// TextEdit is a protocol text edit.
type TextEdit struct {
	Range   Range  `json:"range"`
	NewText string `json:"newText"`
}

// WorkspaceEdit is a protocol workspace edit, keyed by document URI.
type WorkspaceEdit struct {
	Changes map[string][]TextEdit `json:"changes"`
}

// LSPDiagnostic is a protocol diagnostic.
type LSPDiagnostic struct {
	Range    Range  `json:"range"`
	Severity int    `json:"severity,omitempty"`
	Code     string `json:"code,omitempty"`
	Source   string `json:"source,omitempty"`
	Message  string `json:"message"`
}

// CodeAction is a protocol code action.
type CodeAction struct {
	Title       string          `json:"title"`
	Kind        string          `json:"kind,omitempty"`
	Diagnostics []LSPDiagnostic `json:"diagnostics,omitempty"`
	Edit        *WorkspaceEdit  `json:"edit,omitempty"`
}
