// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hints

type Position struct {
	Line      int `json:"line"`
	Character int `json:"character"`
}

func (p Position) before(q Position) bool {
	return p.Line < q.Line || (p.Line == q.Line && p.Character < q.Character)
}

type Range struct {
	Start Position `json:"start"`
	End   Position `json:"end"`
}

func (r Range) holds(p Position) bool {
	return !p.before(r.Start) && p.before(r.End)
}

func (r Range) touches(p Position) bool {
	return !p.before(r.Start) && !r.End.before(p)
}

func (r Range) intersects(o Range) bool {
	return !o.End.before(r.Start) && !r.End.before(o.Start)
}

const (
	KindType = 1
)

type InlayHint struct {
	Position     Position `json:"position"`
	Label        string   `json:"label"`
	Kind         int      `json:"kind,omitempty"`
	Tooltip      string   `json:"tooltip,omitempty"`
	PaddingLeft  bool     `json:"paddingLeft,omitempty"`
	PaddingRight bool     `json:"paddingRight,omitempty"`
}

type MarkupContent struct {
	Kind  string `json:"kind"`
	Value string `json:"value"`
}

type Hover struct {
	Contents MarkupContent `json:"contents"`
	Range    *Range        `json:"range,omitempty"`
}

type TextEdit struct {
	Range   Range  `json:"range"`
	NewText string `json:"newText"`
}

type WorkspaceEdit struct {
	Changes map[string][]TextEdit `json:"changes"`
}

type LSPDiagnostic struct {
	Range    Range  `json:"range"`
	Severity int    `json:"severity,omitempty"`
	Code     string `json:"code,omitempty"`
	Source   string `json:"source,omitempty"`
	Message  string `json:"message"`
}

type CodeAction struct {
	Title       string          `json:"title"`
	Kind        string          `json:"kind,omitempty"`
	Diagnostics []LSPDiagnostic `json:"diagnostics,omitempty"`
	Edit        *WorkspaceEdit  `json:"edit,omitempty"`
}
