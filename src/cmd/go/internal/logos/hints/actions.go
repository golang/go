// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hints

import (
	"encoding/json"
	"fmt"
)

const DiagnosticVersion = "1.0"

type Span struct {
	StartLine int `json:"start_line"`
	StartCol  int `json:"start_col"`
	EndLine   int `json:"end_line"`
	EndCol    int `json:"end_col"`
}

type Fix struct {
	Description      string `json:"description"`
	Replacement      string `json:"replacement"`
	InsertBeforeLine int    `json:"insert_before_line"`
}

type Diagnostic struct {
	SchemaVersion     string `json:"schema_version"`
	Severity          string `json:"severity"`
	RuleID            string `json:"rule_id"`
	File              string `json:"file"`
	Span              Span   `json:"span"`
	Message           string `json:"message"`
	AutofixCandidates []Fix  `json:"autofix_candidates"`
}

func DecodeDiagnostic(line []byte) (Diagnostic, error) {
	var d Diagnostic
	if err := json.Unmarshal(line, &d); err != nil {
		return d, fmt.Errorf("hints: %w", err)
	}
	if d.SchemaVersion != DiagnosticVersion {
		return d, fmt.Errorf("hints: unsupported diagnostic schema version %q (want %q)", d.SchemaVersion, DiagnosticVersion)
	}
	return d, nil
}

func CodeActions(diags []Diagnostic, uri string, src []byte, rng Range) []CodeAction {
	idx := newLineIndex(src)
	var actions []CodeAction
	for _, d := range diags {
		span, ok := idx.spanRange(d.Span)
		if !ok || !span.intersects(rng) {
			continue
		}
		for _, f := range d.AutofixCandidates {
			actions = append(actions, fixAction(d, f, span, uri))
		}
	}
	return actions
}

func (x lineIndex) spanRange(s Span) (Range, bool) {
	start, ok1 := x.position(s.StartLine, s.StartCol)
	end, ok2 := x.position(s.EndLine, s.EndCol)
	return Range{Start: start, End: end}, ok1 && ok2
}

func fixAction(d Diagnostic, f Fix, span Range, uri string) CodeAction {
	edit := TextEdit{Range: span, NewText: f.Replacement}
	if f.InsertBeforeLine > 0 {
		at := Position{Line: f.InsertBeforeLine - 1}
		edit.Range = Range{Start: at, End: at}
	}
	return CodeAction{
		Title:       f.Description,
		Kind:        "quickfix",
		Diagnostics: []LSPDiagnostic{lspDiagnostic(d, span)},
		Edit:        &WorkspaceEdit{Changes: map[string][]TextEdit{uri: {edit}}},
	}
}

func lspDiagnostic(d Diagnostic, span Range) LSPDiagnostic {
	severity := 1
	if d.Severity == "warn" {
		severity = 2
	}
	return LSPDiagnostic{Range: span, Severity: severity, Code: d.RuleID, Source: "logos", Message: d.Message}
}
