// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package driverutil

// This file defined output helpers common to all drivers.

import (
	"encoding/json"
	"fmt"
	"go/token"
	"io"
	"log"
	"os"
	"strings"

	"golang.org/x/tools/go/analysis"
)

// TODO(adonovan): don't accept an io.Writer if we don't report errors.
// Either accept a bytes.Buffer (infallible), or return a []byte.

// PrintPlain prints a diagnostic in plain text form.
// If contextLines is nonnegative, it also prints the
// offending line plus this many lines of context.
func PrintPlain(out io.Writer, fset *token.FileSet, contextLines int, diag analysis.Diagnostic) {
	print := func(pos, end token.Pos, message string) {
		posn := fset.Position(pos)
		fmt.Fprintf(out, "%s: %s\n", posn, message)

		// show offending line plus N lines of context.
		if contextLines >= 0 {
			end := fset.Position(end)
			if !end.IsValid() {
				end = posn
			}
			// TODO(adonovan): highlight the portion of the line indicated
			// by pos...end using ASCII art, terminal colors, etc?
			data, _ := os.ReadFile(posn.Filename)
			lines := strings.Split(string(data), "\n")
			for i := posn.Line - contextLines; i <= end.Line+contextLines; i++ {
				if 1 <= i && i <= len(lines) {
					fmt.Fprintf(out, "%d\t%s\n", i, lines[i-1])
				}
			}
		}
	}

	print(diag.Pos, diag.End, diag.Message)
	for _, rel := range diag.Related {
		print(rel.Pos, rel.End, "\t"+rel.Message)
	}
}

// A JSONTree is a mapping from package ID to analysis name to result.
// Each result is either a jsonError or a list of JSONDiagnostic.
type JSONTree map[string]map[string]any

// A TextEdit describes the replacement of a portion of a file.
// Start and End are zero-based half-open indices into the original byte
// sequence of the file, and New is the new text.
type JSONTextEdit struct {
	Filename string `json:"filename"`
	Start    int    `json:"start"`
	End      int    `json:"end"`
	New      string `json:"new"`
}

// A JSONSuggestedFix describes an edit that should be applied as a whole or not
// at all. It might contain multiple TextEdits/text_edits if the SuggestedFix
// consists of multiple non-contiguous edits.
type JSONSuggestedFix struct {
	Message string         `json:"message"`
	Edits   []JSONTextEdit `json:"edits"`
}

// A JSONDiagnostic describes the JSON schema of an analysis.Diagnostic.
//
// TODO(matloob): include End position if present.
type JSONDiagnostic struct {
	Category       string                   `json:"category,omitempty"`
	Posn           string                   `json:"posn"` // e.g. "file.go:line:column"
	Message        string                   `json:"message"`
	SuggestedFixes []JSONSuggestedFix       `json:"suggested_fixes,omitempty"`
	Related        []JSONRelatedInformation `json:"related,omitempty"`
}

// A JSONRelated describes a secondary position and message related to
// a primary diagnostic.
//
// TODO(adonovan): include End position if present.
type JSONRelatedInformation struct {
	Posn    string `json:"posn"` // e.g. "file.go:line:column"
	Message string `json:"message"`
}

// Add adds the result of analysis 'name' on package 'id'.
// The result is either a list of diagnostics or an error.
func (tree JSONTree) Add(fset *token.FileSet, id, name string, diags []analysis.Diagnostic, err error) {
	var v any
	if err != nil {
		type jsonError struct {
			Err string `json:"error"`
		}
		v = jsonError{err.Error()}
	} else if len(diags) > 0 {
		diagnostics := make([]JSONDiagnostic, 0, len(diags))
		for _, f := range diags {
			var fixes []JSONSuggestedFix
			for _, fix := range f.SuggestedFixes {
				var edits []JSONTextEdit
				for _, edit := range fix.TextEdits {
					edits = append(edits, JSONTextEdit{
						Filename: fset.Position(edit.Pos).Filename,
						Start:    fset.Position(edit.Pos).Offset,
						End:      fset.Position(edit.End).Offset,
						New:      string(edit.NewText),
					})
				}
				fixes = append(fixes, JSONSuggestedFix{
					Message: fix.Message,
					Edits:   edits,
				})
			}
			var related []JSONRelatedInformation
			for _, r := range f.Related {
				related = append(related, JSONRelatedInformation{
					Posn:    fset.Position(r.Pos).String(),
					Message: r.Message,
				})
			}
			jdiag := JSONDiagnostic{
				Category:       f.Category,
				Posn:           fset.Position(f.Pos).String(),
				Message:        f.Message,
				SuggestedFixes: fixes,
				Related:        related,
			}
			diagnostics = append(diagnostics, jdiag)
		}
		v = diagnostics
	}
	if v != nil {
		m, ok := tree[id]
		if !ok {
			m = make(map[string]any)
			tree[id] = m
		}
		m[name] = v
	}
}

func (tree JSONTree) Print(out io.Writer) error {
	data, err := json.MarshalIndent(tree, "", "\t")
	if err != nil {
		log.Panicf("internal error: JSON marshaling failed: %v", err)
	}
	_, err = fmt.Fprintf(out, "%s\n", data)
	return err
}
