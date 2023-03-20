// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/span"
)

func TestParseErrorMessage(t *testing.T) {
	tests := []struct {
		name             string
		in               string
		expectedFileName string
		expectedLine     int
		expectedColumn   int
	}{
		{
			name:             "from go list output",
			in:               "\nattributes.go:13:1: expected 'package', found 'type'",
			expectedFileName: "attributes.go",
			expectedLine:     13,
			expectedColumn:   1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			spn := parseGoListError(tt.in, ".")
			fn := spn.URI().Filename()

			if !strings.HasSuffix(fn, tt.expectedFileName) {
				t.Errorf("expected filename with suffix %v but got %v", tt.expectedFileName, fn)
			}

			if !spn.HasPosition() {
				t.Fatalf("expected span to have position")
			}

			pos := spn.Start()
			if pos.Line() != tt.expectedLine {
				t.Errorf("expected line %v but got %v", tt.expectedLine, pos.Line())
			}

			if pos.Column() != tt.expectedColumn {
				t.Errorf("expected line %v but got %v", tt.expectedLine, pos.Line())
			}
		})
	}
}

func TestDiagnosticEncoding(t *testing.T) {
	diags := []*source.Diagnostic{
		{}, // empty
		{
			URI: "file///foo",
			Range: protocol.Range{
				Start: protocol.Position{Line: 4, Character: 2},
				End:   protocol.Position{Line: 6, Character: 7},
			},
			Severity: protocol.SeverityWarning,
			Code:     "red",
			CodeHref: "https://go.dev",
			Source:   "test",
			Message:  "something bad happened",
			Tags:     []protocol.DiagnosticTag{81},
			Related: []protocol.DiagnosticRelatedInformation{
				{
					Location: protocol.Location{
						URI: "file:///other",
						Range: protocol.Range{
							Start: protocol.Position{Line: 3, Character: 6},
							End:   protocol.Position{Line: 4, Character: 9},
						},
					},
					Message: "psst, over here",
				},
			},

			// Fields below are used internally to generate quick fixes. They aren't
			// part of the LSP spec and don't leave the server.
			SuggestedFixes: []source.SuggestedFix{
				{
					Title: "fix it!",
					Edits: map[span.URI][]protocol.TextEdit{
						"file:///foo": {{
							Range: protocol.Range{
								Start: protocol.Position{Line: 4, Character: 2},
								End:   protocol.Position{Line: 6, Character: 7},
							},
							NewText: "abc",
						}},
						"file:///other": {{
							Range: protocol.Range{
								Start: protocol.Position{Line: 4, Character: 2},
								End:   protocol.Position{Line: 6, Character: 7},
							},
							NewText: "!@#!",
						}},
					},
					Command: &protocol.Command{
						Title:     "run a command",
						Command:   "gopls.fix",
						Arguments: []json.RawMessage{json.RawMessage(`{"a":1}`)},
					},
					ActionKind: protocol.QuickFix,
				},
			},
		},
		{
			URI: "file//bar",
			// other fields tested above
		},
	}

	data := encodeDiagnostics(diags)
	diags2 := decodeDiagnostics(data)

	if diff := cmp.Diff(diags, diags2); diff != "" {
		t.Errorf("decoded diagnostics do not match (-original +decoded):\n%s", diff)
	}
}
