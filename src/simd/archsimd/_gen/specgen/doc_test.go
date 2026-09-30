// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package specgen

import (
	"testing"
)

func TestParseSpecDoc(t *testing.T) {
	doc := `Add adds elements of
x and y.

	z[i] = x[i] + y[i]
`
	var root contextRoot
	ctx := context{root: &root}
	paras := parseSpecDoc(ctx, doc)
	if err := root.gatherErrors(); err != nil {
		t.Fatalf("unexpected error parsing spec doc: %v", err)
	}

	expected := []struct {
		kind ParagraphKind
		text string
	}{
		{SpecOwned, "Add adds elements of\nx and y."},
		{SpecOwned, "\tz[i] = x[i] + y[i]"},
	}
	if len(paras) != len(expected) {
		t.Fatalf("expected %d paragraphs, got %d", len(expected), len(paras))
	}
	for i, want := range expected {
		if paras[i].Kind != want.kind {
			t.Errorf("para %d: got kind %v, want %v", i, paras[i].Kind, want.kind)
		}
		if paras[i].Text != want.text {
			t.Errorf("para %d: got text %q, want %q", i, paras[i].Text, want.text)
		}
	}

	// Test FormatComment round-trip
	formatted, err := FormatComment(paras)
	if err != nil {
		t.Fatalf("unexpected error formatting comment: %v", err)
	}
	wantFormatted := `// Add adds elements of
// x and y.
//
//	z[i] = x[i] + y[i]
`
	if formatted != wantFormatted {
		t.Errorf("FormatComment mismatch:\ngot:\n%s\nwant:\n%s", formatted, wantFormatted)
	}
}

func TestParseSpecDocRejectsNotes(t *testing.T) {
	doc := `Add adds elements.

Asm: VPADDD`

	var root contextRoot
	ctx := context{root: &root}
	_ = parseSpecDoc(ctx, doc)
	if err := root.gatherErrors(); err == nil {
		t.Fatalf("expected error for doc containing implementation note, got nil")
	}
}

func TestFormatCommentOrdering(t *testing.T) {
	tests := []struct {
		name    string
		paras   []Paragraph
		wantErr bool
	}{
		{
			name: "valid order",
			paras: []Paragraph{
				{Kind: SpecOwned, Text: "Spec\n"},
				{Kind: ImplementationNote, Text: "Asm: VFOO\n"},
				{Kind: DirectiveComment, Text: "go:noescape\n"},
			},
			wantErr: false,
		},
		{
			name: "spec owned after note",
			paras: []Paragraph{
				{Kind: ImplementationNote, Text: "Asm: VFOO\n"},
				{Kind: SpecOwned, Text: "Spec\n"},
			},
			wantErr: true,
		},
		{
			name: "spec owned after directive",
			paras: []Paragraph{
				{Kind: DirectiveComment, Text: "go:noescape\n"},
				{Kind: SpecOwned, Text: "Spec\n"},
			},
			wantErr: true,
		},
		{
			name: "note after directive",
			paras: []Paragraph{
				{Kind: DirectiveComment, Text: "go:noescape\n"},
				{Kind: ImplementationNote, Text: "Asm: VFOO\n"},
			},
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := FormatComment(tc.paras)
			if (err != nil) != tc.wantErr {
				t.Errorf("FormatComment() error = %v, wantErr %v", err, tc.wantErr)
			}
		})
	}
}
