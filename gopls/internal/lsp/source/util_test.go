// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"bytes"
	"go/scanner"
	"go/token"
	"testing"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/span"
)

func TestMappedRangeAdjustment(t *testing.T) {
	// Test that mapped range adjusts positions in compiled files to positions in
	// the corresponding edited file.

	compiled := []byte(`// Generated. DO NOT EDIT.

package p

//line edited.go:3:1
const a𐐀b = 42`)
	edited := []byte(`package p

const a𐐀b = 42`)

	fset := token.NewFileSet()
	cf := scanFile(fset, "compiled.go", compiled)
	ef := scanFile(fset, "edited.go", edited)
	eURI := span.URIFromPath(ef.Name())

	mapper := &protocol.ColumnMapper{
		URI:     eURI,
		TokFile: ef,
		Content: edited,
	}

	start := cf.Pos(bytes.Index(compiled, []byte("a𐐀b")))
	end := start + token.Pos(len("a𐐀b"))
	mr := NewMappedRange(cf, mapper, start, end)
	gotRange, err := mr.Range()
	if err != nil {
		t.Fatal(err)
	}
	wantRange := protocol.Range{
		Start: protocol.Position{Line: 2, Character: 6},
		End:   protocol.Position{Line: 2, Character: 10},
	}
	if gotRange != wantRange {
		t.Errorf("NewMappedRange(...).Range(): got %v, want %v", gotRange, wantRange)
	}

	// Verify that the mapped span is also in the edited file.
	gotSpan, err := mr.Span()
	if err != nil {
		t.Fatal(err)
	}
	if gotURI := gotSpan.URI(); gotURI != eURI {
		t.Errorf("mr.Span().URI() = %v, want %v", gotURI, eURI)
	}
	wantOffset := bytes.Index(edited, []byte("a𐐀b"))
	if gotOffset := gotSpan.Start().Offset(); gotOffset != wantOffset {
		t.Errorf("mr.Span().Start().Offset() = %d, want %d", gotOffset, wantOffset)
	}
}

// scanFile scans the a file into fset, in order to honor line directives.
func scanFile(fset *token.FileSet, name string, content []byte) *token.File {
	f := fset.AddFile(name, -1, len(content))
	var s scanner.Scanner
	s.Init(f, content, nil, scanner.ScanComments)
	for {
		_, tok, _ := s.Scan()
		if tok == token.EOF {
			break
		}
	}
	return f
}
