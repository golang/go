// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsppos_test

import (
	"go/token"
	"testing"

	. "golang.org/x/tools/gopls/internal/lsp/lsppos"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
)

func makeTokenMapper(content []byte) (*TokenMapper, *token.File) {
	file := token.NewFileSet().AddFile("p.go", -1, len(content))
	file.SetLinesForContent(content)
	return NewTokenMapper(content, file), file
}

func TestInvalidPosition(t *testing.T) {
	content := []byte("a𐐀b\r\nx\ny")
	m, _ := makeTokenMapper(content)

	for _, pos := range []token.Pos{-1, 100} {
		posn, ok := m.Position(pos)
		if ok {
			t.Errorf("Position(%d) = %v, want error", pos, posn)
		}
	}
}

func TestTokenPosition(t *testing.T) {
	for _, test := range tests {
		m, f := makeTokenMapper([]byte(test.content))
		pos := token.Pos(f.Base() + test.offset())
		got, ok := m.Position(pos)
		if !ok {
			t.Error("invalid position for", test.substrOrOffset)
			continue
		}
		want := protocol.Position{Line: uint32(test.wantLine), Character: uint32(test.wantChar)}
		if got != want {
			t.Errorf("Position(%d) = %v, want %v", pos, got, want)
		}
		gotRange, err := m.Range(token.Pos(f.Base()), pos)
		if err != nil {
			t.Fatal(err)
		}
		wantRange := protocol.Range{
			End: want,
		}
		if gotRange != wantRange {
			t.Errorf("Range(%d) = %v, want %v", pos, got, want)
		}
	}
}
