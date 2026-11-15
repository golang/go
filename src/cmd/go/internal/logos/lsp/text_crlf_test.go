// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"testing"
)

func TestCRLFEndOfLineStaysBeforeTheBreak(t *testing.T) {
	doc := newText([]byte("a\r\nb\n"))
	if got := doc.offset(Position{Line: 0, Character: 2}); got != 1 {
		t.Fatalf("offset of line 0 char 2 = %d, want 1 before the carriage return", got)
	}
	if got := doc.position(1); got != (Position{Line: 0, Character: 1}) {
		t.Fatalf("position of 1 = %v, want {0 1}", got)
	}
	after := doc.replace(doc.offset(Position{Line: 0, Character: 1}), doc.offset(Position{Line: 0, Character: 1}), "x")
	if string(after.data) != "ax\r\nb\n" {
		t.Fatalf("insert at line 0 char 1 = %q, want %q", after.data, "ax\r\nb\n")
	}
}
