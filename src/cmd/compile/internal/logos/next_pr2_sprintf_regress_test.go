// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"testing"
)

func TestNextSprintfNilAndFormatterStayCalls(t *testing.T) {
	imp := newTestImporter(t)
	stable := func(t *testing.T, body string) int {
		t.Helper()
		src := "package p\n\nimport \"fmt\"\n\n" + body + "\n"
		edits, _ := scanSnippet(t, imp, src)
		return len(edits)
	}
	t.Run("nilErrorV", func(t *testing.T) {
		if n := stable(t, "func g(err error) string { return fmt.Sprintf(\"%v\", err) }"); n != 0 {
			t.Fatalf("nil-able error %%v: got %d edits, want 0", n)
		}
	})
	t.Run("nilStringerV", func(t *testing.T) {
		if n := stable(t, "func g(s fmt.Stringer) string { return fmt.Sprintf(\"%v\", s) }"); n != 0 {
			t.Fatalf("nil-able Stringer %%v: got %d edits, want 0", n)
		}
	})
	t.Run("errorS", func(t *testing.T) {
		if n := stable(t, "func g(err error) string { return fmt.Sprintf(\"%s\", err) }"); n != 0 {
			t.Fatalf("error %%s: got %d edits, want 0", n)
		}
	})
	t.Run("formatterV", func(t *testing.T) {
		body := "type N int\nfunc (n N) Format(s fmt.State, c rune) { s.Write([]byte(\"custom\")) }\nfunc g(v N) string { return fmt.Sprintf(\"%v\", v) }"
		if n := stable(t, body); n != 0 {
			t.Fatalf("Formatter %%v: got %d edits, want 0", n)
		}
	})
	t.Run("formatterD", func(t *testing.T) {
		body := "type N int\nfunc (n N) Format(s fmt.State, c rune) { s.Write([]byte(\"custom\")) }\nfunc g(v N) string { return fmt.Sprintf(\"%d\", v) }"
		if n := stable(t, body); n != 0 {
			t.Fatalf("Formatter %%d: got %d edits, want 0", n)
		}
	})
	t.Run("basicStringInt", func(t *testing.T) {
		if n := stable(t, "func g(name string, n int) string { return fmt.Sprintf(\"hello %s %d\", name, n) }"); n != 1 {
			t.Fatalf("basic string/int: got %d edits, want 1", n)
		}
	})
	t.Run("boolV", func(t *testing.T) {
		if n := stable(t, "func g(ok bool) string { return fmt.Sprintf(\"ok=%v\", ok) }"); n != 1 {
			t.Fatalf("bool %%v: got %d edits, want 1", n)
		}
	})
}
