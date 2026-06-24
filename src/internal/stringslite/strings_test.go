// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package stringslite_test

import (
	"internal/stringslite"
	"testing"
	"unicode"
	"unicode/utf8"
)

func TestIsSpace(t *testing.T) {
	for r := rune(0); r <= utf8.MaxRune; r++ {
		if stringslite.IsSpace(r) != unicode.IsSpace(r) {
			t.Fatalf("IsSpace(%U) = %v, want %v", r, stringslite.IsSpace(r), unicode.IsSpace(r))
		}
	}
}
