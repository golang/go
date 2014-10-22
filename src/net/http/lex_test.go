// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"testing"
)

func isChar(c rune) bool { return c <= 127 }

func isCtl(c rune) bool { return c <= 31 || c == 127 }

func isSeparator(c rune) bool {
	switch c {
	case '(', ')', '<', '>', '@', ',', ';', ':', '\\', '"', '/', '[', ']', '?', '=', '{', '}', ' ', '\t':
		return true
	}
	return false
}

func TestIsToken(t *testing.T) {
	for i := 0; i <= 130; i++ {
		r := rune(i)
		expected := isChar(r) && !isCtl(r) && !isSeparator(r)
		if isToken(r) != expected {
			t.Errorf("isToken(0x%x) = %v", r, !expected)
		}
	}
}
