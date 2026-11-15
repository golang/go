// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import "testing"

func TestDirectiveArg(t *testing.T) {
	const marker = "//logos:rodata"
	tests := []struct {
		text string
		arg  string
		ok   bool
	}{
		{"//logos:rodata", "", true},
		{"//logos:rodata ", "", true},
		{"//logos:rodata\t", "", true},
		{"//logos:rodata \t \r", "", true},
		{"//logos:rodata x", "x", true},
		{"//logos:rodata\tx y ", "x y", true},
		{"//logos:rodata // note", "// note", true},
		{"//logos:rodatax", "", false},
		{"//logos:rodata_old", "", false},
		{"//logos:rodata:", "", false},
		{"//logos:rodata(", "", false},
		{"//logos:rodata1", "", false},
		{"//logos:rodata//x", "", false},
		{"//logos:rodat", "", false},
		{"// logos:rodata", "", false},
		{" //logos:rodata", "", false},
		{"//LOGOS:rodata", "", false},
		{"", "", false},
	}
	for _, tt := range tests {
		arg, ok := DirectiveArg(tt.text, marker)
		if arg != tt.arg || ok != tt.ok {
			t.Errorf("DirectiveArg(%q, %q) = %q, %v; want %q, %v", tt.text, marker, arg, ok, tt.arg, tt.ok)
		}
	}
}
