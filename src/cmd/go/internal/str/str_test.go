// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package str

import (
	"testing"
)

var foldDupTests = []struct {
	list   []string
	f1, f2 string
}{
	{StringList("math/rand", "math/big"), "", ""},
	{StringList("math", "strings"), "", ""},
	{StringList("strings"), "", ""},
	{StringList("strings", "strings"), "strings", "strings"},
	{StringList("Rand", "rand", "math", "math/rand", "math/Rand"), "Rand", "rand"},
}

func TestFoldDup(t *testing.T) {
	for _, tt := range foldDupTests {
		f1, f2 := FoldDup(tt.list)
		if f1 != tt.f1 || f2 != tt.f2 {
			t.Errorf("foldDup(%q) = %q, %q, want %q, %q", tt.list, f1, f2, tt.f1, tt.f2)
		}
	}
}
