// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "testing"

var foldDupTests = []struct {
	list   []string
	f1, f2 string
}{
	{stringList("math/rand", "math/big"), "", ""},
	{stringList("math", "strings"), "", ""},
	{stringList("strings"), "", ""},
	{stringList("strings", "strings"), "strings", "strings"},
	{stringList("Rand", "rand", "math", "math/rand", "math/Rand"), "Rand", "rand"},
}

func TestFoldDup(t *testing.T) {
	for _, tt := range foldDupTests {
		f1, f2 := foldDup(tt.list)
		if f1 != tt.f1 || f2 != tt.f2 {
			t.Errorf("foldDup(%q) = %q, %q, want %q, %q", tt.list, f1, f2, tt.f1, tt.f2)
		}
	}
}
