// errorcheck

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that error messages say what the source file says
// (uint8 vs byte, int32 vs. rune).
// Does not compile.

package main

import (
	"fmt"
	"unicode/utf8"
)

func f(byte)  {}
func g(uint8) {}

func main() {
	var x float64
	f(x) // ERROR "byte"
	g(x) // ERROR "uint8"

	// Test across imports.

	var ff fmt.Formatter
	var fs fmt.State
	ff.Format(fs, x) // ERROR "rune"

	utf8.RuneStart(x) // ERROR "byte"
}
