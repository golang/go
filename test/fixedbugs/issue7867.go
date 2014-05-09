// runoutput

// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 7867.

package main

import "fmt"

const tpl = `
func Test%d(t %s) {
	_ = t
	_ = t
}
`

func main() {
	fmt.Println("package main")
	types := []string{
		// These types always passed
		"bool", "int", "rune",
		"*int", "uintptr",
		"float32", "float64",
		"chan struct{}",
		"map[string]struct{}",
		"func()", "func(string)error",

		// These types caused compilation failures
		"complex64", "complex128",
		"struct{}", "struct{n int}", "struct{e error}", "struct{m map[string]string}",
		"string",
		"[4]byte",
		"[]byte",
		"interface{}", "error",
	}
	for i, typ := range types {
		fmt.Printf(tpl, i, typ)
	}
	fmt.Println("func main() {}")
}
