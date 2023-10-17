// skip

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test the -X facility of the gc linker (6l etc.).
// This test is run by linkx_run.go.

package main

import "fmt"

var tbd string
var overwrite string = "dibs"

var tbdcopy = tbd
var overwritecopy = overwrite
var arraycopy = [2]string{tbd, overwrite}

var b bool
var x int

func main() {
	fmt.Println(tbd)
	fmt.Println(tbdcopy)
	fmt.Println(arraycopy[0])

	fmt.Println(overwrite)
	fmt.Println(overwritecopy)
	fmt.Println(arraycopy[1])

	// Check non-string symbols are not overwritten.
	// This also make them used.
	if b || x != 0 {
		panic("b or x overwritten")
	}
}
