// skip

// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test the -X facility of the gc linker (6l etc.).
// This test is run by linkx_run.go.

package main

var tbd string
var overwrite string = "dibs"

func main() {
	println(tbd)
	println(overwrite)
}
