// $G $D/$F.go && $L -X main.tbd hello -X main.overwrite trumped -X main.nosuchsymbol neverseen $F.$A && ./$A.out

// NOTE: This test is not run by 'run.go' and so not run by all.bash.
// To run this test you must use the ./run shell script.

// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test the -X facility of the gc linker (6l etc.).

package main

var tbd string
var overwrite string = "dibs"

func main() {
	if tbd != "hello" {
		println("BUG: test/linkx tbd", len(tbd), tbd)
	}
	if overwrite != "trumped" {
		println("BUG: test/linkx overwrite", len(overwrite), overwrite)
	}
}
