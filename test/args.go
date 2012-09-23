// $G $F.go && $L $F.$A && ./$A.out arg1 arg2

// NOTE: This test is not run by 'run.go' and so not run by all.bash.
// To run this test you must use the ./run shell script.

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test os.Args.

package main

import "os"

func main() {
	if len(os.Args) != 3 {
		panic("argc")
	}
	if os.Args[1] != "arg1" {
		panic("arg1")
	}
	if os.Args[2] != "arg2" {
		panic("arg2")
	}
}
