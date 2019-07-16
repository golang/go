// run arg1 arg2

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
