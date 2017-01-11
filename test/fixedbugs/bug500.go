// run

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Gccgo generated incorrect GC info when a global variable was
// initialized to a slice of a value containing pointers.  The initial
// backing array for the slice was allocated in the .data section,
// which is fine, but the backing array was not registered as a GC
// root.

package main

import (
	"runtime"
)

type s struct {
	str string
}

var a = []struct {
	str string
}{
	{""},
}

var b = "b"
var c = "c"

func init() {
	a[0].str = b + c
}

func main() {
	runtime.GC()
	if a[0].str != b + c {
		panic(a[0].str)
	}
}
