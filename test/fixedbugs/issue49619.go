// build -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This testcase caused a linker crash in DWARF generation.

package main

//go:noinline
func f() any {
	var a []any
	return a[0]
}

func main() {
	f()
}
