// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 26153. The write to ps was incorrectly
// removed by the dead auto elimination pass.

package main

const hello = "hello world"

func main() {
	var s string
	mangle(&s)
	if s != hello {
		panic("write incorrectly elided")
	}
}

//go:noinline
func mangle(ps *string) {
	if ps == nil {
		var s string
		ps = &s
	}
	*ps = hello
}
