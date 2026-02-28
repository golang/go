// run

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that empty init functions are skipped.

package main

import _ "unsafe" // for go:linkname

type initTask struct {
	state uintptr
	ndeps uintptr
	nfns  uintptr
}

//go:linkname main_inittask main..inittask
var main_inittask initTask

func main() {
	if nfns := main_inittask.nfns; nfns != 0 {
		println(nfns)
		panic("unexpected init funcs")
	}
}

func init() {
}

func init() {
	if false {
	}
}

func init() {
	for false {
	}
}
