// errorcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// We have a limit of 1GB for stack frames.
// Make sure we include the callee args section.

package main

func f() { // GC_ERROR "stack frame too large"
	var x [800e6]byte
	g(x)
	return
}

//go:noinline
func g([800e6]byte) {}
