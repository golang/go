// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

const p0exp = "foo"
const p1exp = 10101
const p2exp = 3030303
const p3exp = 505050505
const p4exp = 70707070707

//go:noinline
//go:registerparams
func callee(p0 string, p1 uint64, p2 uint64, p3 uint64, p4 uint64) {
	if p0 != p0exp {
		panic("bad p0")
	}
	if p1 != p1exp {
		panic("bad p1")
	}
	if p2 != p2exp {
		panic("bad p2")
	}
	if p3 != p3exp {
		panic("bad p3")
	}
	if p4 != p4exp {
		panic("bad p4")
	}
	defer func(p0 string, p2 uint64) {
		if p0 != p0exp {
			panic("defer bad p0")
		}
		if p1 != p1exp {
			panic("defer bad p1")
		}
		if p2 != p2exp {
			panic("defer bad p2")
		}
	}(p0, p2)
}

func main() {
	callee(p0exp, p1exp, p2exp, p3exp, p4exp)
}
