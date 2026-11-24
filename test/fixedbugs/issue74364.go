// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"runtime"
	"testing"
	"unsafe"
)

var input = 7
var sink any
var unsafeSink unsafe.Pointer
var aliasSink **int

//go:noinline
func choose(p *int, direct bool) any {
	if direct {
		return p
	}
	return &p
}

//go:noinline
func chooseUnsafe(p *int, direct bool) unsafe.Pointer {
	if direct {
		return unsafe.Pointer(p)
	}
	return unsafe.Pointer(&p)
}

//go:noinline
func withAlias(p *int, direct bool) any {
	aliasSink = &p
	if direct {
		return p
	}
	return &p
}

//go:noinline
func withDefer(p *int, direct bool) any {
	defer func() { p = &input }()
	if direct {
		return p
	}
	return &p
}

func main() {
	if n := testing.AllocsPerRun(100, func() { sink = choose(&input, true) }); n != 0 {
		panic("allocation on direct return")
	}
	if n := testing.AllocsPerRun(100, func() { sink = choose(&input, false) }); n != 1 {
		panic("incorrect number of allocations on indirect return")
	}
	if n := testing.AllocsPerRun(100, func() { unsafeSink = chooseUnsafe(&input, true) }); n != 0 {
		panic("allocation on direct unsafe.Pointer return")
	}
	// The copied variable and its pointee both need to remain live.
	p := new(int)
	*p = 42
	box := choose(p, false).(**int)
	runtime.GC()
	if **box != 42 {
		panic("lost pointee")
	}
	if got := withAlias(&input, false).(**int); got != aliasSink {
		panic("lost address identity")
	}
	if got := *withDefer(new(int), false).(**int); got != &input {
		panic("lost deferred update")
	}
}
