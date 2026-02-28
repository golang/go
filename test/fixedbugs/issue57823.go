// run

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"runtime"
	"unsafe"
)

//go:noinline
func g(x *byte) *byte { return x }

func main() {
	slice()
	str("AAAAAAAA", "BBBBBBBBB")
}

func wait(done <-chan struct{}) bool {
	for i := 0; i < 10; i++ {
		runtime.GC()
		select {
		case <-done:
			return true
		default:
		}
	}
	return false
}

func slice() {
	s := make([]byte, 100)
	s[0] = 1
	one := unsafe.SliceData(s)

	done := make(chan struct{})
	runtime.SetFinalizer(one, func(*byte) { close(done) })

	h := g(one)

	if wait(done) {
		panic("GC'd early")
	}

	if *h != 1 {
		panic("lost one")
	}

	if !wait(done) {
		panic("never GC'd")
	}
}

var strDone = make(chan struct{})

//go:noinline
func str(x, y string) {
	s := x + y // put in temporary on stack
	p := unsafe.StringData(s)
	runtime.SetFinalizer(p, func(*byte) { close(strDone) })

	if wait(strDone) {
		panic("GC'd early")
	}

	if *p != 'A' {
		panic("lost p")
	}

	if !wait(strDone) {
		panic("never GC'd")
	}
}
