// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "runtime"

type T [4]int

//go:noinline
func g(x []*T) ([]*T, []*T) { return x, x }

func main() {
	const Jenny = 8675309
	s := [10]*T{{Jenny}}

	done := make(chan struct{})
	runtime.SetFinalizer(s[0], func(p *T) { close(done) })

	var h, _ interface{} = g(s[:])

	if wait(done) {
		panic("GC'd early")
	}

	if h.([]*T)[0][0] != Jenny {
		panic("lost Jenny's number")
	}

	if !wait(done) {
		panic("never GC'd")
	}
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
