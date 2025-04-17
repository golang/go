// run

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "runtime"

func main() {
	if wait() {
		panic("GC'd early")
	}
	m = nil
	if !wait() {
		panic("never GC'd")
	}
}

var m = New[int]().M

func New[X any]() *T[X] {
	p := new(T[X])
	runtime.SetFinalizer(p, func(*T[X]) { close(done) })
	return p
}

type T[X any] [4]int // N.B., [4]int avoids runtime's tiny object allocator

func (*T[X]) M() {}

var done = make(chan int)

func wait() bool {
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
