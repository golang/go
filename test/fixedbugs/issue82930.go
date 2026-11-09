// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "runtime"

// growStack uses about n KiB of stack.
//
//go:noinline
func growStack(n int) {
	var buf [1024]byte
	if n > 0 {
		growStack(n - 1)
	}
	runtime.KeepAlive(&buf)
}

// The iterators below are not inlined, so the compiler cannot see what
// they do with the loop body, and the body escapes to the heap.

// growSeq grows the stack before calling yield.
//
//go:noinline
func growSeq() func(func(int) bool) {
	return func(yield func(int) bool) {
		growStack(256)
		yield(0)
	}
}

// shrinkSeq runs a GC before calling yield. Called on a goroutine that
// has a large stack but uses little of it, the GC shrinks the stack.
//
//go:noinline
func shrinkSeq() func(func(int) bool) {
	return func(yield func(int) bool) {
		runtime.GC()
		yield(0)
	}
}

func f(seq func() func(func(int) bool)) (r any) {
	defer func() {
		if recover() != nil {
			r = nil // the panic escaped the loop body
		}
	}()
	func() {
		defer func() {
			for range seq() {
				r = recover()
			}
		}()
		panic("boom")
	}()
	return r
}

func main() {
	// Run each case on a new goroutine, whose stack starts small, so that
	// growStack really grows it.
	ch := make(chan any)
	go func() { ch <- f(growSeq) }()
	if r := <-ch; r != "boom" {
		panic("grow: recover in the loop body did not recover the panic")
	}

	// Grow the stack first, so that the GC in shrinkSeq finds it mostly
	// unused and shrinks it.
	go func() {
		growStack(256)
		ch <- f(shrinkSeq)
	}()
	if r := <-ch; r != "boom" {
		panic("shrink: recover in the loop body did not recover the panic")
	}
}
