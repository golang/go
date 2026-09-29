// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// A recover called in the body of a range-over-func loop takes effect,
// just as it would anywhere else in the function containing the loop.
// The body runs in a frame of its own, called from the iterator, but in
// the source it is part of that function, and the frames the compiler
// uses to run the loop must not change what recover means.

package main

import "fmt"

func iter(yield func(int) bool) { yield(0) }

var fails int

func check(name string, got, want any) {
	if got != want {
		fmt.Printf("%s = %v, want %v\n", name, got, want)
		fails++
	}
}

// recovered reports what a call of f, deferred by a panicking function,
// recovered, or nil if the panic went unrecovered.
func recovered(f func(*any)) (got any) {
	defer func() {
		if recover() != nil {
			got = nil // the panic escaped f
		}
	}()
	var r any
	func() {
		defer f(&r)
		panic("boom")
	}()
	return r
}

// A recover in a loop body recovers, and returns the panic value.
func body(r *any) {
	for range iter {
		*r = recover()
	}
}

// So does one in the body of a loop nested in another loop's body.
func nested(r *any) {
	for range iter {
		for range iter {
			*r = recover()
		}
	}
}

// A recover in a func literal in the body does not: it is two calls away
// from the deferred function, not one, exactly as outside a loop.
func inClosure(r *any) {
	for range iter {
		func() { *r = recover() }()
	}
}

// A recover in a loop in a function that was not the one deferred does
// not recover either.
func inCallee(r *any) {
	helper(r)
}

func helper(r *any) {
	for range iter {
		*r = recover()
	}
}

// Ranging over something other than a function must behave the same way.
func overInt(r *any) {
	for range 1 {
		*r = recover()
	}
}

func overSlice(r *any) {
	for range []int{0} {
		*r = recover()
	}
}

func overChan(r *any) {
	ch := make(chan int, 1)
	ch <- 0
	close(ch)
	for range ch {
		*r = recover()
	}
}

// A defer in the body defers to the containing function, and a recover in
// the body still works alongside it.
func withDefer(r *any) {
	for range iter {
		defer func() { *r = "deferred ran" }()
		if recover() == nil {
			*r = "did not recover"
			return
		}
	}
}

// Recovering twice: the second call reports no panic left to recover.
func twice(r *any) {
	for range iter {
		first := recover()
		second := recover()
		*r = fmt.Sprint(first, " ", second)
	}
}

// A whole panic and recover inside the body, before recovering the panic
// the body was deferred for.
func inner(r *any) {
	for range iter {
		func() {
			defer func() { recover() }()
			panic("inner")
		}()
		*r = recover()
	}
}

func main() {
	check("body", recovered(body), "boom")
	check("nested", recovered(nested), "boom")
	check("inClosure", recovered(inClosure), nil)
	check("inCallee", recovered(inCallee), nil)
	check("overInt", recovered(overInt), "boom")
	check("overSlice", recovered(overSlice), "boom")
	check("overChan", recovered(overChan), "boom")
	check("withDefer", recovered(withDefer), "deferred ran")
	check("twice", recovered(twice), "boom <nil>")
	check("inner", recovered(inner), "boom")

	// With nothing panicking, a recover in a loop body returns nil and the
	// loop runs as usual.
	n := 0
	var got any = "not nil"
	for range iter {
		got = recover()
		n++
	}
	check("no panic: recover()", got, nil)
	check("no panic: iterations", n, 1)

	if fails > 0 {
		panic("failed")
	}
}
