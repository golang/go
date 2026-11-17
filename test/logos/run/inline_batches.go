// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// Calls through returned closures appear only after earlier calls inline.
// The two uses in twice must be discovered together.
func twice(f func(int) int, x int) int {
	return f(x) + f(x+1)
}

func stages(seed int) func(int) int {
	return func(x int) int {
		f := func(y int) int { return seed + 2*x + y }
		return twice(f, 3)
	}
}

type factory int

func (f factory) Closure() func(int) int { return stages(int(f)) }

type generic[T ~int] struct{ seed T }

func (g generic[T]) Apply(x T) T { return g.seed + 2*x }

func event(trace []int, n int) int {
	trace[0] = trace[0]*10 + n
	return n
}

func produce(trace []int, n int) func(int) int {
	event(trace, n)
	return func(x int) int { return event(trace, x) }
}

func reassigned(seed int) int {
	f := func(x int) int { return seed + x }
	a := f(1)
	f = func(x int) int { return seed - x }
	return 100*a + f(2)
}

func capturedAssignment(seed int) int {
	f := func() int { return seed + 1 }
	change := func() { f = func() int { return seed + 2 } }
	a := f()
	change()
	return 100*a + f()
}

func even(n int) bool {
	if n == 0 {
		return true
	}
	return odd(n - 1)
}

func odd(n int) bool {
	if n == 0 {
		return false
	}
	return even(n - 1)
}

func nilCall(trace []int) (caught bool) {
	defer func() { caught = recover() != nil }()
	var f func(int) int
	f(event(trace, 4))
	return
}

func main() {
	if stages(7)(5) != 41 {
		panic("returned closure batches")
	}
	// The compiler generates and inlines method wrappers after the package
	// fixed point. Their calls must use the same discovery rules.
	method := factory(7).Closure
	if method()(5) != 41 {
		panic("method wrapper batches")
	}
	g := generic[int]{seed: 7}
	apply := g.Apply
	if apply(5) != 17 {
		panic("generic method wrapper")
	}
	trace := []int{0}
	f := produce(trace, 1)
	if twice(f, 2) != 5 || trace[0] != 123 {
		panic("multiple closure calls")
	}
	trace[0] = 0
	if produce(trace, 1)(event(trace, 2)) != 2 || trace[0] != 122 {
		panic("callee and argument order")
	}
	if reassigned(10) != 1108 || capturedAssignment(10) != 1112 {
		panic("closure reassignment")
	}
	if !even(10) || !odd(9) || even(9) || odd(10) {
		panic("recursive component")
	}
	if !nilCall(trace) || trace[0] != 1224 {
		panic("argument effect before nil call")
	}
}
