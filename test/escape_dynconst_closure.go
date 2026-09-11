// errorcheck -0 -m -d=closure
//go:build !goexperiment.newinliner

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that closures do not capture variables that hold a constant, and
// stop being closures if they capture nothing else.

package main

var sink func()

// The closure captures s only, which holds a constant.
func constOnly() {
	s := ""
	f := func() { println(s) } // ERROR "can inline constOnly.func1" "func literal does not escape" "closure converted to global"
	g(f)
}

// Same, but the closure escapes, so its closure record would be heap
// allocated if it still captured anything.
func constOnlyEscapes() {
	i := 42
	sink = func() { println(i) } // ERROR "can inline constOnlyEscapes.func1" "func literal escapes to heap" "closure converted to global"
	g(nil)
}

// Same as constOnlyEscapes, but without any call, so that no
// ReassignOracle has been built for the enclosing function yet.
func constOnlyNoCall() { // ERROR "can inline constOnlyNoCall"
	i := 42
	sink = func() { println(i) } // ERROR "can inline constOnlyNoCall.func1" "func literal escapes to heap" "closure converted to global"
}

// Variables that do not hold a constant are still captured.
func mixed(n int) {
	s := ""
	r := ""
	f := func() { // ERROR "can inline mixed.func1" "func literal does not escape" "stack closure, captured vars = \[n r\]"
		println(s)
		println(n)
		println(&r)
	}
	g(f)
}

// Nested closures capturing the same variable.
func nested() { // ERROR "can inline nested"
	s := ""
	g(func() { // ERROR "can inline nested.func1" "func literal does not escape" "closure converted to global"
		g(func() { println(s) }) // ERROR "can inline nested.func1.1" "func literal does not escape" "closure converted to global"
	})
}

// The constant is still visible to the interface conversion below, which
// therefore does not need to allocate. See also test/fixedbugs/issue4085b.go,
// where replacing the captured variable with a literal instead would turn a
// run time panic into a compile time error.
func iface() {
	i := 42
	f := func() { h(i) } // ERROR "can inline iface.func1" "func literal does not escape" "closure converted to global" "42 does not escape"
	g(f)
}

// A variable that is reassigned after being captured is captured by
// reference, so its value is not a constant.
func reassigned() {
	s := ""
	f := func() { println(s) } // ERROR "can inline reassigned.func1" "func literal does not escape" "stack closure, captured vars = \[s\]"
	s = "x"
	g(f)
}

// A variable whose address is taken is captured by reference.
func addrtaken() {
	s := ""
	f := func() { println(s) } // ERROR "can inline addrtaken.func1" "func literal does not escape" "stack closure, captured vars = \[s\]"
	_ = &s
	g(f)
}

//go:noinline
func g(func()) {
}

//go:noinline
func h(any) {
}
