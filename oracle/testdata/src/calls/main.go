package main

import (
	"fmt"
)

// Tests of call-graph queries.
// See go.tools/oracle/oracle_test.go for explanation.
// See calls.golden for expected query results.

func A(x *int) { // @pointsto pointsto-A-x "x"
	// @callers callers-A "^"
	// @callstack callstack-A "^"
}

func B(x *int) { // @pointsto pointsto-B-x "x"
	// @callers callers-B "^"
}

func foo() {
}

// apply is not (yet) treated context-sensitively.
func apply(f func(x *int), x *int) {
	f(x) // @callees callees-apply "f"
	// @callers callers-apply "^"
}

// store *is* treated context-sensitively,
// so the points-to sets for pc, pd are precise.
func store(ptr **int, value *int) {
	*ptr = value
	// @callers callers-store "^"
}

func call(f func() *int) {
	// Result points to anon function.
	f() // @pointsto pointsto-result-f "f"

	// Target of call is anon function.
	f() // @callees callees-main.call-f "f"

	// @callers callers-main.call "^"
}

func main() {
	var a, b int
	go apply(A, &a) // @callees callees-main-apply1 "app"
	defer apply(B, &b)

	var c, d int
	var pc, pd *int // @pointsto pointsto-pc "pc"
	store(&pc, &c)
	store(&pd, &d)
	_ = pd // @pointsto pointsto-pd "pd"

	call(func() *int {
		// We are called twice from main.call
		// @callers callers-main.anon "^"
		return &a
	})

	// Errors
	_ = "no function call here"   // @callees callees-err-no-call "no"
	print("builtin")              // @callees callees-err-builtin "builtin"
	_ = string("type conversion") // @callees callees-err-conversion "str"
	call(nil)                     // @callees callees-err-bad-selection "call\\(nil"
	if false {
		main() // @callees callees-err-deadcode1 "main"
	}
	var nilFunc func()
	nilFunc() // @callees callees-err-nil-func "nilFunc"
	var i interface {
		f()
	}
	i.f() // @callees callees-err-nil-interface "i.f"

	i = new(myint)
	i.f() // @callees callees-not-a-wrapper "f"

	// statically dispatched calls. Handled specially by callees, so test that they work.
	foo()         // @callees callees-static-call "foo"
	fmt.Println() // @callees callees-qualified-call "Println"
	m := new(method)
	m.f() // @callees callees-static-method-call "f"
}

type myint int

func (myint) f() {
	// @callers callers-not-a-wrapper "^"
}

type method int

func (method) f() {
}

var dynamic = func() {}

func deadcode() {
	main() // @callees callees-err-deadcode2 "main"
	// @callers callers-err-deadcode "^"
	// @callstack callstack-err-deadcode "^"

	// Within dead code, dynamic calls have no callees.
	dynamic() // @callees callees-err-deadcode3 "dynamic"
}

// This code belongs to init.
var global = 123 // @callers callers-global "global"

// The package initializer may be called by other packages' inits, or
// in this case, the root of the callgraph.  The source-level init functions
// are in turn called by it.
func init() {
	// @callstack callstack-init "^"
}
