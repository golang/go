//go:build ignore
// +build ignore

package main

// Test of generic function calls.

type I interface {
	Foo()
}

type A struct{}

func (a A) Foo() {}

type B struct{}

func (b B) Foo() {}

func instantiated[X I](x X) {
	x.Foo()
}

var a A
var b B

func main() {
	instantiated[A](a) // static call
	instantiated[B](b) // static call

	local[C]().Foo()

	lambda[A]()()()
}

func local[X I]() I {
	var x X
	return x
}

type C struct{}

func (c C) Foo() {}

func lambda[X I]() func() func() {
	return func() func() {
		var x X
		return x.Foo
	}
}

// WANT:
// All calls
//   (*C).Foo --> (C).Foo
//   (A).Foo$bound --> (A).Foo
//   instantiated[main.A] --> (A).Foo
//   instantiated[main.B] --> (B).Foo
//   main --> (*C).Foo
//   main --> (A).Foo$bound
//   main --> (C).Foo
//   main --> instantiated[main.A]
//   main --> instantiated[main.B]
//   main --> lambda[main.A]
//   main --> lambda[main.A]$1
//   main --> local[main.C]
// Reachable functions
//   (*C).Foo
//   (A).Foo
//   (A).Foo$bound
//   (B).Foo
//   (C).Foo
//   instantiated[main.A]
//   instantiated[main.B]
//   lambda[main.A]
//   lambda[main.A]$1
//   local[main.C]
// Reflect types
//   *C
//   C
