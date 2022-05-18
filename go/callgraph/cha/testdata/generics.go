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

func Bar() {}

func f(h func(), g func(I), k func(A), a A, b B) {
	h()

	k(a)
	g(b) // g:func(I) is not matched by instantiated[B]:func(B)

	instantiated[A](a) // static call
	instantiated[B](b) // static call
}

// WANT:
// All calls
//   (*A).Foo --> (A).Foo
//   (*B).Foo --> (B).Foo
//   f --> Bar
//   f --> instantiated[main.A]
//   f --> instantiated[main.A]
//   f --> instantiated[main.B]
//   instantiated[main.A] --> (A).Foo
//   instantiated[main.B] --> (B).Foo
