// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file shows some examples of methods on type-parameterized types.

package p

// Parameterized types may have methods.
type T1[A any] struct{ a A }

// When declaring a method for a parameterized type, the "instantiated"
// receiver type acts as an implicit declaration of the type parameters
// for the receiver type. In the example below, method m1 on type T1 has
// the receiver type T1[A] which declares the type parameter A for use
// with this method. That is, within the method m1, A stands for the
// actual type argument provided to an instantiated T1.
func (t T1[A]) m1() A { return t.a }

// For instance, if T1 is instantiated with the type int, the type
// parameter A in m1 assumes that type (int) as well and we can write
// code like this:
var x T1[int]
var _ int = x.m1()

// Because the type parameter provided to a parameterized receiver type
// is declared through that receiver declaration, it must be an identifier.
// It cannot possibly be some other type because the receiver type is not
// instantiated with concrete types, it is standing for the parameterized
// receiver type.
func (t T1[[ /* ERROR must be an identifier */ ]int]) m2() {}

// Note that using what looks like a predeclared identifier, say int,
// as type parameter in this situation is deceptive and considered bad
// style. In m3 below, int is the name of the local receiver type parameter
// and it shadows the predeclared identifier int which then cannot be used
// anymore as expected.
// This is no different from locally re-declaring a predeclared identifier
// and usually should be avoided. There are some notable exceptions; e.g.,
// sometimes it makes sense to use the identifier "copy" which happens to
// also be the name of a predeclared built-in function.
func (t T1[int]) m3() { var _ int = 42 /* ERROR cannot use 42 .* as int */ }

// The names of the type parameters used in a parameterized receiver
// type don't have to match the type parameter names in the declaration
// of the type used for the receiver. In our example, even though T1 is
// declared with type parameter named A, methods using that receiver type
// are free to use their own name for that type parameter. That is, the
// name of type parameters is always local to the declaration where they
// are introduced. In our example we can write a method m2 and use the
// name X instead of A for the type parameter w/o any difference.
func (t T1[X]) m4() X { return t.a }

// If the receiver type is parameterized, type parameters must always be
// provided: this simply follows from the general rule that a parameterized
// type must be instantiated before it can be used. A method receiver
// declaration using a parameterized receiver type is no exception. It is
// simply that such receiver type expressions perform two tasks simultaneously:
// they declare the (local) type parameters and then use them to instantiate
// the receiver type. Forgetting to provide a type parameter leads to an error.
func (t T1 /* ERROR generic type .* without instantiation */ ) m5() {}

// However, sometimes we don't need the type parameter, and thus it is
// inconvenient to have to choose a name. Since the receiver type expression
// serves as a declaration for its type parameters, we are free to choose the
// blank identifier:
func (t T1[_]) m6() {}

// Naturally, these rules apply to any number of type parameters on the receiver
// type. Here are some more complex examples.
type T2[A, B, C any] struct {
        a A
        b B
        c C
}

// Naming of the type parameters is local and has no semantic impact:
func (t T2[A, B, C]) m1() (A, B, C) { return t.a, t.b, t.c }
func (t T2[C, B, A]) m2() (C, B, A) { return t.a, t.b, t.c }
func (t T2[X, Y, Z]) m3() (X, Y, Z) { return t.a, t.b, t.c }

// Type parameters may be left blank if they are not needed:
func (t T2[A, _, C]) m4() (A, C) { return t.a, t.c }
func (t T2[_, _, X]) m5() X { return t.c }
func (t T2[_, _, _]) m6() {}

// As usual, blank names may be used for any object which we don't care about
// using later. For instance, we may write an unnamed method with a receiver
// that cannot be accessed:
func (_ T2[_, _, _]) _() int { return 42 }

// Because a receiver parameter list is simply a parameter list, we can
// leave the receiver argument away for receiver types.
type T0 struct{}
func (T0) _() {}
func (T1[A]) _() {}

// For now, a lone type parameter is not permitted as RHS in a type declaration (issue #45639).
// // A generic receiver type may constrain its type parameter such
// // that it must be a pointer type. Such receiver types are not
// // permitted.
// type T3a[P interface{ ~int | ~string | ~float64 }] P
// 
// func (T3a[_]) m() {} // this is ok
// 
// type T3b[P interface{ ~unsafe.Pointer }] P
// 
// func (T3b /* ERROR invalid receiver */ [_]) m() {}
// 
// type T3c[P interface{ *int | *string }] P
// 
// func (T3c /* ERROR invalid receiver */ [_]) m() {}
