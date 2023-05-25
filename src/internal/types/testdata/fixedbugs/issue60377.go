// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

// The type parameter P is not used in interface T1.
// T1 is a defined parameterized interface type which
// can be assigned to any other interface with the same
// methods. We cannot infer a type argument in this case
// because any type would do.

type T1[P any] interface{ m() }

func g[P any](T1[P]) {}

func _() {
	var x T1[int]
	g /* ERROR "cannot infer P" */ (x)
	g[int](x)    // int is ok for P
	g[string](x) // string is also ok for P!
}

// This is analogous to the above example,
// but uses two interface types of the same structure.

type T2[P any] interface{ m() }

func _() {
	var x T2[int]
	g /* ERROR "cannot infer P" */ (x)
	g[int](x)    // int is ok for P
	g[string](x) // string is also ok for P!
}

// Analogous to the T2 example but using an unparameterized interface T3.

type T3 interface{ m() }

func _() {
	var x T3
	g /* ERROR "cannot infer P" */ (x)
	g[int](x)    // int is ok for P
	g[string](x) // string is also ok for P!
}

// The type parameter P is not used in struct S.
// S is a defined parameterized (non-interface) type which can only
// be assigned to another type S with the same type argument.
// Therefore we can infer a type argument in this case.

type S[P any] struct{}

func g4[P any](S[P]) {}

func _() {
	var x S[int]
	g4(x)      // we can infer int for P
	g4[int](x) // int is the correct type argument
	g4[string](x /* ERROR "cannot use x (variable of type S[int]) as S[string] value in argument to g4[string]" */)
}

// This is similar to the first example but here T1 is a component
// of a func type. In this case we should be able to infer a type
// argument for P because component types must be identical even
// in the case of interfaces.
// This is a short-coming of type inference at the moment, but it
// is better to not be able to infer a type here (we can always
// supply one), than to infer the wrong type in other cases (see
// below). Finally, if we decide to accept go.dev/issues/8082,
// the behavior here is correct.

func g5[P any](func(T1[P])) {}

func _() {
	var f func(T1[int])
	g5 /* ERROR "cannot infer P" */ (f)
	g5[int](f)
	g5[string](f /* ERROR "cannot use f (variable of type func(T1[int])) as func(T1[string]) value in argument to g5[string]" */)
}

// This example would fail if we were to infer the type argument int for P
// exactly because any type argument would be ok for the first argument.
// Choosing the wrong type would cause the second argument to not match.

type T[P any] interface{}

func g6[P any](T[P], P) {}

func _() {
	var x T[int]
	g6(x, 1.2)
	g6(x, "")
}
