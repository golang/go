// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type T[T any] struct{}

// The inner T in T[T] must not conflict with the receiver base type T.
func (T[T]) m1() {}

// The receiver parameter r is declared after the receiver type parameter
// r in T[r]. An error is expected for the receiver parameter.
func (r /* ERROR "r redeclared" */ T[r]) m2() {}

type C any

// The scope of type parameter C starts after the type name (_)
// because we want to be able to use type parameters in the type
// parameter list. Hence, the 2nd C in the type parameter list below
// refers to the first C. Since constraints cannot be type parameters
// this is an error.
type _[C C /* ERROR "cannot use a type parameter as constraint" */] struct{}

// Same issue here.
func _[C C /* ERROR "cannot use a type parameter as constraint" */]() {}

// The scope of ordinary parameter C starts after the function signature.
// Therefore, the 2nd C in the parameter list below refers to the type C.
// This code is correct.
func _(C C) {} // okay
