// -lang=go1.23 -gotypesalias=1

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package aliasTypes

type _ = int
type _[P any] = int

// A type alias may have fewer type parameters than its RHS.
type RHS[P any, Q ~int] struct {
	p P
	q Q
}

type _[P any] = RHS[P, int]

// Or it may have more type parameters than its RHS.
type _[P any, Q ~int, R comparable] = RHS[P, Q]

// The type parameters of a type alias must implement the
// corresponding type constraints of the type parameters
// on the RHS (if any)
type _[P any, Q ~int] = RHS[P, Q]
type _[P any, Q int] = RHS[P, Q]
type _[P int | float64] = RHS[P, int]
type _[P, Q any] = RHS[P, Q /* ERROR "Q does not satisfy ~int" */]

// ----------------------------------------------------------------------------
// NOTE: The code below does now work yet.
// TODO: Implement this.

// A generic type alias may be used like any other generic type.
type A[P any] = RHS[P, int]

func _(a A /* ERROR "not a generic type" */ [string]) {
	a.p = "foo"
	a.q = 42
}
