// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type myInt int

// Parameterized type declarations

// For now, a lone type parameter is not permitted as RHS in a type declaration (issue #45639).
type T1[P any] P // ERROR "cannot use a type parameter as RHS in type declaration"

type T2[P any] struct {
        f P
        g int // int should still be in scope chain
}

type List[P any] []P

// Alias type declarations cannot have type parameters.
// Issue #46477 proposes to change that.
type A1[P any] = /* ERROR "cannot be alias" */ struct{}

// Pending clarification of #46477 we disallow aliases
// of generic types.
type A2 = List // ERROR "cannot use generic type"
var _ A2[int]
var _ A2

type A3 = List[int]
var _ A3

// Parameterized type instantiations

var x int
type _ x /* ERROR "not a type" */ [int]

type _ int /* ERROR "not a generic type" */ [] // ERROR "expected type argument list"
type _ myInt /* ERROR "not a generic type" */ [] // ERROR "expected type argument list"

// TODO(gri) better error messages
type _ T1[] // ERROR "expected type argument list"
type _ T1[x /* ERROR "not a type" */ ]
type _ T1 /* ERROR "too many type arguments for type T1: have 2, want 1" */ [int, float32]

var _ T2[int] = T2[int]{}

var _ List[int] = []int{1, 2, 3}
var _ List[[]int] = [][]int{{1, 2, 3}}
var _ List[List[List[int]]]

// Parameterized types containing parameterized types

type T3[P any] List[P]

var _ T3[int] = T3[int](List[int]{1, 2, 3})

// Self-recursive generic types are not permitted

type self1[P any] self1 /* ERROR "invalid recursive type" */ [P]
type self2[P any] *self2[P] // this is ok
