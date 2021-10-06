// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type myInt int

// Parameterized type declarations

type T1[P any] P

type T2[P any] struct {
        f P
        g int // int should still be in scope chain
}

type List[P any] []P

// Alias type declarations cannot have type parameters. Syntax error.
// TODO(gri) Disabled for now as we don't check syntax error here.
// type A1[P any] = /* ERROR cannot be alias */ P

// But an alias may refer to a generic, uninstantiated type.
type A2 = List
var _ A2[int]
var _ A2 /* ERROR without instantiation */

type A3 = List[int]
var _ A3

// Parameterized type instantiations

var x int
type _ x /* ERROR not a type */ [int]

type _ int /* ERROR not a generic type */ [int]
type _ myInt /* ERROR not a generic type */ [int]

// TODO(gri) better error messages
type _ T1[int]
type _ T1[x /* ERROR not a type */ ]
type _ T1 /* ERROR got 2 arguments but 1 type parameters */ [int, float32]

var _ T2[int] = T2[int]{}

var _ List[int] = []int{1, 2, 3}
var _ List[[]int] = [][]int{{1, 2, 3}}
var _ List[List[List[int]]]

// Parameterized types containing parameterized types

type T3[P any] List[P]

var _ T3[int] = T3[int](List[int]{1, 2, 3})

// Self-recursive generic types are not permitted

type self1[P any] self1 /* ERROR illegal cycle */ [P]
type self2[P any] *self2[P] // this is ok
