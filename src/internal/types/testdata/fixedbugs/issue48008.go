// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type T[P any] struct{}

func _(x interface{}) {
	switch x.(type) {
	case nil:
	case int:

	case T[int]:
	case []T[int]:
	case [10]T[int]:
	case struct{T[int]}:
	case *T[int]:
	case func(T[int]):
	case interface{m(T[int])}:
	case map[T[int]] string:
	case chan T[int]:

	case T /* ERROR "cannot use generic type T[P any] without instantiation" */ :
	case []T /* ERROR "cannot use generic type" */ :
	case [10]T /* ERROR "cannot use generic type" */ :
	case struct{T /* ERROR "cannot use generic type" */ }:
	case *T /* ERROR "cannot use generic type" */ :
	case func(T /* ERROR "cannot use generic type" */ ):
	case interface{m(T /* ERROR "cannot use generic type" */ )}:
	case map[T /* ERROR "cannot use generic type" */ ] string:
	case chan T /* ERROR "cannot use generic type" */ :

	case T /* ERROR "cannot use generic type" */ , *T /* ERROR "cannot use generic type" */ :
	}
}

// Make sure a parenthesized nil is ok.

func _(x interface{}) {
	switch x.(type) {
	case ((nil)), int:
	}
}

// Make sure we look for the predeclared nil.

func _(x interface{}) {
	type nil int
	switch x.(type) {
	case nil: // ok - this is the type nil
	}
}

func _(x interface{}) {
	var nil int
	switch x.(type) {
	case nil /* ERROR "not a type" */ : // not ok - this is the variable nil
	}
}
