// compile

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file checks simple code using type parameters.

package smoketest

// type parameters for functions
func f1[P any]() {}
func f2[P1, P2 any, P3 any]() {}
func f3[P interface{}](x P, y T1[int]) {}

// function instantiations
var _ = f1[int]
var _ = f2[int, string, struct{}]
var _ = f3[bool]

// type parameters for types
type T1[P any] struct{}
type T2[P1, P2 any, P3 any] struct{}
type T3[P interface{}] interface{}

// type instantiations
type _ T1[int]
type _ T2[int, string, struct{}]
type _ T3[bool]

// methods
func (T1[P]) m1()           {}
func (T1[_]) m2()           {}
func (x T2[P1, P2, P3]) m() {}

// type lists
type _ interface {
	m1()
	m2()
	int | float32 | string
	m3()
}

// embedded instantiated types
type _ struct {
	f1, f2 int
	T1[int]
	T2[int, string, struct{}]
	T3[bool]
}

type _ interface {
	m1()
	m2()
	T3[bool]
}
