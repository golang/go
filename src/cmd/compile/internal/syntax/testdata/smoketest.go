// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains basic generic code snippets.

package p

// type parameter lists
type B[P any] struct{}
type _[P interface{}] struct{}
type _[P B] struct{}
type _[P B[P]] struct{}

type _[A, B, C any] struct{}
type _[A, B, C B] struct{}
type _[A, B, C B[A, B, C]] struct{}
type _[A1, A2 B1, A3 B2, A4, A5, A6 B3] struct{}

type _[A interface{}] struct{}
type _[A, B interface{ m() }] struct{}

type _[A, B, C any] struct{}

// in functions
func _[P any]()
func _[P interface{}]()
func _[P B]()
func _[P B[P]]()

// type instantiations
type _ T[int]

// in expressions
var _ = T[int]{}

// in embedded types
type _ struct{ T[int] }

// interfaces
type _ interface {
	m()
	~int
}

type _ interface {
	~int | ~float | ~string
	~complex128
	underlying(underlying underlying) underlying
}

type _ interface {
	T
	T[int]
}

// tricky cases
func _(T[P], T[P1, P2])
func _(a [N]T)

type _ struct {
	T[P]
	T[P1, P2]
	f[N]
}
type _ interface {
	m()

	// instantiated types
	T[ /* ERROR empty type argument list */ ]
	T[P]
	T[P1, P2]
}
