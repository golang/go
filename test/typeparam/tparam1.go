// errorcheck -G

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Basic type parameter list type-checking (not syntax) errors.

package tparam1

// The predeclared identifier "any" may be used in place of interface{}.
var _ any

func _(_ any)

type _[_ any] struct{}

const N = 10

type (
	_                     []struct{}  // slice
	_                     [N]struct{} // array
	_[T any]              struct{}
	_[T, T any]           struct{} // ERROR "T redeclared"
	_[T1, T2 any, T3 any] struct{}
)

func _[T any]()             {}
func _[T, T any]()          {} // ERROR "T redeclared"
func _[T1, T2 any](x T1) T2 { panic(0) }

// Type parameters are visible from opening [ to end of function.
type C interface{}

func _[T interface{}]()        {}
func _[T C]()                  {}
func _[T struct{}]()           {} // ok if #48424 is accepted
func _[T interface{ m() T }]() {}
func _[T1 interface{ m() T2 }, T2 interface{ m() T1 }]() {
	var _ T1
}

// TODO(gri) expand this
