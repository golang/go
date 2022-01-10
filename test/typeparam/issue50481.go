// errorcheck -G=3

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type _[_ any] struct{}
type _[_, _ any] struct{}             // ERROR "cannot have multiple blank type parameters"
type _[_, _, _ any] struct{}          // ERROR "cannot have multiple blank type parameters"
type _[a, _, b, _, c, _ any] struct{} // ERROR "cannot have multiple blank type parameters"

func _[_ any]()                {}
func _[_, _ any]()             {} // ERROR "cannot have multiple blank type parameters"
func _[_, _, _ any]()          {} // ERROR "cannot have multiple blank type parameters"
func _[a, _, b, _, c, _ any]() {} // ERROR "cannot have multiple blank type parameters"

type S[P1, P2 any] struct{}

func (_ S[_, _]) m() {} // this is ok
