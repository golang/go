// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func _() {
	NewS( /* ERROR cannot infer T */ ) .M()
}

type S struct {}

func NewS[T any]() *S { panic(0) }

func (_ *S /* ERROR S is not a generic type */ [T]) M()
