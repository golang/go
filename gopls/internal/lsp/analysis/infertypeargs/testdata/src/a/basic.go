// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the infertyepargs checker.

package a

func f[T any](T) {}

func g[T any]() T { var x T; return x }

func h[P interface{ ~*T }, T any]() {}

func _() {
	f[string]("hello") // want "unnecessary type arguments"
	f[int](2)          // want "unnecessary type arguments"
	_ = g[int]()
	h[*int, int]() // want "unnecessary type arguments"
}
