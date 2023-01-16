// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

// simplified example
func f[T ~func(T)](a, b T) {}

type F func(F)

func _() {
	var i F
	var j func(F)

	f(i, j)
	f(j, i)
}

// example from issue
func g[T interface{ Equal(T) bool }](a, b T) {}

type I interface{ Equal(I) bool }

func _() {
	var i I
	var j interface{ Equal(I) bool }

	g(i, j)
	g(j, i)
}
