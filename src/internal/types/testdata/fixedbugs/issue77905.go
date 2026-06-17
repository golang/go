// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type M[T any] interface {
	m() T
}

func f[T any](x interface{ m() T }) T { return x.m() }
func g[T any](x M[T]) T               { return x.m() }

type S struct{}

// inference must work here even though m is declared only afterwards
// (inference must type-check m as needed)
var _ = f(S{})
var _ = g(S{})

func _() {
	var s S
	var _ = f(s)
	var _ = g(s)
}

func (S) m() int { return 0 }

var _ = f(S{})
var _ = g(S{})
