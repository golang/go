// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type A1 [2]uint64
type A2 [2]uint64

func (a A1) m() A1 { return a }
func (a A2) m() A2 { return a }

func f[B any, T interface {
	A1 | A2
	m() T
}](v T) {
}

func _() {
	var v A2
	// Use function type inference to infer type A2 for T.
	// Don't use constraint type inference before function
	// type inference for typed arguments, otherwise it would
	// infer type [2]uint64 for T which doesn't have method m
	// (was the bug).
	f[int](v)
}

// Keep using constraint type inference before function type
// inference for untyped arguments so we infer type float64
// for E below, and not int (which would not work).
func g[S ~[]E, E any](S, E) {}

func _() {
	var s []float64
	g[[]float64](s, 0)
}

// Keep using constraint type inference after function
// type inference for untyped arguments so we infer
// missing type arguments for which we only have the
// untyped arguments as starting point.
func h[E any, R []E](v E) R { return R{v} }
func _() []int              { return h(0) }
