// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typeInference

// As of issue #51527, type-type inference has been disabled.

// basic inference
type Tb[P ~*Q, Q any] int

func _() {
	var x Tb /* ERROR "not enough type arguments for type Tb: have 1, want 2" */ [*int]
	var y Tb[*int, int]
	x = y /* ERRORx `cannot use y .* in assignment` */
	_ = x
}

// recursive inference
type Tr[A any, B *C, C *D, D *A] int

func _() {
	var x Tr /* ERROR "not enough type arguments for type Tr: have 1, want 4" */ [string]
	var y Tr[string, ***string, **string, *string]
	var z Tr[int, ***int, **int, *int]
	x = y /* ERRORx `cannot use y .* in assignment` */
	x = z // ERRORx `cannot use z .* as Tr`
	_ = x
}

// other patterns of inference
type To0[A any, B []A] int
type To1[A any, B struct{ a A }] int
type To2[A any, B [][]A] int
type To3[A any, B [3]*A] int
type To4[A any, B any, C struct {
	a A
	b B
}] int

func _() {
	var _ To0 /* ERROR "not enough type arguments for type To0: have 1, want 2" */ [int]
	var _ To1 /* ERROR "not enough type arguments for type To1: have 1, want 2" */ [int]
	var _ To2 /* ERROR "not enough type arguments for type To2: have 1, want 2" */ [int]
	var _ To3 /* ERROR "not enough type arguments for type To3: have 1, want 2" */ [int]
	var _ To4 /* ERROR "not enough type arguments for type To4: have 2, want 3" */ [int, string]
}

// failed inference
type Tf0[A, B any] int
type Tf1[A any, B ~struct {
	a A
	c C
}, C any] int

func _() {
	var _ Tf0 /* ERROR "not enough type arguments for type Tf0: have 1, want 2" */ [int]
	var _ Tf1 /* ERROR "not enough type arguments for type Tf1: have 1, want 3" */ [int]
}
