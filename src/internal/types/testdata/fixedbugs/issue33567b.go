// -lang=go1.24

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func _() {
	type bool = string
	var _ bool = 0 /* ERROR "predeclared bool is shadowed" */
}

func _() {
	type int[T any] = []T
	var _ int[string] = 0 /* ERROR "predeclared int is shadowed" */
}

func _[int any](x int) {
	_ = x /* ERROR "predeclared int is shadowed" */ + 1
}

func _() {
	any := 0
	var x any /* ERROR "predeclared any is shadowed" */
	_ = x
}

func _(min int) {
	_ = min /* ERROR "built-in min is shadowed" */ (1)
}

func _() {
	type error struct{ next *error }
	var _ *error = "bad" /* ERROR "predeclared error is shadowed" */
}

func missing[int any]() int {
	var x int
	return x
}

func _() {
	_ = missing /* ERROR "predeclared int is shadowed" */ ()
}

func len[len any]() len {
	var x len
	return x
}

func _() {
	_ = len /* ERROR "built-in len is shadowed" */ ()
}

// Constraint failures are checked after the instantiation expression.
func cap[T ~string](T) {}

func _() {
	cap /* ERROR "built-in cap is shadowed" */ (1)
	cap[int /* ERROR "built-in cap is shadowed" */](1)
	cap[int, bool /* ERROR "built-in cap is shadowed" */](1)
	_ = cap[int, bool /* ERROR "built-in cap is shadowed" */]
}

func _() {
	type int[T ~string] struct{}
	var _ int[bool /* ERROR "predeclared int is shadowed" */]
	var _ int /* ERROR "predeclared int is shadowed" */ [bool, string]
}

func _() {
	type int[T any] = []T
	var _ int /* ERROR "predeclared int is shadowed" */ [bool, string]
}

func _() {
	type int = string
	type C interface {
		int | string /* ERROR "predeclared int is shadowed" */
	}
}

type T[any interface{}] any /* ERROR "predeclared any is shadowed" */

func _() {
	type int interface{ ~string }
	type Foo[T int] struct{}
	var _ Foo /* ERROR "predeclared int is shadowed" */
}

type error interface{ ~string }

func _[T error](x T) {
	_ = -x /* ERROR "predeclared error is shadowed" */
}

type F[T ~string] struct{}

type S[T any] struct{}
var min bool

// Receiver type parameters declare names rather than refer to existing ones.
func (S[* /* ERRORx "receiver type parameter \\*min must be an identifier$" */ min]) M() {}

// A later declaration in the same scope must not change an earlier binding.
func _() {
	type (
		X = F[int /* ERRORx "int does not satisfy ~string \\(int missing in ~string\\)$" */]
		int struct{}
	)
}
