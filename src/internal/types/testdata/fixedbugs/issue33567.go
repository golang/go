// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import "errors"

var _ = errors.New

// Errors caused by a declaration that hides a predeclared object, an imported
// package, a type, or a function point out that declaration.

func _(xs []int) {
	len := 0
	_ = len /* ERRORx "int is not a function \\(predeclared len is shadowed by declaration at .*\\)$" */ (xs)
}

func helper() int { return 0 }

func _() {
	helper := 3
	_ = helper /* ERROR "function helper is shadowed" */ ()
}

func _() {
	int := 0
	var _ int /* ERROR "predeclared int is shadowed" */
}

type fancyMap = map[string]string

func _(fancyMap fancyMap) {
	var _ fancyMap /* ERROR "type fancyMap is shadowed" */
}

func _() {
	var errors []error
	_ = errors.New /* ERROR "imported package errors is shadowed" */ ("x")
	_ = errors.Missing /* ERRORx "has no field or method Missing\\)$" */
}

func _(p *int) {
	nil := "haha"
	_ = p == nil /* ERROR "predeclared nil is shadowed" */
	_ = nil == p /* ERROR "predeclared nil is shadowed" */
	p = nil /* ERROR "predeclared nil is shadowed" */
}

func _() {
	var iota int
	const _ = iota /* ERROR "predeclared iota is shadowed" */
}

func _() {
	true := "yes"
	if true /* ERROR "predeclared true is shadowed" */ {
	}
}

// A type name that hides a predeclared one is annotated once per type.
func _() {
	type int struct{}
	var x int
	var _ map[int]int = 0 /* ERRORx "map\\[int ?/\\* (with int declared at|not predeclared)[^]]*\\]int value" */
	var _ string = x /* ERRORx "struct type int ?/\\* (with int declared at|not predeclared)" */
}

// Different types with the same name (go.dev/issue/9655).
type Range struct{ Start, End int }

func _(r *Range) {
	type Range struct{ Start, End int }
	*r = Range /* ERRORx "\\(different types Range declared at .* and .*\\)$" */ {}
}

// Constraints are verified later, but the note refers to the callee's
// declaration at the call.
func cap[T ~string](T) {}

func _() {
	cap /* ERRORx "cap is shadowed by declaration at .*:79:6\\)$" */ (1)
}

// Hiding variables, using a value that hides a function, and using a built-in
// before a shadowing declaration need no explanation.
func _() {
	f := func() {}
	_ = f
	{
		f := 1
		_ = f /* ERRORx "int is not a function$" */ ()
	}
}

func _() {
	helper := "s"
	var _ int = helper /* ERRORx "as int value in variable declaration$" */
}

func _() {
	_ = len(1 /* ERRORx "invalid argument: .* for built-in len$" */)
	len := 0
	_ = len
}

// Shadowing remains valid.
func _(xs []int) {
	len := func([]int) int { return 0 }
	_ = len(xs)
	const iota, true, nil = 0, 1, 2
	_ = iota + true + nil
}
