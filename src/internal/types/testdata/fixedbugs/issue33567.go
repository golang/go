// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

// Invalid calls to identifiers that shadow built-in functions should explain
// the shadowing, including for constants and parenthesized expressions.
func _(xs []int) {
	len := 0
	_ = len /* ERRORx "int is not a function \\(built-in len is shadowed\\)$" */ (xs)
	_ = ( /* ERROR "built-in len is shadowed" */ len)(xs)
}

func _(cap int, xs []int) {
	_ = cap /* ERROR "built-in cap is shadowed" */ (xs)
}

var append int

func _(xs []int) {
	_ = append /* ERROR "built-in append is shadowed" */ (xs, 0)
}

func _(m map[int]int) {
	const delete = 0
	delete /* ERROR "built-in delete is shadowed" */ (m, 0)
}

func _(make int) {
	_ = ( /* ERROR "built-in make is shadowed" */ (make))(0)
}

// Ordinary non-callable values and fields named like built-ins keep their
// usual diagnostics.
func _(x int) {
	_ = x /* ERRORx "cannot call x .*: int is not a function$" */ (0)
}

func _(holder struct{ len int }) {
	len := 0
	_ = len
	_ = holder /* ERRORx "cannot call holder.len .*: int is not a function$" */ .len(0)
}

func _(holder struct{ len bool }) {
	len := 0
	_ = len
	var _ holder.len /* ERRORx "holder.len is not a type$" */
}

func _(holder struct{ int bool }) {
	int := 0
	_ = int
	var _ holder.int /* ERRORx "holder.int is not a type$" */
}

func _() {
	len := 0
	_ = len
	_ = struct{ len bool }{( /* ERRORx "invalid field name \\(len\\) in struct literal$" */ len): false}
}

// A call may also refer to a variable hiding a predeclared type.
func _() {
	int := 0
	_ = int /* ERROR "predeclared int is shadowed" */ (0)
}

// Shadowed functions can also have the wrong number or types of arguments.
func _(xs []int) {
	len := func() int { return 0 }
	_ = len(xs /* ERROR "built-in len is shadowed" */)
}

func _(xs []int) {
	len := func(string) int { return 0 }
	_ = len(xs /* ERROR "built-in len is shadowed" */)
}

// A variable can hide a type, and a user-defined type can hide a basic type.
func _() {
	int := 0
	var x int /* ERROR "predeclared int is shadowed" */
	_ = x
}

func _() {
	type int struct{}
	var _ int = 0 /* ERROR "predeclared int is shadowed" */
}

func _() {
	type string struct{}
	_ = string("x" /* ERROR "predeclared string is shadowed" */)
}

// Predeclared constants and nil are handled in expression and statement errors.
func _(p *int) {
	nil := "haha"
	_ = p == nil /* ERROR "predeclared nil is shadowed" */
	_ = nil == p /* ERROR "predeclared nil is shadowed" */
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

func _() {
	false := 1
	for false /* ERROR "predeclared false is shadowed" */ {
	}
}

func _() {
	const false = 0
	var _ bool = false /* ERROR "predeclared false is shadowed" */
}

func _(xs []int) {
	const len = "index"
	_ = xs[len /* ERROR "built-in len is shadowed" */]
}

func _() {
	nil := "key"
	_ = map[int]int{nil /* ERROR "predeclared nil is shadowed" */: 0}
}

// Field names and declarations in a function literal are not references to
// same-named objects in the surrounding scope.
func _() {
	len := 0
	_ = len
	_ = struct{ len int }{len: "bad" /* ERRORx "cannot use .* as int value in struct literal$" */}
	var _ int = func /* ERRORx "cannot use .* as int value in variable declaration$" */ () {
		len := 0
		_ = len
	}
}

// An error before the shadowing declaration still refers to the built-in.
func _() {
	_ = len(1 /* ERRORx "invalid argument: .* for built-in len$" */)
	len := 0
	_ = len
}

// An unused declaration does not need a shadowing hint.
func _() {
	len /* ERRORx "declared and not used: len$" */ := 0
}

// Shadowing remains valid when the identifier is used consistently with its
// declaration, including calls to ordinary functions and type conversions.
func _(xs []int) {
	len := 0
	_ = len
}

func _(xs []int) {
	len := func([]int) int { return 0 }
	_ = len(xs)
	_ = (len)(xs)
}

func _() {
	type len int
	_ = len(0)
}

func _(xs []int) {
	_ = len(xs)
}

func _() {
	const iota = 0
	const true, false = 1, 2
	nil := 3
	_ = iota + true + false + nil
}

func _() {
	true := 1
	false := ""
	_ = true == false /* ERRORx "predeclared true is shadowed.*predeclared false is shadowed|predeclared false is shadowed.*predeclared true is shadowed" */
}

// Text in a constant must not suppress the diagnostic's own note.
func _() {
	const len = " (built-in len is shadowed)"
	_ = len /* ERRORx "is not a function \\(built-in len is shadowed\\)$" */ (0)
}

// Argument and result counts include types that may hide predeclared names.
func _() {
	type int string
	f := func(int) {}
	f() /* ERROR "predeclared int is shadowed" */
	g := func() {}
	g(int /* ERROR "predeclared int is shadowed" */ (""))
	_ = func() int { return /* ERROR "predeclared int is shadowed" */ }
}

func _() {
	var len int
	var _ [len /* ERROR "built-in len is shadowed" */]int
	_ = len
}

func _() {
	const len = "abc"
	_ = len[0:1:2 /* ERROR "built-in len is shadowed" */]
}
