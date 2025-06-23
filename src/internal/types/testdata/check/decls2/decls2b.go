// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// method declarations

package decls2

import "io"

const pi = 3.1415

func (T1) m /* ERROR "already declared" */ () {}
func (T2) m(io.Writer) {}

type T3 struct {
	f *T3
}

type T6 struct {
	x int
}

func (t *T6) m1() int {
	return t.x
}

func f() {
	var t *T6
	t.m1()
}

// Double declarations across package files
const c_double /* ERROR "redeclared" */ = 0
type t_double  /* ERROR "redeclared" */ int
var v_double /* ERROR "redeclared" */ int
func f_double /* ERROR "redeclared" */ () {}

// Blank methods need to be type-checked.
// Verify by checking that errors are reported.
func (T /* ERROR "undefined" */ ) _() {}
func (T1) _(undefined /* ERROR "undefined" */ ) {}
func (T1) _() int { return "foo" /* ERRORx "cannot use .* in return statement" */ }

// Methods with undefined receiver type can still be checked.
// Verify by checking that errors are reported.
func (Foo /* ERROR "undefined" */ ) m() {}
func (Foo /* ERROR "undefined" */ ) m(undefined /* ERROR "undefined" */ ) {}
func (Foo /* ERRORx `undefined` */ ) m() int { return "foo" /* ERRORx "cannot use .* in return statement" */ }

func (Foo /* ERROR "undefined" */ ) _() {}
func (Foo /* ERROR "undefined" */ ) _(undefined /* ERROR "undefined" */ ) {}
func (Foo /* ERROR "undefined" */ ) _() int { return "foo" /* ERRORx "cannot use .* in return statement" */ }

// Receiver declarations are regular parameter lists;
// receiver types may use parentheses, and the list
// may have a trailing comma.
type T7 struct {}

func (T7) m1() {}
func ((T7)) m2() {}
func ((*T7)) m3() {}
func (x *(T7),) m4() {}
func (x (*(T7)),) m5() {}
func (x ((*((T7)))),) m6() {}

// Check that methods with parenthesized receiver are actually present (issue #23130).
var (
	_ = T7.m1
	_ = T7.m2
	_ = (*T7).m3
	_ = (*T7).m4
	_ = (*T7).m5
	_ = (*T7).m6
)
