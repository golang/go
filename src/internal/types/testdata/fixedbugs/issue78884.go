// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

// multiple assignments
func f0()                         { return }
func f1() (_ int)                 { return }
func f2() (_ string, _ int)       { return }
func f3() (_, _ string, _ int)    { return }
func f4() (_, _, _ string, _ int) { return }

func _() {
	var _ string = f0 /* ERROR "f0() (no value) used as value" */ ()
	var _ string = f1 /* ERROR "cannot use f1() (value of type int) as string value in variable declaration" */ ()
	var _, _ string = f2 /* ERROR "cannot use 2nd function result (value of type int) as string value in multiple assignment" */ ()
	var _, _, _ string = f3 /* ERROR "cannot use 3rd function result (value of type int) as string value in multiple assignment" */ ()
	var _, _, _, _ string = f4 /* ERROR "cannot use 4th function result (value of type int) as string value in multiple assignment" */ ()
}

// comma, ok values
func _() {
	var (
		m map[string]int
		s string
		i int
		b bool
	)
	_, _, _, _ = m, s, i, b
	s = m /* ERROR "cannot use m[s] (map index expression of type int) as string value in assignment" */ [s]
	s, b = m /* ERROR "cannot use m[s] (map index expression of type int) as string value in multiple assignment" */ [s]
	i, s = m /* ERROR "cannot use ok value of (comma, ok) expression (untyped bool value) as string value in multiple assignment" */ [s]
}

// test case from issue

func f() (int, bool) { return 1, false }

func g() {
	var s string
	s, b := f /* ERROR "cannot use 1st function result (value of type int) as string value in multiple assignment" */ ()
	_, _ = s, b
}
