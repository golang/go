// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// comparisons

package expr2

func _bool() {
	const t = true == true
	const f = true == false
	_ = t /* ERROR cannot compare */ < f
	_ = 0 /* ERROR mismatched types untyped int and untyped bool */ == t
	var b bool
	var x, y float32
	b = x < y
	_ = b
	_ = struct{b bool}{x < y}
}

// corner cases
var (
	v0 = nil == nil // ERROR operator == not defined on untyped nil
)

func arrays() {
	// basics
	var a, b [10]int
	_ = a == b
	_ = a != b
	_ = a /* ERROR < not defined */ < b
	_ = a == nil /* ERROR invalid operation.*mismatched types */

	type C [10]int
	var c C
	_ = a == c

	type D [10]int
	var d D
	_ = c /* ERROR mismatched types */ == d

	var e [10]func() int
	_ = e /* ERROR \[10\]func\(\) int cannot be compared */ == e
}

func structs() {
	// basics
	var s, t struct {
		x int
		a [10]float32
		_ bool
	}
	_ = s == t
	_ = s != t
	_ = s /* ERROR < not defined */ < t
	_ = s == nil /* ERROR invalid operation.*mismatched types */

	type S struct {
		x int
		a [10]float32
		_ bool
	}
	type T struct {
		x int
		a [10]float32
		_ bool
	}
	var ss S
	var tt T
	_ = s == ss
	_ = ss /* ERROR mismatched types */ == tt

	var u struct {
		x int
		a [10]map[string]int
	}
	_ = u /* ERROR cannot compare */ == u
}

func pointers() {
	// nil
	_ = nil == nil // ERROR operator == not defined on untyped nil
	_ = nil != nil // ERROR operator != not defined on untyped nil
	_ = nil /* ERROR < not defined */ < nil
	_ = nil /* ERROR <= not defined */ <= nil
	_ = nil /* ERROR > not defined */ > nil
	_ = nil /* ERROR >= not defined */ >= nil

	// basics
	var p, q *int
	_ = p == q
	_ = p != q

	_ = p == nil
	_ = p != nil
	_ = nil == q
	_ = nil != q

	_ = p /* ERROR < not defined */ < q
	_ = p /* ERROR <= not defined */ <= q
	_ = p /* ERROR > not defined */ > q
	_ = p /* ERROR >= not defined */ >= q

	// various element types
	type (
		S1 struct{}
		S2 struct{}
		P1 *S1
		P2 *S2
	)
	var (
		ps1 *S1
		ps2 *S2
		p1 P1
		p2 P2
	)
	_ = ps1 == ps1
	_ = ps1 /* ERROR mismatched types */ == ps2
	_ = ps2 /* ERROR mismatched types */ == ps1

	_ = p1 == p1
	_ = p1 /* ERROR mismatched types */ == p2

	_ = p1 == ps1
}

func channels() {
	// basics
	var c, d chan int
	_ = c == d
	_ = c != d
	_ = c == nil
	_ = c /* ERROR < not defined */ < d

	// various element types (named types)
	type (
		C1 chan int
		C1r <-chan int
		C1s chan<- int
		C2 chan float32
	)
	var (
		c1 C1
		c1r C1r
		c1s C1s
		c1a chan int
		c2 C2
	)
	_ = c1 == c1
	_ = c1 /* ERROR mismatched types */ == c1r
	_ = c1 /* ERROR mismatched types */ == c1s
	_ = c1r /* ERROR mismatched types */ == c1s
	_ = c1 == c1a
	_ = c1a == c1
	_ = c1 /* ERROR mismatched types */ == c2
	_ = c1a /* ERROR mismatched types */ == c2

	// various element types (unnamed types)
	var (
		d1 chan int
		d1r <-chan int
		d1s chan<- int
		d1a chan<- int
		d2 chan float32
	)
	_ = d1 == d1
	_ = d1 == d1r
	_ = d1 == d1s
	_ = d1r /* ERROR mismatched types */ == d1s
	_ = d1 == d1a
	_ = d1a == d1
	_ = d1 /* ERROR mismatched types */ == d2
	_ = d1a /* ERROR mismatched types */ == d2
}

// for interfaces test
type S1 struct{}
type S11 struct{}
type S2 struct{}
func (*S1) m() int
func (*S11) m() int
func (*S11) n()
func (*S2) m() float32

func interfaces() {
	// basics
	var i, j interface{ m() int }
	_ = i == j
	_ = i != j
	_ = i == nil
	_ = i /* ERROR < not defined */ < j

	// various interfaces
	var ii interface { m() int; n() }
	var k interface { m() float32 }
	_ = i == ii
	_ = i /* ERROR mismatched types */ == k

	// interfaces vs values
	var s1 S1
	var s11 S11
	var s2 S2

	_ = i == 0 /* ERROR cannot convert */
	_ = i /* ERROR mismatched types */ == s1
	_ = i == &s1
	_ = i == &s11

	_ = i /* ERROR mismatched types */ == s2
	_ = i /* ERROR mismatched types */ == &s2

	// issue #28164
	// testcase from issue
	_ = interface{}(nil) == [ /* ERROR slice can only be compared to nil */ ]int(nil)

	// related cases
	var e interface{}
	var s []int
	var x int
	_ = e == s // ERROR slice can only be compared to nil
	_ = s /* ERROR slice can only be compared to nil */ == e
	_ = e /* ERROR operator < not defined on interface */ < x
	_ = x < e // ERROR operator < not defined on interface
}

func slices() {
	// basics
	var s []int
	_ = s == nil
	_ = s != nil
	_ = s /* ERROR < not defined */ < nil

	// slices are not otherwise comparable
	_ = s /* ERROR slice can only be compared to nil */ == s
	_ = s /* ERROR < not defined */ < s
}

func maps() {
	// basics
	var m map[string]int
	_ = m == nil
	_ = m != nil
	_ = m /* ERROR < not defined */ < nil

	// maps are not otherwise comparable
	_ = m /* ERROR map can only be compared to nil */ == m
	_ = m /* ERROR < not defined */ < m
}

func funcs() {
	// basics
	var f func(int) float32
	_ = f == nil
	_ = f != nil
	_ = f /* ERROR < not defined */ < nil

	// funcs are not otherwise comparable
	_ = f /* ERROR func can only be compared to nil */ == f
	_ = f /* ERROR < not defined */ < f
}
