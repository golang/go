// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests.

package decls

import "base"
import base2 "base"

const c0 int = 0
const c1 float = 1.
const (
	c2 byte = 2;
	c3 int = 3;
	c4 float = 4.;
)


type (
	Node0 base.Node
	Node1 *base2.Node
)

type T0 byte
type T1 T0
type (
	T2 [10]T0;
	T3 map [string] int;
)
type T4 struct {
	f1, f2, f3 int;
	f4 [] float;
};
type (
	T5 *T4;
)

type F0 ()
type F1 (a int)
type F2 (a, b int, c float)
type F3 () bool
type F4 (a int) (z T5, ok bool)
type F5 (a, b int, c float) (z T5, ok bool)
type F6 (a int, b float) bool
type F7 (a int, b float, c, d *bool) bool

type T6 chan int
type T7 <- chan *T6
type T8 chan <- *T6

type T9 struct {
	p *T9;
	q [] map [int] *T9;
	f *(x, y *T9) *T9;
}

type T11 struct {
	p *T10;
}

type T10 struct {
	p *T11;
}

type T12 struct {
	p *T12
}

type I0 interface {}
type I1 interface {
	Do0(q *I0);
	Do1(p *I1) bool;
}
type I2 interface {
	M0();
	M1(a int);
	M2(a, b int, c float);
	M3() bool;
	M4(a int) (z T5, ok bool);
	M5(a, b int, c float) (z T5, ok bool);
}


var v0 int
var v1 float = c1

var (
	v2 T2;
	v3 struct {
		f1, f2, f3 *M0;
	}
)


func f0() {}
func f1(a int) {}
func f2(a, b int, c float) {}
func f3() bool { return false; }
func f4(a int) (z T5, ok bool) {}
func f5(a, b int, c float) (z T5, ok bool) {
	u, v := 0, 0;
	return;
}


func (p *T4) m0() {}
func (p *T4) m1(a int) {}
func (p *T4) m2(a, b int, c float) {}
func (p *T4) m3() bool { return false; }
func (p *T4) m4(a int) (z T5, ok bool) { return; }
func (p *T4) m5(a, b int, c float) (z T5, ok bool) {
	L: var x = a;
}


func f2() {
	type T *T14;
}
type T14 int;
