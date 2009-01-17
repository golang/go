// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

export type S string
export type S1 string
export type I int
export type I1 int
export type T struct { x int }
export type T1 T

func (s S) val() int { return 1 }
func (s *S1) val() int { return 2 }
func (i I) val() int { return 3 }
func (i *I1) val() int { return 4 }
//func (t T) val() int { return 7 }
func (t *T1) val() int { return 8 }

export type Val interface {
	val() int
}

func val(v Val) int {
	return v.val()
}

func main() {
	var s S;
	var ps *S1;
	var i I;
	var pi *I1;
	var t T;
	var pt *T1;

	if s.val() != 1 { panicln("s.val:", s.val()) }
	if ps.val() != 2 { panicln("ps.val:", ps.val()) }
	if i.val() != 3 { panicln("i.val:", i.val()) }
	if pi.val() != 4 { panicln("pi.val:", pi.val()) }
//	if t.val() != 7 { panicln("t.val:", t.val()) }
	if pt.val() != 8 { panicln("pt.val:", pt.val()) }

	if val(s) != 1 { panicln("s.val:", val(s)) }
	if val(ps) != 2 { panicln("ps.val:", val(ps)) }
	if val(i) != 3 { panicln("i.val:", val(i)) }
	if val(pi) != 4 { panicln("pi.val:", val(pi)) }
//	if val(t) != 7 { panicln("t.val:", val(t)) }
	if val(pt) != 8 { panicln("pt.val:", val(pt)) }

}
