// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type S string
type S1 string
type I int
type I1 int
type T struct { x int }
type T1 T

func (s S) val() int { return 1 }
func (s *S1) val() int { return 2 }
func (i I) val() int { return 3 }
func (i *I1) val() int { return 4 }
//func (t T) val() int { return 7 }
func (t *T1) val() int { return 8 }

type Val interface {
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
	var pt *T1;

	if s.val() != 1 { panicln("s.val:", s.val()) }
	if S.val(s) != 1 { panicln("S.val(s):", S.val(s)) }
	if (*S).val(&s) != 1 { panicln("(*S).val(s):", (*S).val(&s)) }
	if ps.val() != 2 { panicln("ps.val:", ps.val()) }
	if (*S1).val(ps) != 2 { panicln("(*S1).val(ps):", (*S1).val(ps)) }
	if i.val() != 3 { panicln("i.val:", i.val()) }
	if I.val(i) != 3 { panicln("I.val(i):", I.val(i)) }
	if (*I).val(&i) != 3 { panicln("(*I).val(&i):", (*I).val(&i)) }
	if pi.val() != 4 { panicln("pi.val:", pi.val()) }
	if (*I1).val(pi) != 4 { panicln("(*I1).val(pi):", (*I1).val(pi)) }
//	if t.val() != 7 { panicln("t.val:", t.val()) }
	if pt.val() != 8 { panicln("pt.val:", pt.val()) }
	if (*T1).val(pt) != 8 { panicln("(*T1).val(pt):", (*T1).val(pt)) }

	if val(s) != 1 { panicln("s.val:", val(s)) }
	if val(ps) != 2 { panicln("ps.val:", val(ps)) }
	if val(i) != 3 { panicln("i.val:", val(i)) }
	if val(pi) != 4 { panicln("pi.val:", val(pi)) }
//	if val(t) != 7 { panicln("t.val:", val(t)) }
	if val(pt) != 8 { panicln("pt.val:", val(pt)) }
	
//	if Val.val(i) != 3 { panicln("Val.val(i):", Val.val(i)) }
}
