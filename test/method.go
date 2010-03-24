// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type S string
type S1 string
type I int
type I1 int
type T struct {
	x int
}
type T1 T

func (s S) val() int   { return 1 }
func (s *S1) val() int { return 2 }
func (i I) val() int   { return 3 }
func (i *I1) val() int { return 4 }
//func (t T) val() int { return 7 }
func (t *T1) val() int { return 8 }

type Val interface {
	val() int
}

func val(v Val) int { return v.val() }

func main() {
	var s S
	var ps *S1
	var i I
	var pi *I1
	var pt *T1

	if s.val() != 1 {
		println("s.val:", s.val())
		panic("fail")
	}
	if S.val(s) != 1 {
		println("S.val(s):", S.val(s))
		panic("fail")
	}
	if (*S).val(&s) != 1 {
		println("(*S).val(s):", (*S).val(&s))
		panic("fail")
	}
	if ps.val() != 2 {
		println("ps.val:", ps.val())
		panic("fail")
	}
	if (*S1).val(ps) != 2 {
		println("(*S1).val(ps):", (*S1).val(ps))
		panic("fail")
	}
	if i.val() != 3 {
		println("i.val:", i.val())
		panic("fail")
	}
	if I.val(i) != 3 {
		println("I.val(i):", I.val(i))
		panic("fail")
	}
	if (*I).val(&i) != 3 {
		println("(*I).val(&i):", (*I).val(&i))
		panic("fail")
	}
	if pi.val() != 4 {
		println("pi.val:", pi.val())
		panic("fail")
	}
	if (*I1).val(pi) != 4 {
		println("(*I1).val(pi):", (*I1).val(pi))
		panic("fail")
	}
	//	if t.val() != 7 { prinln("t.val:", t.val()); panic("fail") }
	if pt.val() != 8 {
		println("pt.val:", pt.val())
		panic("fail")
	}
	if (*T1).val(pt) != 8 {
		println("(*T1).val(pt):", (*T1).val(pt))
		panic("fail")
	}

	if val(s) != 1 {
		println("s.val:", val(s))
		panic("fail")
	}
	if val(ps) != 2 {
		println("ps.val:", val(ps))
		panic("fail")
	}
	if val(i) != 3 {
		println("i.val:", val(i))
		panic("fail")
	}
	if val(pi) != 4 {
		println("pi.val:", val(pi))
		panic("fail")
	}
	//	if val(t) != 7 { println("t.val:", val(t)); panic("fail") }
	if val(pt) != 8 {
		println("pt.val:", val(pt))
		panic("fail")
	}

	//	if Val.val(i) != 3 { println("Val.val(i):", Val.val(i)); panic("fail") }
}
