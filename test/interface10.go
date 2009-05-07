// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

const Value = 1e12

type Inter interface { M() int64 }

type T int64
func (t T) M() int64 { return t }
var t = T(Value)
var pt = &t
var ti Inter = t
var pti = &ti

type S struct { Inter }
var s = S{ ti }
var ps = &s

type SP struct { *Inter }
var sp = SP{ &ti }
var psp = &sp

var i Inter
var pi = &i

var ok = true

func check(v int64, s string) {
	if v != Value {
		println(s, v);
		ok = false;
	}
}

func main() {
	check(t.M(), "t.M");
	check(pt.M(), "pt.M");
	check(ti.M(), "ti.M");
	check(pti.M(), "pti.M");
	check(s.M(), "s.M");
	check(ps.M(), "ps.M");
	check(sp.M(), "sp.M");
	check(psp.M(), "psp.M");

	i = t;
	check(i.M(), "i.M - i = t");
	check(pi.M(), "pi.M - i = t");

	i = pt;
	check(i.M(), "i.M - i = pt");
	check(pi.M(), "pi.M - i = pt");

	i = s;
	check(i.M(), "i.M - i = s");
	check(pi.M(), "pi.M - i = s");

	i = ps;
	check(i.M(), "i.M - i = ps");
	check(pi.M(), "pi.M - i = ps");

	i = sp;
	check(i.M(), "i.M - i = sp");
	check(pi.M(), "pi.M - i = sp");

	i = psp;
	check(i.M(), "i.M - i = psp");
	check(pi.M(), "pi.M - i = psp");

	if !ok {
		println("BUG: interface10");
		sys.Exit(1)
	}
}
