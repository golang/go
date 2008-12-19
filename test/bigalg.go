// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"os";
	"fmt";
)

type T struct {
	a float64;
	b int64;
	c string;
	d byte;
}

var a = []int{ 1, 2, 3 }
var NIL []int;

func arraycmptest() {
	a1 := a;
	if NIL != nil {
		println("fail1:", NIL, "!= nil");
	}
	if nil != NIL {
		println("fail2: nil !=", NIL);
	}
	if a == nil || nil == a {
		println("fail3:", a, "== nil");
	}
	if a == NIL || NIL == a {
		println("fail4:", a, "==", NIL);
	}
	if a != a {
		println("fail5:", a, "!=", a);
	}
	if a1 != a {
		println("fail6:", a1, "!=", a);
	}
}

var t = T{1.5, 123, "hello", 255}
var mt = new(map[int]T)
var ma = new(map[int][]int)

func maptest() {
	mt[0] = t;
	t1 := mt[0];
	if t1.a != t.a || t1.b != t.b || t1.c != t.c || t1.d != t.d {
		println("fail: map val struct", t1.a, t1.b, t1.c, t1.d);
	}

	ma[1] = a;
	a1 := ma[1];
	if a1 != a {
		println("fail: map val array", a, a1);
	}
}

var mt1 = new(map[T]int)
var ma1 = new(map[[]int] int)

func maptest2() {
	mt1[t] = 123;
	t1 := t;
	val, ok := mt1[t1];
	if val != 123 || !ok {
		println("fail: map key struct", val, ok);
	}

	ma1[a] = 345;
	a1 := a;
	val, ok = ma1[a1];
	if val != 345 || !ok {
		panic("map key array", val, ok);
	}
}

var ct = new(chan T)
var ca = new(chan []int)

func send() {
	ct <- t;
	ca <- a;
}

func chantest() {
	go send();

	t1 := <-ct;
	if t1.a != t.a || t1.b != t.b || t1.c != t.c || t1.d != t.d {
		println("fail: chan struct", t1.a, t1.b, t1.c, t1.d);
	}

	a1 := <-ca;
	if a1 != a {
		println("fail: chan array", a, a1);
	}
}

func main() {
	arraycmptest();
	maptest();
	maptest2();
	chantest();
}
