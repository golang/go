// $G $F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T struct {
	i    int
	f    float64
	s    string
	next *T
}

type R struct {
	num int
}

func itor(a int) *R {
	r := new(R)
	r.num = a
	return r
}

func eq(a []*R) {
	for i := 0; i < len(a); i++ {
		if a[i].num != i {
			panic("bad")
		}
	}
}

func teq(t *T, n int) {
	for i := 0; i < n; i++ {
		if t == nil || t.i != i {
			panic("bad")
		}
		t = t.next
	}
	if t != nil {
		panic("bad")
	}
}

type P struct {
	a, b int
}

func NewP(a, b int) *P {
	return &P{a, b}
}

func main() {
	var t T
	t = T{0, 7.2, "hi", &t}

	var tp *T
	tp = &T{0, 7.2, "hi", &t}

	tl := &T{i: 0, next: &T{i: 1, next: &T{i: 2, next: &T{i: 3, next: &T{i: 4}}}}}
	teq(tl, 5)

	a1 := []int{1, 2, 3}
	if len(a1) != 3 {
		panic("a1")
	}
	a2 := [10]int{1, 2, 3}
	if len(a2) != 10 || cap(a2) != 10 {
		panic("a2")
	}

	a3 := [10]int{1, 2, 3}
	if len(a3) != 10 || a2[3] != 0 {
		panic("a3")
	}

	var oai []int
	oai = []int{1, 2, 3}
	if len(oai) != 3 {
		panic("oai")
	}

	at := [...]*T{&t, tp, &t}
	if len(at) != 3 {
		panic("at")
	}

	c := make(chan int)
	ac := []chan int{c, c, c}
	if len(ac) != 3 {
		panic("ac")
	}

	aat := [][len(at)]*T{at, at}
	if len(aat) != 2 || len(aat[1]) != 3 {
		panic("aat")
	}

	s := string([]byte{'h', 'e', 'l', 'l', 'o'})
	if s != "hello" {
		panic("s")
	}

	m := map[string]float64{"one": 1.0, "two": 2.0, "pi": 22. / 7.}
	if len(m) != 3 {
		panic("m")
	}

	eq([]*R{itor(0), itor(1), itor(2), itor(3), itor(4), itor(5)})
	eq([]*R{{0}, {1}, {2}, {3}, {4}, {5}})

	p1 := NewP(1, 2)
	p2 := NewP(1, 2)
	if p1 == p2 {
		panic("NewP")
	}
}
