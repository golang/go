// $G $F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T struct { i int; f float; s string; next *T }

func main() {
	var t T;
	t = T(0, 7.2, "hi", &t);

	var tp *T;
	tp = &T(0, 7.2, "hi", &t);

	a1 := []int(1,2,3);
	if len(a1) != 3 { panic("a1") }
	a2 := [10]int(1,2,3);
	if len(a2) != 10 || a2[3] != 0 { panic("a2") }
	//a3 := [10]int(1,2,3,);  // BUG: trailing commas not allowed
	//if len(a3) != 10 || a2[3] != 0 { panic("a3") }

	var oai *[]int;
	oai = &[]int(1,2,3);
	if len(oai) != 3 { panic("oai") }

	at := []*T(&t, &t, &t);
	if len(at) != 3 { panic("at") }

	c := new(chan int);
	ac := []*chan int(c, c, c);
	if len(ac) != 3 { panic("ac") }

	aat := [][len(at)]*T(at, at);
	if len(aat) != 2 || len(aat[1]) != 3 { panic("at") }
	
	s := string([]byte('h', 'e', 'l', 'l', 'o'));
	if s != "hello" { panic("s") }

	m := map[string]float("one":1.0, "two":2.0, "pi":22./7.);
	if len(m) != 3 { panic("m") }
}
