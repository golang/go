// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T struct { m map[int]int }
func main() {
	t := new(T);
	t.m = make(map[int]int);
	var x int;
	var ok bool;
	x, ok = t.m[0];  //bug075.go:11: bad shape across assignment - cr=1 cl=2
	_, _ = x, ok;
}
