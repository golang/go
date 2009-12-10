// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func
main() {
	var x,y int;

	x,y = simple(10,20,30);
	if x+y != 65 { panic(x+y); }
}

func
simple(ia,ib,ic int) (oa,ob int) {
	return ia+5, ib+ic;
}
