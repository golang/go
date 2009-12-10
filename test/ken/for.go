// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.


package main

func
main() {
	var t,i int;

	for i=0; i<100; i=i+1 {
		t = t+i;
	}
	if t != 50*99  { panic(t); }
}
