// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var b[10] float;

func
main()
{
	var a[10] float;

	for i:=short(5); i<10; i=i+1 {
		a[i] = float(i);
	}

	s := float(0);
	for i:=5; i<10; i=i+1 {
		s = s + a[i];
	}

	if s != 35 { panic s; }

	for i:=short(5); i<10; i=i+1 {
		b[i] = float(i);
	}

	s := float(0);
	for i:=5; i<10; i=i+1 {
		s = s + b[i];
	}

	if s != 35 { panic s; }

	b := new([100]int);
	for i:=0; i<100; i=i+1 {
		b[i] = i;
	}

	s := 0;
	for i:=0; i<100; i=i+1 {
		s = s+b[i];
	}

	if s != 4950 { panic s; }
}
