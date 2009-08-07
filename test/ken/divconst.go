// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.


package main

import	"rand"

func	test(a,b,c int64);

func
main()
{
	var a, b int64;

	for i:=0; i<1e6; i++ {
		a := rand.Int63() - 1<<62;
		b = a/1;	test(a,b,1);
		b = a/2;	test(a,b,2);
		b = a/3;	test(a,b,3);
		b = a/4;	test(a,b,4);
		b = a/5;	test(a,b,5);
		b = a/6;	test(a,b,6);
		b = a/7;	test(a,b,7);
		b = a/8;	test(a,b,8);
		b = a/16;	test(a,b,16);
		b = a/32;	test(a,b,32);
		b = a/64;	test(a,b,64);
		b = a/128;	test(a,b,128);
		b = a/256;	test(a,b,256);
		b = a/16384;	test(a,b,16384);
	}
}

func
test(a,b,c int64)
{
	d := a/c;
	if d != b {
		panicln(a, b, c, d);
	}
}
