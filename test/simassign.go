// $G $F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var	a,b,c,d,e,f,g,h,i int;

func
printit()
{
	print(a,b,c,d,e,f,g,h,i,"\n");
}

func
testit() bool
{
	if a+b+c+d+e+f+g+h+i != 45 {
		print("sum does not add to 45\n");
		printit();
		panic();
	}
	return	a == 1 &&
		b == 2 &&
		c == 3 &&
		d == 4 &&
		e == 5 &&
		f == 6 &&
		g == 7 &&
		h == 8 &&
		i == 9;
}

func
swap(x, y int) (u, v int) {
	return y, x
}

func
main()
{
	a = 1;
	b = 2;
	c = 3;
	d = 4;
	e = 5;
	f = 6;
	g = 7;
	h = 8;
	i = 9;

	if !testit() { panic("init val\n"); }

	for z:=0; z<100; z++ {
		a,b,c,d, e,f,g,h,i = b,c,d,a, i,e,f,g,h;

		if testit() {
			if z == 19 {
				break;
			}
			print("on ", z, "th iteration\n");
			printit();
			panic();
		}
	}

	if !testit() {
		print("final val\n");
		printit();
		panic();
	}

	a, b = swap(1, 2);
	if a != 2 || b != 1 {
		panic("bad swap");
	}
//BUG	a, b = swap(swap(a, b));
//	if a != 2 || b != 1 {
//		panic("bad swap");
//	}
}
