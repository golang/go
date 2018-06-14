// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test shift.

package main

var	ians	[18]int;
var	uans	[18]uint;
var	pass	string;

func
testi(i int, t1,t2,t3 int) {
	n := ((t1*3) + t2)*2 + t3;
	if i != ians[n] {
		print("itest ", t1,t2,t3,pass,
			" is ", i, " sb ", ians[n], "\n");
	}
}

func
index(t1,t2,t3 int) int {
	return ((t1*3) + t2)*2 + t3;
}

func
testu(u uint, t1,t2,t3 int) {
	n := index(t1,t2,t3);
	if u != uans[n] {
		print("utest ", t1,t2,t3,pass,
			" is ", u, " sb ", uans[n], "\n");
	}
}

func
main() {
	var i int;
	var u,c uint;

	/*
	 * test constant evaluations
	 */
	pass = "con";	// constant part

	testi( int(1234) <<    0, 0,0,0);
	testi( int(1234) >>    0, 0,0,1);
	testi( int(1234) <<    5, 0,1,0);
	testi( int(1234) >>    5, 0,1,1);

	testi(int(-1234) <<    0, 1,0,0);
	testi(int(-1234) >>    0, 1,0,1);
	testi(int(-1234) <<    5, 1,1,0);
	testi(int(-1234) >>    5, 1,1,1);

	testu(uint(5678) <<    0, 2,0,0);
	testu(uint(5678) >>    0, 2,0,1);
	testu(uint(5678) <<    5, 2,1,0);
	testu(uint(5678) >>    5, 2,1,1);

	/*
	 * test variable evaluations
	 */
	pass = "var";	// variable part

	for t1:=0; t1<3; t1++ {	// +int, -int, uint
	for t2:=0; t2<3; t2++ {	// 0, +small, +large
	for t3:=0; t3<2; t3++ {	// <<, >>
		switch t1 {
		case 0:	i =  1234;
		case 1:	i = -1234;
		case 2:	u =  5678;
		}
		switch t2 {
		case 0:	c =    0;
		case 1:	c =    5;
		case 2:	c = 1025;
		}
		switch t3 {
		case 0:	i <<= c; u <<= c;
		case 1:	i >>= c; u >>= c;
		}
		switch t1 {
		case 0:	testi(i,t1,t2,t3);
		case 1:	testi(i,t1,t2,t3);
		case 2:	testu(u,t1,t2,t3);
		}
	}
	}
	}
}

func
init() {
	/*
	 * set the 'correct' answer
	 */

	ians[index(0,0,0)] =   1234;
	ians[index(0,0,1)] =   1234;
	ians[index(0,1,0)] =  39488;
	ians[index(0,1,1)] =     38;
	ians[index(0,2,0)] =      0;
	ians[index(0,2,1)] =      0;

	ians[index(1,0,0)] =  -1234;
	ians[index(1,0,1)] =  -1234;
	ians[index(1,1,0)] = -39488;
	ians[index(1,1,1)] =    -39;
	ians[index(1,2,0)] =      0;
	ians[index(1,2,1)] =     -1;

	uans[index(2,0,0)] =   5678;
	uans[index(2,0,1)] =   5678;
	uans[index(2,1,0)] = 181696;
	uans[index(2,1,1)] =    177;
	uans[index(2,2,0)] =      0;
	uans[index(2,2,1)] =      0;
}
