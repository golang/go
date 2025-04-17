// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test scoping of variables.


package main

var	x,y	int;

func
main() {

	x = 15;
	y = 20;
	{
		var x int;
		x = 25;
		y = 25;
		_ = x;
	}
	x = x+y;
	if(x != 40) { panic(x); }
}
