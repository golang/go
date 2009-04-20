// $G $F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Correct short declarations and redeclarations.

package main

func f1() int { return 1 }
func f2() (float, int) { return 1, 2 }
func f3() (float, int, string) { return 1, 2, "3" }

func x() (s string) {
	a, b, s := f3();
	return	// tests that result var is in scope for redeclaration
}

func main() {
	i, f, s := f3();
	j, f := f2();	// redeclare f
	k := f1();
	m, g, s := f3();
	m, h, s := f3();
	{
		// new block should be ok.
		i, f, s := f3();
		j, f := f2();	// redeclare f
		k := f1();
		m, g, s := f3();
		m, h, s := f3();
	}
	if x() != "3" {
		println("g() failed");
	}
}
