// errchk $G -e $F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Incorrect short declarations and redeclarations.

package main

func f1() int { return 1 }
func f2() (float, int) { return 1, 2 }
func f3() (float, int, string) { return 1, 2, "3" }

func main() {
	{
		// simple redeclaration
		i := f1();
		i := f1();	// ERROR "redeclared"
	{
		// change of type for f
		i, f, s := f3();
		f, g, t := f3();	// ERROR "redeclared"
	}
	{
		// change of type for i
		i, f, s := f3();
		j, i, t := f3();	// ERROR "redeclared"
	}
	{
		// no new variables
		i, f, s := f3();
		i, f := f2();	// ERROR "redeclared"
	}
	{
		// single redeclaration
		i, f, s := f3();
		i := f1();	// ERROR "redeclared"
	}
		// double redeclaration
	{
		i, f, s := f3();
		i, f := f2();	// ERROR "redeclared"
	}
	{
		// triple redeclaration
		i, f, s := f3();
		i, f, s := f3();	// ERROR "redeclared"
	}
}
