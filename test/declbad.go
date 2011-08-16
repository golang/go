// errchk $G -e $F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Incorrect short declarations and redeclarations.

package main

func f1() int                    { return 1 }
func f2() (float32, int)         { return 1, 2 }
func f3() (float32, int, string) { return 1, 2, "3" }

func main() {
	{
		// simple redeclaration
		i := f1()
		i := f1() // ERROR "redeclared|no new"
		_ = i
	}
	{
		// change of type for f
		i, f, s := f3()
		f, g, t := f3() // ERROR "redeclared|cannot assign|incompatible"
		_, _, _, _, _ = i, f, s, g, t
	}
	{
		// change of type for i
		i, f, s := f3()
		j, i, t := f3() // ERROR "redeclared|cannot assign|incompatible"
		_, _, _, _, _ = i, f, s, j, t
	}
	{
		// no new variables
		i, f, s := f3()
		i, f := f2() // ERROR "redeclared|no new"
		_, _, _ = i, f, s
	}
	{
		// single redeclaration
		i, f, s := f3()
		i := 1 // ERROR "redeclared|no new|incompatible"
		_, _, _ = i, f, s
	}
	// double redeclaration
	{
		i, f, s := f3()
		i, f := f2() // ERROR "redeclared|no new"
		_, _, _ = i, f, s
	}
	{
		// triple redeclaration
		i, f, s := f3()
		i, f, s := f3() // ERROR "redeclared|no new"
		_, _, _ = i, f, s
	}
}
