// cmpout

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test string operations including printing.

package main

func main() {
	var c string

	a := `abc`
	b := `xyz`

	/* print a literal */
	print(`abc`)

	/* print a variable */
	print(b, "-")

	/* catenate literals */
	print(`abc`+`xyz`, "-")

	/* catenate variables */
	print(a+b, "-")

	/* compare literals */
	if `abc` == `xyz` || `abc` != "abc" || `abc` > `xyz` {
		panic("compare literals")
	}

	/* compare variables */
	if a == b || a != a || a > b {
		panic("compare variables")
	}

	/* cat */
	c = a + b
	print(c, "-")

	/* catequal */
	c = a
	c += b
	print(c, "-")

	/* clumsy evaluation */
	c = b
	c = a + c
	print(c, "-")

	/* len */
	if len(c) != 6 {
		print("len ", len(c))
		panic("fail")
	}

	/* index strings */
	for i := 0; i < len(c); i = i + 1 {
		if c[i] != (a + b)[i] {
			print("index ", i, " ", c[i], " ", (a + b)[i])
			panic("fail")
		}
	}

	/* slice strings */
	print(c[0:3], c[3:])

	print("\n")

	/* create string with integer constant */
	c = string('x')
	if c != "x" {
		panic("create int " + c)
	}

	/* create string with integer variable */
	v := 'x'
	c = string(v)
	if c != "x" {
		panic("create int " + c)
	}

	/* create string with byte array */
	var z1 [3]byte
	z1[0] = 'a'
	z1[1] = 'b'
	z1[2] = 'c'
	c = string(z1[0:])
	if c != "abc" {
		panic("create byte array " + c)
	}

	/* create string with int array */
	var z2 [3]rune
	z2[0] = 'a'
	z2[1] = '\u1234'
	z2[2] = 'c'
	c = string(z2[0:])
	if c != "a\u1234c" {
		panic("create int array " + c)
	}

	/* create string with byte array pointer */
	z3 := new([3]byte)
	z3[0] = 'a'
	z3[1] = 'b'
	z3[2] = 'c'
	c = string(z3[0:])
	if c != "abc" {
		panic("create array pointer " + c)
	}
}
