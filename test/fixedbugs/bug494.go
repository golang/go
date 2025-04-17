// run

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Gccgo incorrectly executed functions multiple times when they
// appeared in a composite literal that required a conversion between
// different interface types.

package main

type MyInt int

var c MyInt

func (c *MyInt) S(i int) {
	*c = MyInt(i)
}

func (c *MyInt) V() int {
	return int(*c)
}

type i1 interface {
	S(int)
	V() int
}

type i2 interface {
	V() int
}

type s struct {
	i i2
}

func f() i1 {
	c++
	return &c
}

func main() {
	p := &s{f()}
	if v := p.i.V(); v != 1 {
		panic(v)
	}
	if c != 1 {
		panic(c)
	}
}
