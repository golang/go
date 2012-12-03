// run

// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 4167: inlining of a (*T).Method expression taking
// its arguments from a multiple return breaks the compiler.

package main

type pa []int

type p int

func (this *pa) func1() (v *p, c int) {
	for _ = range *this {
		c++
	}
	v = (*p)(&c)
	return
}

func (this *pa) func2() p {
	return (*p).func3(this.func1())
}

func (this *p) func3(f int) p {
	return *this
}

func (this *pa) func2dots() p {
	return (*p).func3(this.func1())
}

func (this *p) func3dots(f ...int) p {
	return *this
}

func main() {
	arr := make(pa, 13)
	length := arr.func2()
	if int(length) != len(arr) {
		panic("length != len(arr)")
	}
	length = arr.func2dots()
	if int(length) != len(arr) {
		panic("length != len(arr)")
	}
}
