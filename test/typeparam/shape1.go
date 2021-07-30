// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type I interface {
	foo() int
}

// There should be a single instantiation of f in this program.
func f[T I](x T) int {
	return x.foo()
}

type squarer int

func (x squarer) foo() int {
	return int(x*x)
}

type doubler int

func (x doubler) foo() int {
	return int(2*x)
}

func main() {
	println(f(squarer(5)))
	println(f(doubler(5)))
}
