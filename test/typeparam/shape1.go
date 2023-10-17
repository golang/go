// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type I interface {
	foo() int
}

// There should be one instantiation of f for both squarer and doubler.
// Similarly, there should be one instantiation of f for both *incrementer and *decrementer.
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

type incrementer int16

func (x *incrementer) foo() int {
	return int(*x+1)
}

type decrementer int32

func (x *decrementer) foo() int{
	return int(*x-1)
}

func main() {
	println(f(squarer(5)))
	println(f(doubler(5)))
	var i incrementer = 5
	println(f(&i))
	var d decrementer = 5
	println(f(&d))
}
