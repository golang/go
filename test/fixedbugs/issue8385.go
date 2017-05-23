// errorcheck

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 8385: provide a more descriptive error when a method expression
// is called without a receiver.

package main

type Fooer interface {
	Foo(i, j int)
}

func f(x int) {
}

type I interface {
	M(int)
}
type T struct{}

func (t T) M(x int) {
}

func g() func(int)

func main() {
	Fooer.Foo(5, 6) // ERROR "not enough arguments in call to method expression Fooer.Foo"

	var i I
	var t *T

	g()()    // ERROR "not enough arguments in call to g\(\)"
	f()      // ERROR "not enough arguments in call to f"
	i.M()    // ERROR "not enough arguments in call to i\.M"
	I.M()    // ERROR "not enough arguments in call to method expression I\.M"
	t.M()    // ERROR "not enough arguments in call to t\.M"
	T.M()    // ERROR "not enough arguments in call to method expression T\.M"
	(*T).M() // ERROR "not enough arguments in call to method expression \(\*T\)\.M"
}
