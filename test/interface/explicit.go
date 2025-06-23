// errorcheck

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify compiler messages about erroneous static interface conversions.
// Does not compile.

package main

type T struct {
	a int
}

var t *T

type X int

func (x *X) M() {}

type I interface {
	M()
}

var i I

type I2 interface {
	M()
	N()
}

var i2 I2

type E interface{}

var e E

func main() {
	e = t // ok
	t = e // ERROR "need explicit|need type assertion"

	// neither of these can work,
	// because i has an extra method
	// that t does not, so i cannot contain a t.
	i = t // ERROR "incompatible|missing method M"
	t = i // ERROR "incompatible|assignment$"

	i = i2 // ok
	i2 = i // ERROR "incompatible|missing method N"

	i = I(i2)  // ok
	i2 = I2(i) // ERROR "invalid|missing N method|cannot convert"

	e = E(t) // ok
	t = T(e) // ERROR "need explicit|need type assertion|incompatible|cannot convert"

	// cannot type-assert non-interfaces
	f := 2.0
	_ = f.(int) // ERROR "non-interface type|only valid for interface types|not an interface"

}

type M interface {
	M()
}

var m M

var _ = m.(int) // ERROR "impossible type assertion"

type Int int

func (Int) M(float64) {}

var _ = m.(Int) // ERROR "impossible type assertion"

var _ = m.(X) // ERROR "pointer receiver"

var ii int
var jj Int

var m1 M = ii // ERROR "incompatible|missing"
var m2 M = jj // ERROR "incompatible|wrong type for method M"

var m3 = M(ii) // ERROR "invalid|missing|cannot convert"
var m4 = M(jj) // ERROR "invalid|wrong type for M method|cannot convert"

type B1 interface {
	_() // ERROR "methods must have a unique non-blank name"
}

type B2 interface {
	M()
	_() // ERROR "methods must have a unique non-blank name"
}

type T2 struct{}

func (t *T2) M() {}
func (t *T2) _() {}

// Already reported about the invalid blank interface method above;
// no need to report about not implementing it.
var b1 B1 = &T2{}
var b2 B2 = &T2{}
