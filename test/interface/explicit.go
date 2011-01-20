// errchk $G -e $D/$F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Static error messages about interface conversions.

package main

type T struct {
	a int
}

var t *T

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
	i = t // ERROR "incompatible|missing M method"
	t = i // ERROR "incompatible|need type assertion"

	i = i2 // ok
	i2 = i // ERROR "incompatible|missing N method"

	i = I(i2)  // ok
	i2 = I2(i) // ERROR "invalid|missing N method"

	e = E(t) // ok
	t = T(e) // ERROR "need explicit|need type assertion|incompatible"
}

type M interface {
	M()
}

var m M

var _ = m.(int) // ERROR "impossible type assertion"

type Int int

func (Int) M(float64) {}

var _ = m.(Int) // ERROR "impossible type assertion"

var ii int
var jj Int

var m1 M = ii // ERROR "incompatible|missing"
var m2 M = jj // ERROR "incompatible|wrong type for M method"

var m3 = M(ii) // ERROR "invalid|missing"
var m4 = M(jj) // ERROR "invalid|wrong type for M method"
