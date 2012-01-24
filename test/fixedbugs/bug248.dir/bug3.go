// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file

package main

import (
	p0 "./bug0"
	p1 "./bug1"
)

// both p0.T and p1.T are struct { X, Y int }.

var v0 p0.T
var v1 p1.T

// interfaces involving the two

type I0 interface {
	M(p0.T)
}

type I1 interface {
	M(p1.T)
}

// t0 satisfies I0 and p0.I
type t0 int

func (t0) M(p0.T) {}

// t1 satisfies I1 and p1.I
type t1 float64

func (t1) M(p1.T) {}

// check static interface assignments
var i0 I0 = t0(0) // ok
var i1 I1 = t1(0) // ok

var i2 I0 = t1(0) // ERROR "does not implement|incompatible"
var i3 I1 = t0(0) // ERROR "does not implement|incompatible"

var p0i p0.I = t0(0) // ok
var p1i p1.I = t1(0) // ok

var p0i1 p0.I = t1(0) // ERROR "does not implement|incompatible"
var p0i2 p1.I = t0(0) // ERROR "does not implement|incompatible"

func main() {
	// check that cannot assign one to the other,
	// but can convert.
	v0 = v1 // ERROR "assign"
	v1 = v0 // ERROR "assign"

	v0 = p0.T(v1)
	v1 = p1.T(v0)

	i0 = i1   // ERROR "cannot use|incompatible"
	i1 = i0   // ERROR "cannot use|incompatible"
	p0i = i1  // ERROR "cannot use|incompatible"
	p1i = i0  // ERROR "cannot use|incompatible"
	i0 = p1i  // ERROR "cannot use|incompatible"
	i1 = p0i  // ERROR "cannot use|incompatible"
	p0i = p1i // ERROR "cannot use|incompatible"
	p1i = p0i // ERROR "cannot use|incompatible"

	i0 = p0i
	p0i = i0

	i1 = p1i
	p1i = i1
}
