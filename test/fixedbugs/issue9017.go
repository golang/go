// errorcheck

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 9017: Method selector shouldn't automatically dereference a named pointer type.

package main

type T struct{ x int }

func (T) mT() {}

type S struct {
	T
}

func (S) mS() {}

type P *S

type I interface {
	mT()
}

func main() {
	var s S
	s.T.mT()
	s.mT() // == s.T.mT()

	var i I
	_ = i
	i = s.T
	i = s

	var ps = &s
	ps.mS()
	ps.T.mT()
	ps.mT() // == ps.T.mT()

	i = ps.T
	i = ps

	var p P = ps
	(*p).mS()
	p.mS() // ERROR "undefined"

	i = *p
	i = p // ERROR "cannot use|incompatible types"

	p.T.mT()
	p.mT() // ERROR "undefined"

	i = p.T
	i = p // ERROR "cannot use|incompatible types"
}
