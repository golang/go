// run

// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// issue 5056: escape analysis not applied to wrapper functions

package main

type Foo int16

func (f Foo) Esc() *int{
	x := int(f)
	return &x
}

type iface interface {
	Esc() *int
}

var bar, foobar *int

func main() {
	var quux iface
	var x Foo
	
	quux = x
	bar = quux.Esc()
	foobar = quux.Esc()
	if bar == foobar {
		panic("bar == foobar")
	}
}
