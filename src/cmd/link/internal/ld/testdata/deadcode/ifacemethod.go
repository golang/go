// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that a method of a reachable type is not necessarily
// live even if it matches an interface method, as long as
// the type is never converted to an interface.

package main

type I interface{ M() }

type T int

func (T) M() { println("XXX") }

var p *T
var e interface{}

func main() {
	p = new(T) // used T, but never converted to interface in any reachable code
	e.(I).M()  // used I and I.M
}

func Unused() { // convert T to interface, but this function is not reachable
	var i I = T(0)
	i.M()
}

var Unused2 interface{} = T(1) // convert T to interface, in an unreachable global initializer
