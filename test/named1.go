// errorcheck

// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that basic operations on named types are valid
// and preserve the type.
// Does not compile.

package main

type Bool bool

type Map map[int]int

func (Map) M() {}

type Slice []byte

var slice Slice

func asBool(Bool)     {}
func asString(String) {}

type String string

func main() {
	var (
		b    Bool = true
		i, j int
		c    = make(chan int)
		m    = make(Map)
	)

	asBool(b)
	asBool(!b)
	asBool(true)
	asBool(*&b)
	asBool(Bool(true))
	asBool(1 != 2) // ok now
	asBool(i < j)  // ok now

	_, b = m[2] // ERROR "cannot .* bool.*type Bool"

	var inter interface{}
	_, b = inter.(Map) // ERROR "cannot .* bool.*type Bool"
	_ = b

	var minter interface {
		M()
	}
	_, b = minter.(Map) // ERROR "cannot .* bool.*type Bool"
	_ = b

	_, bb := <-c
	asBool(bb) // ERROR "cannot use.*type bool.*as type Bool"
	_, b = <-c // ERROR "cannot .* bool.*type Bool"
	_ = b

	asString(String(slice)) // ok
}
