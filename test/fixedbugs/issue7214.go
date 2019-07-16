// errorcheck

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 7214: No duplicate key error for maps with interface{} key type

package p

var _ = map[interface{}]int{2: 1, 2: 1} // ERROR "duplicate key"
var _ = map[interface{}]int{int(2): 1, int16(2): 1}
var _ = map[interface{}]int{int16(2): 1, int16(2): 1} // ERROR "duplicate key"

type S string

var _ = map[interface{}]int{"a": 1, "a": 1} // ERROR "duplicate key"
var _ = map[interface{}]int{"a": 1, S("a"): 1}
var _ = map[interface{}]int{S("a"): 1, S("a"): 1} // ERROR "duplicate key"

type I interface {
	f()
}

type N int

func (N) f() {}

var _ = map[I]int{N(0): 1, N(2): 1}
var _ = map[I]int{N(2): 1, N(2): 1} // ERROR "duplicate key"
