// errorcheck

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type I interface{ m() }
type T struct{ m func() }
type M struct{}

func (M) m() {}

func main() {
	var t T
	var m M
	var i I

	i = m
	// types2 does not give extra error "T.m is a field, not a method"
	i = t // ERROR "not a method|has no methods|does not implement I"
	_ = i
}
