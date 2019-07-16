// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func f()
func leaf()

var f1called, f2called, f3called bool

func main() {
	f()
	if !f1called {
		panic("f1 not called")
	}
	if !f2called {
		panic("f2 not called")
	}
	leaf()
	if !f3called {
		panic("f3 not called")
	}
}

func f1() { f1called = true }
func f2() { f2called = true }
func f3() { f3called = true }

func unreachable() {
	panic("unreachable function called")
}
