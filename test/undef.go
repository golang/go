// errorcheck

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test line numbers in error messages.
// Does not compile.

package main

var (
	_ = x	// ERROR "undefined.*x"
	_ = x	// ERROR "undefined.*x"
	_ = x	// ERROR "undefined.*x"
)

type T struct {
	y int
}

func foo() *T { return &T{y: 99} }
func bar() int { return y }	// ERROR "undefined.*y"

type T1 struct {
	y1 int
}

func foo1() *T1 { return &T1{y1: 99} }
var y1 = 2
func bar1() int { return y1 }

func f1(val interface{}) {
	switch v := val.(type) {
	default:
		println(v)
	}
}

func f2(val interface{}) {
	switch val.(type) {
	default:
		println(v)	// ERROR "undefined.*v"
	}
}
