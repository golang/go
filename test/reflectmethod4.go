// run

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The linker can prune methods that are not directly called or
// assigned to interfaces, but only if reflect.Value.Method is
// never used. Test it here.

package main

import "reflect"

var called = false

type M int

func (m M) UniqueMethodName() {
	called = true
}

var v M

func main() {
	reflect.ValueOf(v).Method(0).Interface().(func())()
	if !called {
		panic("UniqueMethodName not called")
	}
}
