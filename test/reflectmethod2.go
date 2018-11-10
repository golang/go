// run

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The linker can prune methods that are not directly called or
// assigned to interfaces, but only if reflect.Type.MethodByName is
// never used. Test it here.

package main

import reflect1 "reflect"

var called = false

type M int

func (m M) UniqueMethodName() {
	called = true
}

var v M

type MyType interface {
	MethodByName(string) (reflect1.Method, bool)
}

func main() {
	var t MyType = reflect1.TypeOf(v)
	m, _ := t.MethodByName("UniqueMethodName")
	m.Func.Interface().(func(M))(v)
	if !called {
		panic("UniqueMethodName not called")
	}
}
