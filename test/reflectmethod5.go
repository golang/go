// run

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 38515: failed to mark the method wrapper
// reflect.Type.Method itself as REFLECTMETHOD.

package main

import "reflect"

var called bool

type foo struct{}

func (foo) X() { called = true }

var h = reflect.Type.Method

func main() {
	v := reflect.ValueOf(foo{})
	m := h(v.Type(), 0)
	f := m.Func.Interface().(func(foo))
	f(foo{})
	if !called {
		panic("FAIL")
	}
}
