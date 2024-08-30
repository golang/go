// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Like ifacemethod2.go, this tests that a method *is* live
// if the type is "indirectly" converted to an interface
// using reflection with a method descriptor as intermediate.

package main

import "reflect"

type S int

func (s S) M() { println("S.M") }

type I interface{ M() }

type T float64

func (t T) F(s S) {}

func main() {
	var t T
	ft := reflect.TypeOf(t).Method(0).Type
	at := ft.In(1)
	v := reflect.New(at).Elem()
	v.Interface().(I).M()
}
