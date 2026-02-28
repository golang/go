// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This test only uses MethodByName() with constant names
// of methods to look up. These methods need to be kept,
// but other methods must be eliminated.

package main

import "reflect"

type S int

func (s S) M() { println("S.M") }

func (s S) N() { println("S.N") }

type T float64

func (t T) F(s S) {}

func main() {
	var t T
	meth, _ := reflect.TypeOf(t).MethodByName("F")
	ft := meth.Type
	at := ft.In(1)
	v := reflect.New(at).Elem()
	methV := v.MethodByName("M")
	methV.Call([]reflect.Value{v})
}
