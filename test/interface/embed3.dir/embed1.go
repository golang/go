// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "./embed0"

type X1 struct{}

func (X1) Foo() {}

type X2 struct{}

func (X2) foo() {}

type X3 struct{}

func (X3) foo(int) {}

type X4 struct{ p.M1 }

type X5 struct{ p.M1 }

func (X5) foo(int) {}

type X6 struct{ p.M2 }

type X7 struct{ p.M2 }

func (X7) foo() {}

type X8 struct{ p.M2 }

func (X8) foo(int) {}

func main() {
	var i1 interface{} = X1{}
	check(func() { _ = i1.(p.I1) }, "interface conversion: main.X1 is not p.I1: missing method Foo")

	var i2 interface{} = X2{}
	check(func() { _ = i2.(p.I2) }, "interface conversion: main.X2 is not p.I2: missing method foo")

	var i3 interface{} = X3{}
	check(func() { _ = i3.(p.I2) }, "interface conversion: main.X3 is not p.I2: missing method foo")

	var i4 interface{} = X4{}
	check(func() { _ = i4.(p.I2) }, "interface conversion: main.X4 is not p.I2: missing method foo")

	var i5 interface{} = X5{}
	check(func() { _ = i5.(p.I2) }, "interface conversion: main.X5 is not p.I2: missing method foo")

	var i6 interface{} = X6{}
	check(func() { _ = i6.(p.I2) }, "")

	var i7 interface{} = X7{}
	check(func() { _ = i7.(p.I2) }, "")

	var i8 interface{} = X8{}
	check(func() { _ = i8.(p.I2) }, "")
}

func check(f func(), msg string) {
	defer func() {
		v := recover()
		if v == nil {
			if msg == "" {
				return
			}
			panic("did not panic")
		}
		got := v.(error).Error()
		if msg != got {
			panic("want '" + msg + "', got '" + got + "'")
		}
	}()
	f()
}
