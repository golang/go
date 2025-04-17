// run

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that type parameter methods are handled correctly, even when
// the instantiating type argument has additional methods.

package main

func main() {
	F(X(0))
}

type I interface{ B() }

func F[T I](t T) {
	CallMethod(t)
	MethodExpr[T]()(t)
	MethodVal(t)()
}

func CallMethod[T I](t T)       { t.B() }
func MethodExpr[T I]() func(T)  { return T.B }
func MethodVal[T I](t T) func() { return t.B }

type X int

func (X) A() { panic("FAIL") }
func (X) B() {}
func (X) C() { panic("FAIL") }
