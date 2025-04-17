// compile

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type FooType[t any] interface {
	Foo(BarType[t])
}
type BarType[t any] interface {
	Int(IntType[t]) FooType[int]
}

type IntType[t any] int

func (n IntType[t]) Foo(BarType[t]) {}
func (n IntType[_]) String()    {}
