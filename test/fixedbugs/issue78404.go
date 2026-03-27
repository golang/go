// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type Iface[IO any] interface {
	Foo()
}

type underlyingIfaceImpl struct{}

func (e *underlyingIfaceImpl) Foo() {}

type Impl1[IO any] struct {
	underlyingIfaceImpl
}

type Impl2 struct {
	underlyingIfaceImpl
}

func NewImpl1[IO any]() Iface[IO] {
	return &Impl1[IO]{}
}

var alwaysFalse = false

func main() {
	val := NewImpl1[int]()
	if alwaysFalse { // dead branch
		val = &Impl2{}
	}
	val.Foo() // must not panic
}
