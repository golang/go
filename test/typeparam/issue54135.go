// compile

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type Foo struct{}

func (Foo) Blanker() {}

type Bar[T any] interface {
	Blanker()
}

type Baz interface {
	Some()
}

func check[T comparable](p Bar[T]) {
	_, _ = p.(any)
	_, _ = p.(Baz)
}

func main() {
	check[int](Foo{})
}
