// compile

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type Handler func(in ...interface{})

type Foo[T any] struct{}

func (b *Foo[T]) Bar(in ...interface{}) {}

func (b *Foo[T]) Init() {
	_ = Handler(b.Bar)
}

func main() {
	c := &Foo[int]{}
	c.Init()
}
