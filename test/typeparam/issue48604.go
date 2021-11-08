// build -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type Foo[T any] interface {
	CreateBar() Bar[T]
}

type Bar[T any] func() Bar[T]

func (f Bar[T]) CreateBar() Bar[T] {
	return f
}

func abc[R any]() {
	var _ Foo[R] = Bar[R](nil)()
}

func main() {
	abc[int]()
}