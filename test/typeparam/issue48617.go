// run

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

func abc[T any]() {
	var b Bar[T] = func() Bar[T] {
		var b Bar[T]
		return b
	}
	var _ Foo[T] = b()
}

func main() {
	abc[int]()
}
