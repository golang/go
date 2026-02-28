// run

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func returnOption[T any](n int) Option[T] {
	if n == 1 {
		return Some[T]{}
	} else {
		return None{}
	}
}

type Option[T any] interface {
	sealedOption()
}

type Some[T any] struct {
	val T
}

func (s Some[T]) Value() T {
	return s.val
}

func (s Some[T]) sealedOption() {}

type None struct{}

func (s None) sealedOption() {}

func main() {
	s := returnOption[int](1)
	_ = s.(Some[int])

	s = returnOption[int](0)
	_ = s.(None)

	switch (any)(s).(type) {
	case Some[int]:
		panic("s is a Some[int]")
	case None:
		// ok
	default:
		panic("oops")
	}
}
