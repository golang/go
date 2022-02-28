// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"reflect"
)

type Stream[T any] struct {
}

func (s Stream[T]) DropWhile() Stream[T] {
	return Pipe[T, T](s)
}

func Pipe[T, R any](s Stream[T]) Stream[R] {
	it := func(fn func(R) bool) {
	}
	fmt.Println(reflect.TypeOf(it).String())
	return Stream[R]{}
}

func main() {
	s := Stream[int]{}
	s = s.DropWhile()
}
