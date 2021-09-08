// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"errors"
	"fmt"
)

type Option[T any] struct {
	ok  bool
	val T
}

func (o Option[T]) String() string {
	if o.ok {
		return fmt.Sprintf("Some(%v)", o.val)
	}
	return "None"
}

func Some[T any](val T) Option[T] { return Option[T]{ok: true, val: val} }
func None[T any]() Option[T]      { return Option[T]{ok: false} }

type Result[T, E any] struct {
	ok  bool
	val T
	err E
}

func (r Result[T, E]) String() string {
	if r.ok {
		return fmt.Sprintf("Ok(%v)", r.val)
	}
	return fmt.Sprintf("Err(%v)", r.err)
}

func Ok[T, E any](val T) Result[T, E]  { return Result[T, E]{ok: true, val: val} }
func Err[T, E any](err E) Result[T, E] { return Result[T, E]{ok: false, err: err} }

func main() {
	a := Some[int](1)
	b := None[int]()
	fmt.Println(a, b)

	x := Ok[int, error](1)
	y := Err[int, error](errors.New("test"))
	fmt.Println(x, y)
	// fmt.Println(x)
	_, _, _, _ = a, b, x, y
}
