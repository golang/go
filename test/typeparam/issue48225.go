// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "reflect"

type Foo[T any] struct {
	val int
}

func (foo Foo[T]) Get() *T {
	if foo.val != 1 {
		panic("bad val field in Foo receiver")
	}
	return new(T)
}

var (
	newInt    = Foo[int]{val: 1}.Get
	newString = Foo[string]{val: 1}.Get
)

func main() {
	i := newInt()
	s := newString()

	if t := reflect.TypeOf(i).String(); t != "*int" {
		panic(t)
	}
	if t := reflect.TypeOf(s).String(); t != "*string" {
		panic(t)
	}
}
