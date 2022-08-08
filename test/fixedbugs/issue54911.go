// compile

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type Set[T comparable] map[T]struct{}

func (s Set[T]) Add() Set[T] {
	return s
}

func (s Set[T]) Copy() Set[T] {
	return Set[T].Add(s)
}

func main() {
	_ = Set[int]{42: {}}
}
