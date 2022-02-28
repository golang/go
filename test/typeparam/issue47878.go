// compile -G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type Src1[T any] func() Src1[T]

func (s *Src1[T]) Next() {
	*s = (*s)()
}

type Src2[T any] []func() Src2[T]

func (s Src2[T]) Next() {
	_ = s[0]()
}

type Src3[T comparable] map[T]func() Src3[T]

func (s Src3[T]) Next() {
	var a T
	_ = s[a]()
}

type Src4[T any] chan func() T

func (s Src4[T]) Next() {
	_ = (<-s)()
}

type Src5[T any] func() Src5[T]

func (s Src5[T]) Next() {
	var x interface{} = s
	_ = (x.(Src5[T]))()
}

func main() {
	var src1 Src1[int]
	src1.Next()

	var src2 Src2[int]
	src2.Next()

	var src3 Src3[string]
	src3.Next()

	var src4 Src4[int]
	src4.Next()

	var src5 Src5[int]
	src5.Next()
}
