// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type Src[T any] func() Src[T]

func Seq[T any]() Src[T] {
	return nil
}

func Seq2[T1 any, T2 any](v1 T1, v2 T2) Src[T2] {
	return nil
}

func main() {
	// Type args fully supplied
	Seq[int]()
	// Partial inference of type args
	Seq2[int](5, "abc")
	// Full inference of type args
	Seq2(5, "abc")
}
