// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type A[T any] interface {
	m()
}

type Z struct {
	a,b int
}

func (z *Z) m() {
}

func test[T any]() {
	var a A[T] = &Z{}
	f := a.m
	f()
}
func main() {
	test[string]()
}
