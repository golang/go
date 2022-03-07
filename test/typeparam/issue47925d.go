// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type I[T any] interface {
	foo()
}

type J[T any] interface {
	foo()
	bar()
}

//go:noinline
func f[T J[T]](x T, g func(T) T) I[T] {
	// contains a cast between two nonempty interfaces
	// Also make sure we don't evaluate g(x) twice.
	return I[T](J[T](g(x)))
}

type S struct {
	x int
}

func (s *S) foo() {}
func (s *S) bar() {}

var cnt int

func inc(s *S) *S {
	cnt++
	return s
}

func main() {
	i := f(&S{x: 7}, inc)
	if i.(*S).x != 7 {
		panic("bad")
	}
	if cnt != 1 {
		panic("multiple calls")
	}
}
