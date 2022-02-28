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
func f[T J[T]](x T) I[T] {
	// contains a cast between two nonempty interfaces
	return I[T](J[T](x))
}

type S struct {
	x int
}

func (s *S) foo() {}
func (s *S) bar() {}

func main() {
	i := f(&S{x: 7})
	if i.(*S).x != 7 {
		panic("bad")
	}
}
