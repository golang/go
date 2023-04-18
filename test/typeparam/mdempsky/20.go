// run

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that method expressions with a derived receiver type and
// promoted methods work correctly.

package main

func main() {
	F[int]()
	F[string]()
}

func F[X any]() {
	call(T[X].M, T[X].N)
}

func call[X any](fns ...func(T[X]) int) {
	for want, fn := range fns {
		if have := fn(T[X]{}); have != want {
			println("FAIL:", have, "!=", want)
		}
	}
}

type T[X any] struct {
	E1
	*E2[*X]
}

type E1 struct{}
type E2[_ any] struct{}

func (E1) M() int     { return 0 }
func (*E2[_]) N() int { return 1 }
