// run

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func callRecover() {
	if recover() != nil {
		println("recovered")
	}
}

type T int

func (*T) M() { callRecover() }

type S struct{ *T } // has a wrapper S.M wrapping (*T.M)

var p = S{new(T)}

var fn = S.M // using a function pointer to force using the wrapper

func main() {
	mustPanic(func() {
		defer fn(p)
		panic("XXX")
	})
}

func mustPanic(f func()) {
	defer func() {
		r := recover()
		if r == nil {
			panic("didn't panic")
		}
	}()
	f()
}
