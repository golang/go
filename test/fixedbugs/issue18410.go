// run

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This checks partially initialized structure literals
// used to create value.method functions have their
// non-initialized fields properly zeroed/nil'd

package main

type X struct {
	A, B, C *int
}

//go:noinline
func (t X) Print() {
	if t.B != nil {
		panic("t.B must be nil")
	}
}

//go:noinline
func caller(f func()) {
	f()
}

//go:noinline
func test() {
	var i, j int
	x := X{A: &i, C: &j}
	caller(func() { X{A: &i, C: &j}.Print() })
	caller(X{A: &i, C: &j}.Print)
	caller(x.Print)
}

func main() {
	test()
}
