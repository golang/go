// run

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type I interface {
	m() int
}

type T struct{}

func (T) m() int {
	return 3
}

var t T

var ret = I.m(t)

func main() {
	if ret != 3 {
		println("ret = ", ret)
		panic("ret != 3")
	}
}

