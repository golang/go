// compile

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	f[int]()
}

func f[T1 any]() {
	var x Outer[T1, int]
	x.M()
}

type Outer[T1, T2 any] struct{ Inner[T2] }

type Inner[_ any] int

func (Inner[_]) M() {}
