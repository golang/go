// run

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	F[T[int]]()
}

func F[X interface{ M() }]() {
	var x X
	x.M()
}

type T[X any] struct{ E }

type E struct{}

func (h E) M() {}
