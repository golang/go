// run

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that devirtualization doesn't introduce spurious type
// assertion failures due to shaped and non-shaped interfaces having
// distinct itabs.

package main

func main() {
	F[int]()
}

func F[T any]() {
	var i I[T] = X(0)
	i.M()
}

type I[T any] interface{ M() }

type X int

func (X) M() {}
