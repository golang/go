// compile

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type Interface[T any] interface {
}

func F[T any]() Interface[T] {
	var i int
	return i
}

func main() {
	F[int]()
}
