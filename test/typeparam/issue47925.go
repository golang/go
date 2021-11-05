// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type myifacer[T any] interface{ do(T) error }

type stuff[T any] struct{}

func (s stuff[T]) run() interface{} {
	var i myifacer[T]
	return i
}

func main() {
	stuff[int]{}.run()
}
