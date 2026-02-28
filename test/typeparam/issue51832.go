// compile

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type F func() F

func do[T any]() F {
	return nil
}

type G[T any] func() G[T]

//go:noinline
func dog[T any]() G[T] {
	return nil
}

func main() {
	do[int]()
	dog[int]()
}
