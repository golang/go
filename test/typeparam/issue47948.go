// compile -G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type fun func()

func F[T any]() {
	_ = fun(func() {

	})
}
func main() {
	F[int]()
}
