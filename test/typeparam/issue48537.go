// compile -G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
}

type C interface {
	map[int]string
}

func f[A C]() A {
	return A{
		1: "a",
		2: "b",
	}
}
