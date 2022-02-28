// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

func main() {
	IsZero[int](0)
}

func IsZero[T comparable](val T) bool {
	var zero T
	fmt.Printf("%v:%v\n", zero, val)
	return val != zero
}
