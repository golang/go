// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"unsafe"
)

type S[T any] struct {
	val T
}

// Test type substitution where base type is unsafe.Pointer
type U[T any] unsafe.Pointer

func test[T any]() T {
	var q U[T]
	var v struct {
		// Test derived type that contains an unsafe.Pointer
		p   unsafe.Pointer
		val T
	}
	_ = q
	return v.val
}

func main() {
	want := 0
	got := test[int]()
	if got != want {
		panic(fmt.Sprintf("got %f, want %f", got, want))
	}

}
