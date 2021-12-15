// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"constraints"
	"fmt"
)

func zero[T constraints.Complex]() T {
	return T(0)
}
func pi[T constraints.Complex]() T {
	return T(3.14)
}
func sqrtN1[T constraints.Complex]() T {
	return T(-1i)
}

func main() {
	fmt.Println(zero[complex128]())
	fmt.Println(pi[complex128]())
	fmt.Println(sqrtN1[complex128]())
	fmt.Println(zero[complex64]())
	fmt.Println(pi[complex64]())
	fmt.Println(sqrtN1[complex64]())
}

