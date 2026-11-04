// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package foo_test

import "fmt"

func square(i int) int {
	// Multiply i by itself.
	return i * i
}

type T struct {
	// X is a field.
	X int // trailing
}

var (
	a = 1
	// b is not used by the example.
	b = 2
)

func ExampleA() {
	fmt.Println(square(a), T{})
}

func ExampleB() {
}
