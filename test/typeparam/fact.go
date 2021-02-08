// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
)


func fact[T interface { type float64 }](n T) T {
	if n == T(1) {
		return T(1)
	}
	return n * fact(n - T(1))
}

func main() {
	got := fact(4.0)
	want := 24.0
	if got != want {
		panic(fmt.Sprintf("Got %f, want %f", got, want))
	}

	// Re-enable when types2 bug is fixed (can't do T(1) with more than one
	// type in the type list).
	//got = fact(5)
	//want = 120
	//if want != got {
	//	panic(fmt.Sprintf("Want %d, got %d", want, got))
	//}
}
