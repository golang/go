// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

func fact[T interface{ ~int | ~int64 | ~float64 }](n T) T {
	if n == 1 {
		return 1
	}
	return n * fact(n-1)
}

func main() {
	const want = 120

	if got := fact(5); got != want {
		panic(fmt.Sprintf("got %d, want %d", got, want))
	}

	if got := fact[int64](5); got != want {
		panic(fmt.Sprintf("got %d, want %d", got, want))
	}

	if got := fact(5.0); got != want {
		panic(fmt.Sprintf("got %f, want %f", got, want))
	}
}
