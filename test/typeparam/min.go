// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
)

type Ordered interface {
	~int | ~int64 | ~float64 | ~string
}

func min[T Ordered](x, y T) T {
	if x < y {
		return x
	}
	return y
}

func main() {
	const want = 2
	if got := min[int](2, 3); got != want {
		panic(fmt.Sprintf("got %d, want %d", got, want))
	}

	if got := min(2, 3); got != want {
		panic(fmt.Sprintf("want %d, got %d", want, got))
	}

	if got := min[float64](3.5, 2.0); got != want {
		panic(fmt.Sprintf("got %d, want %d", got, want))
	}

	if got := min(3.5, 2.0); got != want {
		panic(fmt.Sprintf("got %d, want %d", got, want))
	}

	const want2 = "ay"
	if got := min[string]("bb", "ay"); got != want2 {
		panic(fmt.Sprintf("got %d, want %d", got, want2))
	}

	if got := min("bb", "ay"); got != want2 {
		panic(fmt.Sprintf("got %d, want %d", got, want2))
	}
}
