// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
)

type Numeric interface {
	int32 | int64 | float64 | complex64
}

//go:noline
func inc[T Numeric](x T) T {
	x++
	return x
}
func main() {
	if got, want := inc(int32(5)), int32(6); got != want {
		panic(fmt.Sprintf("got %d, want %d", got, want))
	}
	if got, want := inc(float64(5)), float64(6.0); got != want {
		panic(fmt.Sprintf("got %d, want %d", got, want))
	}
	if got, want := inc(complex64(5)), complex64(6.0); got != want {
		panic(fmt.Sprintf("got %d, want %d", got, want))
	}
}
