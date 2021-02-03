// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
)


func min[T interface{ type int }](x, y T) T {
	if x < y {
		return x
	}
	return y
}

func main() {
	want := 2
	got := min[int](2, 3)
	if want != got {
		panic(fmt.Sprintf("Want %d, got %d", want, got))
	}

	got = min(2, 3)
	if want != got {
		panic(fmt.Sprintf("Want %d, got %d", want, got))
	}
}
