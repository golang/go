// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
)

//go:noinline
func shifted(x int64, y uint64) int64 {
	return x >> uint32(y)
}

func main() {
	x := int64(-0x4000000000000000)
	got := shifted(x, ^uint64(0)&^(1<<32-2))
	want := x >> 1
	if got != want {
		panic(fmt.Sprintf("want %d; got %d", want, got))
	}
}
