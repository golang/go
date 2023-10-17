// run

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Do not panic on conversion to anonymous interface, which
// is similar-looking interface types in different packages.

package main

import (
	"fmt"

	ssa1 "issue29612.dir/p1/ssa"
	ssa2 "issue29612.dir/p2/ssa"
)

func main() {
	v1 := &ssa1.T{}
	_ = v1

	v2 := &ssa2.T{}
	ssa2.Works(v2)
	ssa2.Panics(v2) // This call must not panic

	swt(v1, 1)
	swt(v2, 2)
}

//go:noinline
func swt(i interface{}, want int) {
	var got int
	switch i.(type) {
	case *ssa1.T:
		got = 1
	case *ssa2.T:
		got = 2

	case int8, int16, int32, int64:
		got = 3
	case uint8, uint16, uint32, uint64:
		got = 4
	}

	if got != want {
		panic(fmt.Sprintf("switch %v: got %d, want %d", i, got, want))
	}
}
