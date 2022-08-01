// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import (
	"fmt"
	"runtime"
)

var y int

//go:noinline
func Square(x int) {
	var pc0, pc1 [1]uintptr
	runtime.Callers(1, pc0[:]) // get PC at entry

	// a switch using jump table
	switch x {
	case 1:
		y = 1
	case 2:
		y = 4
	case 3:
		y = 9
	case 4:
		y = 16
	case 5:
		y = 25
	case 6:
		y = 36
	case 7:
		y = 49
	case 8:
		y = 64
	default:
		panic("too large")
	}

	// check PC is in the same function
	runtime.Callers(1, pc1[:])
	if pc1[0] < pc0[0] || pc1[0] > pc0[0]+1000000 {
		fmt.Printf("jump across DSO boundary. pc0=%x, pc1=%x\n", pc0[0], pc1[0])
		panic("FAIL")
	}

	if y != x*x {
		fmt.Printf("x=%d y=%d!=%d\n", x, y, x*x)
		panic("FAIL")
	}
}
