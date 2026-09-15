// errorcheck -0 -d=ssa/mem2reg/debug=4

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test for variables accessed through pointers that can
// be promoted to register by the mem2reg pass.

package main

import "unsafe"

// The following functions are ported from src/cmd/compile/internal/ssa/mem2reg_test.go.

// Single-block mem2reg test

func justReturn0() int {
	var x, y, z int // ERROR "promoted x in single block case" "promoted y in single block case" "promoted z in single block case"
	xp := &x
	yp := &y
	zp := &z
	*xp = 0
	*yp = 1
	*zp = 2
	*yp = 3
	*zp = 4
	*yp = 5
	valX1 := *xp
	*zp = valX1 - 1
	*xp = 1
	*yp = 2
	*xp = 3
	*yp = 4
	*xp = 5
	valZ1 := *zp
	*yp = valZ1 + 1
	*zp = 1
	*xp = 2
	*zp = 3
	*xp = 4
	*zp = 5
	valY1 := *yp
	return valY1
}

// Control flow: if/else
//
//go:noinline
func ifElseAddr(c bool) int {
	x := 1 // ERROR "promoted x in cross block case"
	p := &x
	if c {
		*p = 2
	} else {
		*p = 3
	}
	return *p // ERROR "eliminated phi"
}

// Loop
//
//go:noinline
func loopAddr() int {
	sum := 0 // ERROR "promoted sum in cross block case"
	p := &sum
	for i := 0; i < 10; i++ { // ERROR "eliminated phi"
		*p += i
	}
	return *p
}

// Nested Loop
//
//go:noinline
func nestedLoopAddr() int {
	sum := 0 // ERROR "promoted sum in cross block case"
	p := &sum
	for i := 0; i < 10; i++ { // ERROR "eliminated phi"
		for j := 0; j < 10; j++ { // ERROR "eliminated phi"
			*p += i + j
		}
	}
	return *p
}

// Switch Case
//
//go:noinline
func switchCaseAddr(x int) int {
	res := 0 // ERROR "promoted res in cross block case"
	p := &res
	switch x {
	case 0:
		*p = 10
	case 1:
		*p = 20
	default:
		*p = 30
	}
	return *p // ERROR "eliminated phi"
}

// Goto
//
//go:noinline
func gotoAddr(n int) int {
	sum := 0 // ERROR "promoted sum in cross block case"
	p := &sum
	i := 0
loop:
	if i >= n { // ERROR "eliminated phi"
		goto end
	}
	*p += i
	i++
	goto loop
end:
	return *p
}

// Complex: merges loops, switch, if-else, and closures.
//
//go:noinline
func complexAddr(n int) int {
	res := 0 // ERROR "promoted res in cross block case"
	p := &res

	// Closure that is inlined.
	inc := func(val int) {
		*p += val
	}

	for i := 0; i < n; i++ { // ERROR "eliminated phi"
		// Switch with control flow
		switch i % 3 {
		case 0:
			*p += i
		case 1:
			if i > 5 { // ERROR "eliminated phi"
				*p -= i
			} else {
				inc(i) // Call closure
			}
		default:
			// Nested loop
			for j := 0; j < 2; j++ { // ERROR "eliminated phi"
				*p += j
			}
		}
	}
	return *p
}

// A store of a different type through the variable's address (an unsafe
// reinterpretation) must not be merged into a phi of the variable's type:
// x is not promoted, so no diagnostic is expected on it.
func unsafeTypedStoreAddr(c bool) int64 {
	var x int64 = 7
	if c {
		*(*float64)(unsafe.Pointer(&x)) = 1.5
	}
	return x
}
