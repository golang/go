// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacompile

import (
	"testing"
	"unsafe"
)

// Single-block mem2reg test
//
//go:noinline
func justReturn0() int {
	var x, y, z int
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
	x := 1
	p := &x
	if c {
		*p = 2
	} else {
		*p = 3
	}
	return *p
}

//go:noinline
func ifElseReg(c bool) int {
	if c {
		return 2
	}
	return 3
}

// Loop
//
//go:noinline
func loopAddr() int {
	sum := 0
	p := &sum
	for i := 0; i < 10; i++ {
		*p += i
	}
	return *p
}

//go:noinline
func loopReg() int {
	sum := 0
	for i := 0; i < 10; i++ {
		sum += i
	}
	return sum
}

// Nested Loop
//
//go:noinline
func nestedLoopAddr() int {
	sum := 0
	p := &sum
	for i := 0; i < 10; i++ {
		for j := 0; j < 10; j++ {
			*p += i + j
		}
	}
	return *p
}

//go:noinline
func nestedLoopReg() int {
	sum := 0
	for i := 0; i < 10; i++ {
		for j := 0; j < 10; j++ {
			sum += i + j
		}
	}
	return sum
}

// Switch Case
//
//go:noinline
func switchCaseAddr(x int) int {
	res := 0
	p := &res
	switch x {
	case 0:
		*p = 10
	case 1:
		*p = 20
	default:
		*p = 30
	}
	return *p
}

//go:noinline
func switchCaseReg(x int) int {
	switch x {
	case 0:
		return 10
	case 1:
		return 20
	default:
		return 30
	}
}

// Goto
//
//go:noinline
func gotoAddr(n int) int {
	sum := 0
	p := &sum
	i := 0
loop:
	if i >= n {
		goto end
	}
	*p += i
	i++
	goto loop
end:
	return *p
}

//go:noinline
func gotoReg(n int) int {
	sum := 0
	i := 0
loop:
	if i >= n {
		goto end
	}
	sum += i
	i++
	goto loop
end:
	return sum
}

// Complex: merges loops, switch, if-else, and closures.
//
//go:noinline
func complexAddr(n int) int {
	res := 0
	p := &res

	// Closure that might be inlined
	inc := func(val int) {
		*p += val
	}

	for i := 0; i < n; i++ {
		// Switch with control flow
		switch i % 3 {
		case 0:
			*p += i
		case 1:
			if i > 5 {
				*p -= i
			} else {
				inc(i) // Call closure
			}
		default:
			// Nested loop
			for j := 0; j < 2; j++ {
				*p += j
			}
		}
	}
	return *p
}

//go:noinline
func complexReg(n int) int {
	res := 0

	inc := func(val int) {
		res += val
	}

	for i := 0; i < n; i++ {
		switch i % 3 {
		case 0:
			res += i
		case 1:
			if i > 5 {
				res -= i
			} else {
				inc(i)
			}
		default:
			for j := 0; j < 2; j++ {
				res += j
			}
		}
	}
	return res
}

//go:noinline
func unsafeTypedStoreAddr(c bool) int64 {
	// The float64-typed store through x's address must not flow into
	// an int64 phi for x; mem2reg has to leave x in memory (or
	// reinterpret), on every architecture.
	var x int64 = 7
	if c {
		*(*float64)(unsafe.Pointer(&x)) = 1.5
	}
	return x
}

//go:noinline
func unsafeTypedStoreReg(c bool) int64 {
	x := int64(7)
	if c {
		x = int64(0x3ff8000000000000) // bits of float64(1.5)
	}
	return x
}

func TestMem2Reg(t *testing.T) {
	if r := justReturn0(); r != 0 {
		t.Errorf("justReturn0() = %d, want 0", r)
	}
	if got, want := ifElseAddr(true), ifElseReg(true); got != want {
		t.Errorf("ifElse(true): got %d, want %d", got, want)
	}
	if got, want := ifElseAddr(false), ifElseReg(false); got != want {
		t.Errorf("ifElse(false): got %d, want %d", got, want)
	}

	if got, want := loopAddr(), loopReg(); got != want {
		t.Errorf("loop: got %d, want %d", got, want)
	}
	if got, want := nestedLoopAddr(), nestedLoopReg(); got != want {
		t.Errorf("nestedLoop: got %d, want %d", got, want)
	}
	for _, x := range []int{0, 1, 2} {
		if got, want := switchCaseAddr(x), switchCaseReg(x); got != want {
			t.Errorf("switchCase(%d): got %d, want %d", x, got, want)
		}
	}
	if got, want := gotoAddr(10), gotoReg(10); got != want {
		t.Errorf("goto: got %d, want %d", got, want)
	}
	if got, want := complexAddr(20), complexReg(20); got != want {
		t.Errorf("complex: got %d, want %d", got, want)
	}
	for _, c := range []bool{true, false} {
		if got, want := unsafeTypedStoreAddr(c), unsafeTypedStoreReg(c); got != want {
			t.Errorf("unsafeTypedStore(%v): got %#x, want %#x", c, got, want)
		}
	}
}
