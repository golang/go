// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacompile_test

import (
	"encoding/binary"
	"testing"
	"unsafe"
)

type point struct {
	x, y int
}

//go:noinline
func simpleStructAddr(c bool) int {
	p := point{10, 20}
	ptr := &p
	if c {
		ptr.x += 1
	} else {
		ptr.y += 1
	}
	return ptr.x + ptr.y
}

//go:noinline
func simpleStructReg(c bool) int {
	x, y := 10, 20
	if c {
		x += 1
	} else {
		y += 1
	}
	return x + y
}

type inner struct {
	a, b int
}

type outer struct {
	in inner
	c  int
}

//go:noinline
func nestedStructAddr(c bool) int {
	o := outer{inner{1, 2}, 3}
	ptr := &o
	if c {
		ptr.in.a += 10
	} else {
		ptr.in.b += 20
	}
	return ptr.in.a + ptr.in.b + ptr.c
}

//go:noinline
func nestedStructReg(c bool) int {
	a, b, c_val := 1, 2, 3
	if c {
		a += 10
	} else {
		b += 20
	}
	return a + b + c_val
}

type mixed struct {
	a int32
	b float64
}

//go:noinline
func mixedStructAddr(c bool) float64 {
	m := mixed{1, 2.0}
	ptr := &m
	if c {
		ptr.a = 10
	} else {
		ptr.b = 20.0
	}
	return float64(ptr.a) + ptr.b
}

//go:noinline
func mixedStructReg(c bool) float64 {
	a := int32(1)
	b := 2.0
	if c {
		a = 10
	} else {
		b = 20.0
	}
	return float64(a) + b
}

//go:noinline
func arrayAddr(c bool) int {
	var a [3]int
	a[0] = 10
	a[1] = 20
	a[2] = 30
	if c {
		a[1] += 5
	}
	return a[0] + a[1] + a[2]
}

//go:noinline
func arrayReg(c bool) int {
	a0, a1, a2 := 10, 20, 30
	if c {
		a1 += 5
	}
	return a0 + a1 + a2
}

//go:noinline
func zeroStructAddr(c bool) int {
	var p point
	ptr := &p
	if c {
		ptr.x = 10
	} else {
		ptr.y = 20
	}
	return ptr.x + ptr.y
}

//go:noinline
func zeroStructReg(c bool) int {
	var x, y int
	if c {
		x = 10
	} else {
		y = 20
	}
	return x + y
}

//go:noinline
func moveStructAddr(c bool) int {
	p1 := point{10, 20}
	var p2 point
	if c {
		p2 = p1
	} else {
		p2 = point{30, 40}
	}
	return p2.x + p2.y
}

//go:noinline
func moveStructReg(c bool) int {
	p1x, p1y := 10, 20
	var p2x, p2y int
	if c {
		p2x, p2y = p1x, p1y
	} else {
		p2x, p2y = 30, 40
	}
	return p2x + p2y
}

//go:noinline
func shadowedStructAddr(c bool) int {
	p := point{10, 20}
	if c {
		p := point{30, 40}
		return p.x + p.y
	}
	return p.x + p.y
}

//go:noinline
func shadowedStructReg(c bool) int {
	px, py := 10, 20
	if c {
		px2, py2 := 30, 40
		return px2 + py2
	}
	return px + py
}

type big struct {
	a [100]int
}

//go:noinline
func tooBigStructAddr() int {
	var b big
	b.a[0] = 10
	return b.a[0]
}

//go:noinline
func tooBigStructReg() int {
	var a0 int = 10
	return a0
}

type arrayStruct struct {
	a [2]int
	b int
}

//go:noinline
func arrayStructAddr(c bool) int {
	s := arrayStruct{[2]int{1, 2}, 3}
	if c {
		s.a[0] += 10
	} else {
		s.b += 20
	}
	return s.a[0] + s.a[1] + s.b
}

//go:noinline
func arrayStructReg(c bool) int {
	a0, a1, b := 1, 2, 3
	if c {
		a0 += 10
	} else {
		b += 20
	}
	return a0 + a1 + b
}

type stringStruct struct {
	s string
	i int
}

//go:noinline
func stringStructAddr(c bool) int {
	// string has non-scalar fields (ptr + len), so it won't be decomposed by my current implementation
	// which only allows true scalars.
	ss := stringStruct{"hello", 10}
	if c {
		ss.i += 5
	}
	return len(ss.s) + ss.i
}

//go:noinline
func stringStructReg(c bool) int {
	s, i := "hello", 10
	if c {
		i += 5
	}
	return len(s) + i
}

//go:noinline
func partialZeroAddr(bool) int {
	type T struct {
		a int
		b [2]int
		c int
	}

	t := T{a: 9, c: 9} // ERROR "promoted t"
	p := &t.b          // OffPtr
	*p = [2]int{}      // Zero
	return t.a * t.c
}

//go:noinline
func partialMoveAddr(bool) int {
	type T struct {
		a int
		b [2]int
		c int
	}

	b2 := [2]int{1, 2} // ERROR "promoted b2"

	t := T{a: 9, c: 9} // ERROR "promoted t"
	p := &t.b          // OffPtr
	*p = b2            // Move
	return t.a * t.c
}

//go:noinline
func unsafeSelfMoveAddr(c bool) int {
	var s struct {
		a [2]uint16
		b uint32
	}
	s.a[0], s.a[1] = 1, 2
	*(*[4]byte)(unsafe.Pointer(&s.b)) = *(*[4]byte)(unsafe.Pointer(&s.a))
	if c {
		s.b++
	}
	return int(s.b)
}

func unsafeAliasMoveAddr(c bool) int {
	// A Move between two differently-shaped candidates via an unsafe
	// cast: breaking the Move down by one side's leaves creates
	// accesses that straddle the other side's leaves, so only one of
	// the two may be decomposed (see the Move partner disqualification
	// in decomposeAddr).
	var x [4]byte
	x[0], x[1], x[2], x[3] = 1, 2, 3, 4
	var y [2]uint16
	y = *(*[2]uint16)(unsafe.Pointer(&x))
	if c {
		y[0]++
	}
	return int(y[0]) + int(y[1])
}

//go:noinline
func unsafeAliasMoveReg(c bool) int {
	// x's bytes reinterpreted as [2]uint16, in the target's byte order.
	v0 := binary.NativeEndian.Uint16([]byte{1, 2})
	v1 := binary.NativeEndian.Uint16([]byte{3, 4})
	if c {
		v0++
	}
	return int(v0) + int(v1)
}

func TestDecomposeAddr(t *testing.T) {
	tests := []struct {
		name string
		f1   func(bool) int
		f2   func(bool) int
	}{
		{"simpleStruct", simpleStructAddr, simpleStructReg},
		{"nestedStruct", nestedStructAddr, nestedStructReg},
		{"array", arrayAddr, arrayReg},
		{"zeroStruct", zeroStructAddr, zeroStructReg},
		{"moveStruct", moveStructAddr, moveStructReg},
		{"shadowedStruct", shadowedStructAddr, shadowedStructReg},
		{"arrayStruct", arrayStructAddr, arrayStructReg},
		{"stringStruct", stringStructAddr, stringStructReg},
		{"unsafeAliasMove", unsafeAliasMoveAddr, unsafeAliasMoveReg},
		{"unsafeSelfMove", unsafeSelfMoveAddr, func(c bool) int {
			// s.a's bytes reinterpreted as uint32, in the target's byte order.
			b := binary.NativeEndian.Uint32(binary.NativeEndian.AppendUint16(binary.NativeEndian.AppendUint16(nil, 1), 2))
			if c {
				return int(b + 1)
			}
			return int(b)
		}},
		{"partialZero", partialZeroAddr, func(bool) int { return 81 }},
		{"partialMove", partialMoveAddr, func(bool) int { return 81 }},
	}

	for _, tc := range tests {
		for _, c := range []bool{true, false} {
			if got, want := tc.f1(c), tc.f2(c); got != want {
				t.Errorf("%s(%v): got %d, want %d", tc.name, c, got, want)
			}
		}
	}

	if got, want := mixedStructAddr(true), mixedStructReg(true); got != want {
		t.Errorf("mixedStruct(true): got %f, want %f", got, want)
	}
	if got, want := mixedStructAddr(false), mixedStructReg(false); got != want {
		t.Errorf("mixedStruct(false): got %f, want %f", got, want)
	}

	if got, want := tooBigStructAddr(), tooBigStructReg(); got != want {
		t.Errorf("tooBigStruct: got %d, want %d", got, want)
	}
}
