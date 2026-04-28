// errorcheck -0 -d=ssa/known_bits/debug=1

//go:build amd64 || arm64 || s390x || ppc64le || riscv64

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

func knownBitsPhiAnd(cond bool) int {
	x := 1
	if cond {
		x = 3
	}
	return x & 1 // ERROR "known value of v[0-9]+ \(And64\): 1$"
}

func knownBitsPhiAndGarbage(cond bool, x int) int {
	x &^= 1
	if cond {
		x = 2
	}
	return x & 1 // ERROR "known value of v[0-9]+ \(And64\): 0$"
}

func unknownBitsPhiAnd(cond bool) int {
	x := 1
	if cond {
		x = 2
	}
	return x & 1
}

func knownBitsOrGarbage(x, unknown int) int {
	x |= 7
	x |= unknown &^ 3
	return x & 3 // ERROR "known value of v[0-9]+ \(And64\): 3$"
}

func unknownBitsOrGarbage(x, unknown int) int {
	x |= 1
	x |= unknown
	return x & 3
}

func knownBitsDeferPattern(a, b bool) int {
	bits := 0
	bits |= 1 << 0
	if a {
		bits |= 1 << 1
	}
	bits |= 1 << 2
	if b {
		bits |= 1 << 3
	}
	return bits & (1<<2 | 1<<0) // ERROR "known value of v[0-9]+ \(And64\): 5$"
}

func knownBitsDeferPatternGarbage(a, b bool, garbage int) int {
	bits := 0
	bits |= 1 << 0
	if a {
		bits |= 1 << 1
	}
	bits |= 1 << 2
	if b {
		bits |= 1 << 3
	}
	bits ^= garbage &^ (1<<2 | 1<<0)
	return bits & (1<<2 | 1<<0) // ERROR "known value of v[0-9]+ \(And64\): 5$"
}

func knownBitsXorToggle(a, b, c bool) int {
	bits := 0
	bits ^= 1 << 0
	if a {
		bits ^= 1 << 1
	}
	bits ^= 1 << 2
	if b {
		bits ^= 1 << 3
	}
	bits ^= 1 << 2
	if c {
		bits ^= 1 << 4
	}
	return bits & (1<<2 | 1<<0) // ERROR "known value of v[0-9]+ \(And64\): 1$"
}

func knownBitsXorToggleGarbage(a, b, c bool, garbage int) int {
	bits := 0
	bits ^= 1 << 0
	if a {
		bits ^= 1 << 1
	}
	bits ^= 1 << 2
	if b {
		bits ^= 1 << 3
	}
	bits ^= 1 << 2
	if c {
		bits ^= 1 << 4
	}
	bits ^= garbage &^ (1<<2 | 1<<0)
	return bits & (1<<2 | 1<<0) // ERROR "known value of v[0-9]+ \(And64\): 1$"
}

func unknownBitsXorToggle(a, b, c bool) int {
	bits := 0
	bits ^= 1 << 0
	if a {
		bits ^= 1 << 1
	}
	bits ^= 1 << 2
	if b {
		bits ^= 1 << 2
	}
	bits ^= 1 << 2
	if c {
		bits ^= 1 << 4
	}
	return bits & (1<<2 | 1<<0)
}

func knownBitsPhiComAnd(cond bool) int {
	x := 1
	if cond {
		x = 3
	}
	return ^x & 1 // ERROR "known value of v[0-9]+ \(And64\): 0$"
}

func knownBitsPhiComAndGarbage(cond bool, garbage int) int {
	x := 1
	if cond {
		x = 3
	}
	x ^= garbage &^ 1
	return ^x & 1 // ERROR "known value of v[0-9]+ \(And64\): 0$"
}

func unknownBitsPhiComAnd(cond bool) int {
	x := 1
	if cond {
		x = 2
	}
	return ^x & 1
}

func knownBitsEqFalse(x, y uint64) bool {
	x |= 1
	y &^= 1
	return x == y // ERROR "known value of v[0-9]+ \(Eq64\): false$"
}

func knownBitsEqTrue(x uint64, cond bool) bool {
	x |= (1<<32 - 1) << 32
	if cond {
		x |= 42
	}
	x |= 1<<32 - 1      // ERROR "known value of v[0-9]+ \(Or64\): -1$"
	return x == 1<<64-1 // ERROR "known value of v[0-9]+ \(Eq64\): true$"
}

func unknownBitsEq(x, y uint64) bool {
	x |= 1
	return x == y
}

func knownBitsNeqTrue(x, y uint64) bool {
	x |= 1
	y &^= 1
	return x != y // ERROR "known value of v[0-9]+ \(Neq64\): true$"
}

func knownBitsNeqFalse(x uint64, cond bool) bool {
	x |= (1<<32 - 1) << 32
	if cond {
		x |= 42
	}
	x |= 1<<32 - 1
	return x != 1<<64-1 // ERROR "known value of v[0-9]+ \(Neq64\): false$"
}

func unknownBitsNeq(x, y uint64) bool {
	x |= 1
	return x != y
}

func knownBitsZeroExtPassThrough(x uint8) uint64 {
	x |= 6
	return uint64(x) & 6 // ERROR "known value of v[0-9]+ \(And64\): 6$"
}

func knownBitsZeroExtUpperHalf(x uint16) uint32 {
	return uint32(x) & 0xFFFF0000 // ERROR "known value of v[0-9]+ \(And32\): 0$"
}

func unknownBitsZeroExt(x uint16) uint32 {
	x |= 0xAAAA
	return uint32(x) & 0xFFFFF000
}

func cvtBoolToUint8(cond bool) (r uint8) {
	if cond {
		r = 1
	}
	return r
}

func knownBitsCvtBoolToUint8False(x, y uint64) uint8 {
	x |= 1
	y &^= 1
	bool := x == y                  // ERROR "known value of v[0-9]+ \(Eq64\): false$"
	return cvtBoolToUint8(bool) & 1 // ERROR "known value of v[0-9]+ \(CvtBoolToUint8\): 0$"
}

func knownBitsCvtBoolToUint8True(x int64, cond bool) uint8 {
	x |= 6
	if cond {
		x |= 1
		x |= 4
	}
	// I would expect "known value of v[0-9]+ \(And64\): 6$" to be required, but somehow it's not there even tho the AND is being folded.
	// I think it's an issue with the And's LOC meaning known bits prints it without a LOC and errorcheck skips it.
	r := cvtBoolToUint8(x&6 == 6) // ERROR "known value of v[0-9]+ \(Eq64\): true$" "known value of v[0-9]+ \(CvtBoolToUint8\): 1$"
	if cond {
		r |= 4 // ERROR "known value of v[0-9]+ \(Or8\): 5$"
	}
	return r & 3 // ERROR "known value of v[0-9]+ \(And8\): 1$"
}

func unknownBitsCvtBoolToUint8(cond bool) uint8 {
	return cvtBoolToUint8(cond) & 1
}

func knownBitsSignExtPassThrough(x int8) int64 {
	x |= 6
	return int64(x) & 6 // ERROR "known value of v[0-9]+ \(And64\): 6$"
}

func knownBitsSignExtUpperHalf(x int16) int32 {
	x |= -1 << 15
	return int32(x) & (-1 << 16) // ERROR "known value of v[0-9]+ \(And32\): -65536$"
}

func unknownBitsSignExt(x int16) int32 {
	x |= -0b010101010101010
	return int32(x) & -1 << 12
}

func knownBitsLsh(x, y uint32) uint32 {
	x |= 0b11110
	x &^= 0b100000
	y &= 2

	// ???01111?
	// ?01111???
	// ---------
	// ????11???

	return (x << y) & 0b11000 // ERROR "known value of v[0-9]+ \(And32\): 24$"
}

func knownBitsLshZero(x, y uint64) uint64 {
	x &^= 2
	y &^= 2
	y |= 128

	return (x << y) & 8 // ERROR "known value of v[0-9]+ \(And64\): 0$" "known value of v[0-9]+ \(Lsh64x[0-9]+\): 0$"
}

func unknownBitsLshLeftSideMsb(x uint32, y uint32) uint32 {
	x |= 0b11110
	x &^= 0b100000
	y &= 2

	return (x << y) & 0b111000
}

func unknownBitsLshLeftSideLsb(x uint32, y uint32) uint32 {
	x |= 0b11110
	x &^= 0b100000
	y &= 2

	return (x << y) & 0b011100
}

func unknownBitsLshRightSide(x uint32, y uint32) uint32 {
	x |= 0b11110
	x &^= 0b100000
	y &= 6

	return (x << y) & 0b11000
}

func knownBitsRshU(x, y uint32) uint32 {
	x |= 0b11110
	x &^= 0b100000
	y &= 2

	// ?01111?
	// ???0111
	// -------
	// ????11?

	return (x >> y) & 0b110 // ERROR "known value of v[0-9]+ \(And32\): 6$"
}

func knownBitsRshUZero(x, y uint64) uint64 {
	x &^= 4
	y &^= 2
	y |= 128

	return (x >> y) & 1 // ERROR "known value of v[0-9]+ \(And64\): 0$" "known value of v[0-9]+ \(Rsh64Ux[0-9]+\): 0$"
}

func unknownBitsRshULeftSideMsb(x uint32, y uint32) uint32 {
	x |= 0b11110
	x &^= 0b100000
	y &= 2

	return (x >> y) & 0b1110
}

func unknownBitsRshULeftSideLsb(x uint32, y uint32) uint32 {
	x |= 0b11110
	x &^= 0b100000
	y &= 2

	return (x >> y) & 0b111
}

func unknownBitsRshURightSide(x uint32, y uint32) uint32 {
	x |= 0b11110
	x &^= 0b100000
	y &= 6

	return (x >> y) & 0b110
}

func knownBitsRsh(x, y int32) int32 {
	x |= 0b11110
	x &^= 0b100000
	y &= 2

	// ?01111?
	// ???0111
	// -------
	// ????11?

	return (x >> y) & 0b110 // ERROR "known value of v[0-9]+ \(And32\): 6$"
}

func knownBitsRshSignCopy(x, y int64) int64 {
	x |= -1 << 63
	y |= 128

	return (x >> y) & 1 // ERROR "known value of v[0-9]+ \(And64\): 1$"
}

func unknownBitsRshLeftSideMsb(x int32, y int32) int32 {
	x |= 0b11110
	x &^= 0b100000
	y &= 2

	return (x >> y) & 0b1110
}

func unknownBitsRshLeftSideLsb(x int32, y int32) int32 {
	x |= 0b11110
	x &^= 0b100000
	y &= 2

	return (x >> y) & 0b111
}

func unknownBitsRshRightSide(x int32, y int32) int32 {
	x |= 0b11110
	x &^= 0b100000
	y &= 6

	return (x >> y) & 0b110
}

func knownBitsTrunc(x int64, cond bool) int32 {
	x |= 2
	if cond {
		x = 3
	}

	return int32(x) & 2 // ERROR "known value of v[0-9]+ \(And32\): 2$"
}

func unknownBitsTrunc(x int64, cond bool) int32 {
	x |= 2
	if cond {
		x = 3
	}

	return int32(x) & 1
}

func knownBitsSextAfterTrunc(x int64, cond1, cond2 bool) int64 {
	x |= -1 << 31
	if cond1 {
		x = -3
	}

	truncated := int32(x)

	if cond2 {
		truncated |= 4
	}

	return int64(truncated) & (-1 << 63) // ERROR "known value of v[0-9]+ \(And64\): -9223372036854775808$"
}

func unknownBitsSextAfterTrunc(x int64, cond1, cond2 bool) int64 {
	x |= 2
	if cond1 {
		x = 3
	}

	truncated := int32(x)

	if cond2 {
		truncated |= 4
	}

	return int64(truncated) & (-1 << 63)
}
