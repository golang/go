// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package bits implements bit counting and manipulation
// functions for the predeclared unsigned integer types.
package bits

// UintSize is the size of a uint in bits.
const UintSize = uintSize

// --- LeadingZeros ---

// LeadingZeros returns the number of leading zero bits in x; the result is UintSize for x == 0.
func LeadingZeros(x uint) int { return UintSize - blen(uint64(x)) }

// LeadingZeros8 returns the number of leading zero bits in x; the result is 8 for x == 0.
func LeadingZeros8(x uint8) int { return 8 - blen(uint64(x)) }

// LeadingZeros16 returns the number of leading zero bits in x; the result is 16 for x == 0.
func LeadingZeros16(x uint16) int { return 16 - blen(uint64(x)) }

// LeadingZeros32 returns the number of leading zero bits in x; the result is 32 for x == 0.
func LeadingZeros32(x uint32) int { return 32 - blen(uint64(x)) }

// LeadingZeros64 returns the number of leading zero bits in x; the result is 64 for x == 0.
func LeadingZeros64(x uint64) int { return 64 - blen(uint64(x)) }

// --- TrailingZeros ---

// TrailingZeros returns the number of trailing zero bits in x; the result is 0 for x == 0.
func TrailingZeros(x uint) int { return ntz(x) }

// TrailingZeros8 returns the number of trailing zero bits in x; the result is 0 for x == 0.
func TrailingZeros8(x uint8) int { return ntz8(x) }

// TrailingZeros16 returns the number of trailing zero bits in x; the result is 0 for x == 0.
func TrailingZeros16(x uint16) int { return ntz16(x) }

// TrailingZeros32 returns the number of trailing zero bits in x; the result is 0 for x == 0.
func TrailingZeros32(x uint32) int { return ntz32(x) }

// TrailingZeros64 returns the number of trailing zero bits in x; the result is 0 for x == 0.
func TrailingZeros64(x uint64) int { return ntz64(x) }

// --- OnesCount ---

// OnesCount returns the number of one bits ("population count") in x.
func OnesCount(x uint) int { return pop(uint64(x)) }

// OnesCount8 returns the number of one bits ("population count") in x.
func OnesCount8(x uint8) int { return pop(uint64(x)) }

// OnesCount16 returns the number of one bits ("population count") in x.
func OnesCount16(x uint16) int { return pop(uint64(x)) }

// OnesCount32 returns the number of one bits ("population count") in x.
func OnesCount32(x uint32) int { return pop(uint64(x)) }

// OnesCount64 returns the number of one bits ("population count") in x.
func OnesCount64(x uint64) int { return pop(uint64(x)) }

// --- RotateLeft ---

// RotateLeft returns the value of x rotated left by k bits; k must not be negative.
func RotateLeft(x uint, k int) uint { return uint(rot(uint64(x), UintSize, pos(k)%UintSize)) }

// RotateLeft8 returns the value of x rotated left by k bits; k must not be negative.
func RotateLeft8(x uint8, k int) uint8 { return uint8(rot(uint64(x), 8, pos(k)%8)) }

// RotateLeft16 returns the value of x rotated left by k bits; k must not be negative.
func RotateLeft16(x uint16, k int) uint16 { return uint16(rot(uint64(x), 16, pos(k)%16)) }

// RotateLeft32 returns the value of x rotated left by k bits; k must not be negative.
func RotateLeft32(x uint32, k int) uint32 { return uint32(rot(uint64(x), 32, pos(k)%32)) }

// RotateLeft64 returns the value of x rotated left by k bits; k must not be negative.
func RotateLeft64(x uint64, k int) uint64 { return uint64(rot(uint64(x), 64, pos(k)%64)) }

// --- RotateRight ---

// RotateRight returns the value of x rotated right by k bits; k must not be negative.
func RotateRight(x uint, k int) uint { return uint(rot(uint64(x), UintSize, UintSize-pos(k)%UintSize)) }

// RotateRight8 returns the value of x rotated right by k bits; k must not be negative.
func RotateRight8(x uint8, k int) uint8 { return uint8(rot(uint64(x), 8, 8-pos(k)%8)) }

// RotateRight16 returns the value of x rotated right by k bits; k must not be negative.
func RotateRight16(x uint16, k int) uint16 { return uint16(rot(uint64(x), 16, 16-pos(k)%16)) }

// RotateRight32 returns the value of x rotated right by k bits; k must not be negative.
func RotateRight32(x uint32, k int) uint32 { return uint32(rot(uint64(x), 32, 32-pos(k)%32)) }

// RotateRight64 returns the value of x rotated right by k bits; k must not be negative.
func RotateRight64(x uint64, k int) uint64 { return uint64(rot(uint64(x), 64, 64-pos(k)%64)) }

// --- Reverse ---

const m0 = 0xaaaaaaaaaaaaaaaa // 10101010 ...
const m1 = 0xcccccccccccccccc // 11001100 ...
const m2 = 0xf0f0f0f0f0f0f0f0 // 11110000 ...
const m3 = 0xff00ff00ff00ff00 // etc.
const m4 = 0xffff0000ffff0000
const m5 = 0xffffffff00000000

// Reverse returns the value of x with its bits in reversed order.
func Reverse(x uint) uint {
	if UintSize == 32 {
		return uint(Reverse32(uint32(x)))
	}
	return uint(Reverse64(uint64(x)))
}

// Reverse8 returns the value of x with its bits in reversed order.
func Reverse8(x uint8) uint8 {
	const m = 1<<8 - 1
	x = x&(m0&m)>>1 | x&^(m0&m)<<1
	x = x&(m1&m)>>2 | x&^(m1&m)<<2
	x = x&(m2&m)>>4 | x&^(m2&m)<<4
	return x
}

// Reverse16 returns the value of x with its bits in reversed order.
func Reverse16(x uint16) uint16 {
	const m = 1<<16 - 1
	x = x&(m0&m)>>1 | x&^(m0&m)<<1
	x = x&(m1&m)>>2 | x&^(m1&m)<<2
	x = x&(m2&m)>>4 | x&^(m2&m)<<4
	x = x&(m3&m)>>8 | x&^(m3&m)<<8
	return x
}

// Reverse32 returns the value of x with its bits in reversed order.
func Reverse32(x uint32) uint32 {
	const m = 1<<32 - 1
	x = x&(m0&m)>>1 | x&^(m0&m)<<1
	x = x&(m1&m)>>2 | x&^(m1&m)<<2
	x = x&(m2&m)>>4 | x&^(m2&m)<<4
	x = x&(m3&m)>>8 | x&^(m3&m)<<8
	x = x&(m4&m)>>16 | x&^(m4&m)<<16
	return x
}

// Reverse64 returns the value of x with its bits in reversed order.
func Reverse64(x uint64) uint64 {
	const m = 1<<64 - 1
	x = x&(m0&m)>>1 | x&^(m0&m)<<1
	x = x&(m1&m)>>2 | x&^(m1&m)<<2
	x = x&(m2&m)>>4 | x&^(m2&m)<<4
	x = x&(m3&m)>>8 | x&^(m3&m)<<8
	x = x&(m4&m)>>16 | x&^(m4&m)<<16
	x = x&(m5&m)>>32 | x&^(m5&m)<<32
	return x
}

// --- ReverseBytes ---

// ReverseBytes returns the value of x with its bytes in reversed order.
func ReverseBytes(x uint) uint {
	if UintSize == 32 {
		return uint(ReverseBytes32(uint32(x)))
	}
	return uint(ReverseBytes64(uint64(x)))
}

// ReverseBytes16 returns the value of x with its bytes in reversed order.
func ReverseBytes16(x uint16) uint16 {
	const m = 1<<16 - 1
	x = x&(m3&m)>>8 | x&^(m3&m)<<8
	return x
}

// ReverseBytes32 returns the value of x with its bytes in reversed order.
func ReverseBytes32(x uint32) uint32 {
	const m = 1<<32 - 1
	x = x&(m3&m)>>8 | x&^(m3&m)<<8
	x = x&(m4&m)>>16 | x&^(m4&m)<<16
	return x
}

// ReverseBytes64 returns the value of x with its bytes in reversed order.
func ReverseBytes64(x uint64) uint64 {
	const m = 1<<64 - 1
	x = x&(m3&m)>>8 | x&^(m3&m)<<8
	x = x&(m4&m)>>16 | x&^(m4&m)<<16
	x = x&(m5&m)>>32 | x&^(m5&m)<<32
	return x
}

// --- Len ---

// Len returns the minimum number of bits required to represent x; the result is 0 for x == 0.
func Len(x uint) int { return blen(uint64(x)) }

// Len8 returns the minimum number of bits required to represent x; the result is 0 for x == 0.
func Len8(x uint8) int { return blen(uint64(x)) }

// Len16 returns the minimum number of bits required to represent x; the result is 0 for x == 0.
func Len16(x uint16) int { return blen(uint64(x)) }

// Len32 returns the minimum number of bits required to represent x; the result is 0 for x == 0.
func Len32(x uint32) int { return blen(uint64(x)) }

// Len64 returns the minimum number of bits required to represent x; the result is 0 for x == 0.
func Len64(x uint64) int { return blen(uint64(x)) }
