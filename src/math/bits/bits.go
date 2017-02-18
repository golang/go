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

const m0 = 0x5555555555555555 // 01010101 ...
const m1 = 0x3333333333333333 // 00110011 ...
const m2 = 0x0f0f0f0f0f0f0f0f // 00001111 ...
const m3 = 0x00ff00ff00ff00ff // etc.
const m4 = 0x0000ffff0000ffff

// OnesCount returns the number of one bits ("population count") in x.
func OnesCount(x uint) int {
	if UintSize == 32 {
		return OnesCount32(uint32(x))
	}
	return OnesCount64(uint64(x))
}

// OnesCount8 returns the number of one bits ("population count") in x.
func OnesCount8(x uint8) int {
	const m = 1<<8 - 1
	x = x>>1&(m0&m) + x&(m0&m)
	x = x>>2&(m1&m) + x&(m1&m)
	x += x >> 4
	return int(x) & (1<<4 - 1)
}

// OnesCount16 returns the number of one bits ("population count") in x.
func OnesCount16(x uint16) int {
	const m = 1<<16 - 1
	x = x>>1&(m0&m) + x&(m0&m)
	x = x>>2&(m1&m) + x&(m1&m)
	x = (x>>4 + x) & (m2 & m)
	x += x >> 8
	return int(x) & (1<<5 - 1)
}

// OnesCount32 returns the number of one bits ("population count") in x.
func OnesCount32(x uint32) int {
	const m = 1<<32 - 1
	x = x>>1&(m0&m) + x&(m0&m)
	x = x>>2&(m1&m) + x&(m1&m)
	x = (x>>4 + x) & (m2 & m)
	x += x >> 8
	x += x >> 16
	return int(x) & (1<<6 - 1)
}

// OnesCount64 returns the number of one bits ("population count") in x.
func OnesCount64(x uint64) int {
	// Implementation: Parallel summing of adjacent bits.
	// See "Hacker's Delight", Chap. 5: Counting Bits.
	// The following pattern shows the general approach:
	//
	//   x = x>>1&(m0&m) + x&(m0&m)
	//   x = x>>2&(m1&m) + x&(m1&m)
	//   x = x>>4&(m2&m) + x&(m2&m)
	//   x = x>>8&(m3&m) + x&(m3&m)
	//   x = x>>16&(m4&m) + x&(m4&m)
	//   x = x>>32&(m5&m) + x&(m5&m)
	//   return int(x)
	//
	// Masking (& operations) can be left away when there's no
	// danger that a field's sum will carry over into the next
	// field: Since the result cannot be > 64, 8 bits is enough
	// and we can ignore the masks for the shifts by 8 and up.
	// Per "Hacker's Delight", the first line can be simplified
	// more, but it saves at best one instruction, so we leave
	// it alone for clarity.
	const m = 1<<64 - 1
	x = x>>1&(m0&m) + x&(m0&m)
	x = x>>2&(m1&m) + x&(m1&m)
	x = (x>>4 + x) & (m2 & m)
	x += x >> 8
	x += x >> 16
	x += x >> 32
	return int(x) & (1<<7 - 1)
}

// --- RotateLeft ---

// RotateLeft returns the value of x rotated left by k bits; k must not be negative.
func RotateLeft(x uint, k int) uint {
	if UintSize == 32 {
		return uint(RotateLeft32(uint32(x), k))
	}
	return uint(RotateLeft64(uint64(x), k))
}

// RotateLeft8 returns the value of x rotated left by k bits; k must not be negative.
func RotateLeft8(x uint8, k int) uint8 {
	if k < 0 {
		panic("negative rotation count")
	}
	const n = 8
	s := uint(k) & (n - 1)
	return x<<s | x>>(n-s)
}

// RotateLeft16 returns the value of x rotated left by k bits; k must not be negative.
func RotateLeft16(x uint16, k int) uint16 {
	if k < 0 {
		panic("negative rotation count")
	}
	const n = 16
	s := uint(k) & (n - 1)
	return x<<s | x>>(n-s)
}

// RotateLeft32 returns the value of x rotated left by k bits; k must not be negative.
func RotateLeft32(x uint32, k int) uint32 {
	if k < 0 {
		panic("negative rotation count")
	}
	const n = 32
	s := uint(k) & (n - 1)
	return x<<s | x>>(n-s)
}

// RotateLeft64 returns the value of x rotated left by k bits; k must not be negative.
func RotateLeft64(x uint64, k int) uint64 {
	if k < 0 {
		panic("negative rotation count")
	}
	const n = 64
	s := uint(k) & (n - 1)
	return x<<s | x>>(n-s)
}

// --- RotateRight ---

// RotateRight returns the value of x rotated left by k bits; k must not be negative.
func RotateRight(x uint, k int) uint {
	if UintSize == 32 {
		return uint(RotateRight32(uint32(x), k))
	}
	return uint(RotateRight64(uint64(x), k))
}

// RotateRight8 returns the value of x rotated left by k bits; k must not be negative.
func RotateRight8(x uint8, k int) uint8 {
	if k < 0 {
		panic("negative rotation count")
	}
	const n = 8
	s := uint(k) & (n - 1)
	return x<<(n-s) | x>>s
}

// RotateRight16 returns the value of x rotated left by k bits; k must not be negative.
func RotateRight16(x uint16, k int) uint16 {
	if k < 0 {
		panic("negative rotation count")
	}
	const n = 16
	s := uint(k) & (n - 1)
	return x<<(n-s) | x>>s
}

// RotateRight32 returns the value of x rotated left by k bits; k must not be negative.
func RotateRight32(x uint32, k int) uint32 {
	if k < 0 {
		panic("negative rotation count")
	}
	const n = 32
	s := uint(k) & (n - 1)
	return x<<(n-s) | x>>s
}

// RotateRight64 returns the value of x rotated left by k bits; k must not be negative.
func RotateRight64(x uint64, k int) uint64 {
	if k < 0 {
		panic("negative rotation count")
	}
	const n = 64
	s := uint(k) & (n - 1)
	return x<<(n-s) | x>>s
}

// --- Reverse ---

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
	x = x>>1&(m0&m) | x&(m0&m)<<1
	x = x>>2&(m1&m) | x&(m1&m)<<2
	return x>>4 | x<<4
}

// Reverse16 returns the value of x with its bits in reversed order.
func Reverse16(x uint16) uint16 {
	const m = 1<<16 - 1
	x = x>>1&(m0&m) | x&(m0&m)<<1
	x = x>>2&(m1&m) | x&(m1&m)<<2
	x = x>>4&(m2&m) | x&(m2&m)<<4
	return x>>8 | x<<8
}

// Reverse32 returns the value of x with its bits in reversed order.
func Reverse32(x uint32) uint32 {
	const m = 1<<32 - 1
	x = x>>1&(m0&m) | x&(m0&m)<<1
	x = x>>2&(m1&m) | x&(m1&m)<<2
	x = x>>4&(m2&m) | x&(m2&m)<<4
	x = x>>8&(m3&m) | x&(m3&m)<<8
	return x>>16 | x<<16
}

// Reverse64 returns the value of x with its bits in reversed order.
func Reverse64(x uint64) uint64 {
	const m = 1<<64 - 1
	x = x>>1&(m0&m) | x&(m0&m)<<1
	x = x>>2&(m1&m) | x&(m1&m)<<2
	x = x>>4&(m2&m) | x&(m2&m)<<4
	x = x>>8&(m3&m) | x&(m3&m)<<8
	x = x>>16&(m4&m) | x&(m4&m)<<16
	return x>>32 | x<<32
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
	return x>>8 | x<<8
}

// ReverseBytes32 returns the value of x with its bytes in reversed order.
func ReverseBytes32(x uint32) uint32 {
	const m = 1<<32 - 1
	x = x>>8&(m3&m) | x&(m3&m)<<8
	return x>>16 | x<<16
}

// ReverseBytes64 returns the value of x with its bytes in reversed order.
func ReverseBytes64(x uint64) uint64 {
	const m = 1<<64 - 1
	x = x>>8&(m3&m) | x&(m3&m)<<8
	x = x>>16&(m4&m) | x&(m4&m)<<16
	return x>>32 | x<<32
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
