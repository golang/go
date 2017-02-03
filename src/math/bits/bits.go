// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package bits implements bit counting and manipulation
// functions for the predeclared unsigned integer types.
package bits

// UintSize is the size of a uint in bits.
const UintSize = uintSize

// LeadingZerosN returns the number of leading zero bits in x.
// N is absent for uint, or one of 8, 16, 32, 64.
// The result is the size of x in bits for x == 0.
func LeadingZeros(x uint) int     { return UintSize - blen(uint64(x)) }
func LeadingZeros8(x uint8) int   { return 8 - blen(uint64(x)) }
func LeadingZeros16(x uint16) int { return 16 - blen(uint64(x)) }
func LeadingZeros32(x uint32) int { return 32 - blen(uint64(x)) }
func LeadingZeros64(x uint64) int { return 64 - blen(uint64(x)) }

// TrailingZerosN returns the number of trailing zero bits in x.
// N is absent for uint, or one of 8, 16, 32, 64.
// The result is the size of x in bits for x == 0.
func TrailingZeros(x uint) int     { return ntz(x) }
func TrailingZeros8(x uint8) int   { return ntz8(x) }
func TrailingZeros16(x uint16) int { return ntz16(x) }
func TrailingZeros32(x uint32) int { return ntz32(x) }
func TrailingZeros64(x uint64) int { return ntz64(x) }

// OnesCountN returns the number of one bits ("population count") in x.
// N is absent for uint, or one of 8, 16, 32, 64.
func OnesCount(x uint) int     { return pop(uint64(x)) }
func OnesCount8(x uint8) int   { return pop(uint64(x)) }
func OnesCount16(x uint16) int { return pop(uint64(x)) }
func OnesCount32(x uint32) int { return pop(uint64(x)) }
func OnesCount64(x uint64) int { return pop(uint64(x)) }

// RotateLeftN returns the value of x rotated left by k bits; k must not be negative.
// N is absent for uint, or one of 8, 16, 32, 64.
func RotateLeft(x uint, k int) uint       { return uint(rot(uint64(x), UintSize, pos(k)%UintSize)) }
func RotateLeft8(x uint8, k int) uint8    { return uint8(rot(uint64(x), 8, pos(k)%8)) }
func RotateLeft16(x uint16, k int) uint16 { return uint16(rot(uint64(x), 16, pos(k)%16)) }
func RotateLeft32(x uint32, k int) uint32 { return uint32(rot(uint64(x), 32, pos(k)%32)) }
func RotateLeft64(x uint64, k int) uint64 { return uint64(rot(uint64(x), 64, pos(k)%64)) }

// RotateRightN returns the value of x rotated right by k bits; k must not be negative.
// N is absent for uint, or one of 8, 16, 32, 64.
func RotateRight(x uint, k int) uint       { return uint(rot(uint64(x), UintSize, UintSize-pos(k)%UintSize)) }
func RotateRight8(x uint8, k int) uint8    { return uint8(rot(uint64(x), 8, 8-pos(k)%8)) }
func RotateRight16(x uint16, k int) uint16 { return uint16(rot(uint64(x), 16, 16-pos(k)%16)) }
func RotateRight32(x uint32, k int) uint32 { return uint32(rot(uint64(x), 32, 32-pos(k)%32)) }
func RotateRight64(x uint64, k int) uint64 { return uint64(rot(uint64(x), 64, 64-pos(k)%64)) }

// ReverseN returns the value of x with its bits in reversed order.
// N is absent for uint, or one of 8, 16, 32, 64.
func Reverse(x uint) uint       { return uint(rev(uint64(x), UintSize)) }
func Reverse8(x uint8) uint8    { return uint8(rev(uint64(x), 8)) }
func Reverse16(x uint16) uint16 { return uint16(rev(uint64(x), 16)) }
func Reverse32(x uint32) uint32 { return uint32(rev(uint64(x), 32)) }
func Reverse64(x uint64) uint64 { return uint64(rev(uint64(x), 64)) }

// ReverseBytesN returns the value of x with its bytes in reversed order.
// N is absent for uint, or one of 8, 16, 32, 64.
func ReverseBytes(x uint) uint       { return uint(swap(uint64(x), UintSize)) }
func ReverseBytes16(x uint16) uint16 { return uint16(swap(uint64(x), 16)) }
func ReverseBytes32(x uint32) uint32 { return uint32(swap(uint64(x), 32)) }
func ReverseBytes64(x uint64) uint64 { return uint64(swap(uint64(x), 64)) }

// LenN returns the minimum number of bits required to represent x.
// LenN(x) - 1 is the index of the most significant bit of x.
// N is absent for uint, or one of 8, 16, 32, 64.
// The result is 0 for x == 0.
func Len(x uint) int     { return blen(uint64(x)) }
func Len8(x uint8) int   { return blen(uint64(x)) }
func Len16(x uint16) int { return blen(uint64(x)) }
func Len32(x uint32) int { return blen(uint64(x)) }
func Len64(x uint64) int { return blen(uint64(x)) }
