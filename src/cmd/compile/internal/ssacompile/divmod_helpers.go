// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacompile

// logX returns logarithm of n base 2.
// n must be a positive power of 2 (isPowerOfTwoX returns true).
func log8(n int8) int64 { return Log8u(uint8(n)) }

func smagic16(c int16) SmagicData { return Smagic(16, int64(c)) }

func smagic32(c int32) SmagicData { return Smagic(32, int64(c)) }

func smagic64(c int64) SmagicData { return Smagic(64, c) }

func smagic8(c int8) SmagicData { return Smagic(8, int64(c)) }

func smagicOK16(c int16) bool { return SmagicOK(16, int64(c)) }

func smagicOK32(c int32) bool { return SmagicOK(32, int64(c)) }

func smagicOK64(c int64) bool { return SmagicOK(64, c) }

// smagicOKn reports whether we should strength reduce a signed n-bit divide by c.
func smagicOK8(c int8) bool { return SmagicOK(8, int64(c)) }

func umagic16(c int16) UmagicData { return Umagic(16, int64(c)) }

func umagic32(c int32) UmagicData { return Umagic(32, int64(c)) }

// umagic32PreShifted returns the pre-shifted 64-bit magic constant for unsigned 32-bit
// division by c on 64-bit targets that have a native 64x64->128-bit multiply instruction
// (amd64 MULQ, arm64 UMULH, riscv64 MULHU, etc.), enabling:
//
//	x / c = Hmul64u(ZeroExt32to64(x), umagic32PreShifted(c))
//
// Given umagic32(c) returning m and s, the constant is (2^32 + m) << (32 - s).
// Valid when umagicOK32(c) is true. Result always fits in uint64.
func umagic32PreShifted(c int32) uint64 {
	magic := umagic32(c)
	return (1<<32 + magic.M) << uint(32-magic.S)
}

func umagic64(c int64) UmagicData { return Umagic(64, c) }

func umagic8(c int8) UmagicData { return Umagic(8, int64(c)) }

func umagicOK16(c int16) bool { return c&(c-1) != 0 }

func umagicOK32(c int32) bool { return c&(c-1) != 0 }

func umagicOK64(c int64) bool { return c&(c-1) != 0 }

// umagicOKn reports whether we should strength reduce an unsigned n-bit divide by c.
// We can strength reduce when c != 0 and c is not a power of two.
func umagicOK8(c int8) bool { return c&(c-1) != 0 }
