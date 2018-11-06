// asmcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

import "math/bits"

// ----------------------- //
//    bits.LeadingZeros    //
// ----------------------- //

func LeadingZeros(n uint) int {
	// amd64:"BSRQ"
	// s390x:"FLOGR"
	// arm:"CLZ" arm64:"CLZ"
	// mips:"CLZ"
	return bits.LeadingZeros(n)
}

func LeadingZeros64(n uint64) int {
	// amd64:"BSRQ"
	// s390x:"FLOGR"
	// arm:"CLZ" arm64:"CLZ"
	// mips:"CLZ"
	return bits.LeadingZeros64(n)
}

func LeadingZeros32(n uint32) int {
	// amd64:"BSRQ","LEAQ",-"CMOVQEQ"
	// s390x:"FLOGR"
	// arm:"CLZ" arm64:"CLZ"
	// mips:"CLZ"
	return bits.LeadingZeros32(n)
}

func LeadingZeros16(n uint16) int {
	// amd64:"BSRL","LEAL",-"CMOVQEQ"
	// s390x:"FLOGR"
	// arm:"CLZ" arm64:"CLZ"
	// mips:"CLZ"
	return bits.LeadingZeros16(n)
}

func LeadingZeros8(n uint8) int {
	// amd64:"BSRL","LEAL",-"CMOVQEQ"
	// s390x:"FLOGR"
	// arm:"CLZ" arm64:"CLZ"
	// mips:"CLZ"
	return bits.LeadingZeros8(n)
}

// --------------- //
//    bits.Len*    //
// --------------- //

func Len(n uint) int {
	// amd64:"BSRQ"
	// s390x:"FLOGR"
	// arm:"CLZ" arm64:"CLZ"
	// mips:"CLZ"
	return bits.Len(n)
}

func Len64(n uint64) int {
	// amd64:"BSRQ"
	// s390x:"FLOGR"
	// arm:"CLZ" arm64:"CLZ"
	// mips:"CLZ"
	return bits.Len64(n)
}

func Len32(n uint32) int {
	// amd64:"BSRQ","LEAQ",-"CMOVQEQ"
	// s390x:"FLOGR"
	// arm:"CLZ" arm64:"CLZ"
	// mips:"CLZ"
	return bits.Len32(n)
}

func Len16(n uint16) int {
	// amd64:"BSRL","LEAL",-"CMOVQEQ"
	// s390x:"FLOGR"
	// arm:"CLZ" arm64:"CLZ"
	// mips:"CLZ"
	return bits.Len16(n)
}

func Len8(n uint8) int {
	// amd64:"BSRL","LEAL",-"CMOVQEQ"
	// s390x:"FLOGR"
	// arm:"CLZ" arm64:"CLZ"
	// mips:"CLZ"
	return bits.Len8(n)
}

// -------------------- //
//    bits.OnesCount    //
// -------------------- //

func OnesCount(n uint) int {
	// amd64:"POPCNTQ",".*x86HasPOPCNT"
	// arm64:"VCNT","VUADDLV"
	// s390x:"POPCNT"
	// ppc64:"POPCNTD"
	// ppc64le:"POPCNTD"
	return bits.OnesCount(n)
}

func OnesCount64(n uint64) int {
	// amd64:"POPCNTQ",".*x86HasPOPCNT"
	// arm64:"VCNT","VUADDLV"
	// s390x:"POPCNT"
	// ppc64:"POPCNTD"
	// ppc64le:"POPCNTD"
	return bits.OnesCount64(n)
}

func OnesCount32(n uint32) int {
	// amd64:"POPCNTL",".*x86HasPOPCNT"
	// arm64:"VCNT","VUADDLV"
	// s390x:"POPCNT"
	// ppc64:"POPCNTW"
	// ppc64le:"POPCNTW"
	return bits.OnesCount32(n)
}

func OnesCount16(n uint16) int {
	// amd64:"POPCNTL",".*x86HasPOPCNT"
	// arm64:"VCNT","VUADDLV"
	// s390x:"POPCNT"
	// ppc64:"POPCNTW"
	// ppc64le:"POPCNTW"
	return bits.OnesCount16(n)
}

func OnesCount8(n uint8) int {
	// s390x:"POPCNT"
	// ppc64:"POPCNTB"
	// ppc64le:"POPCNTB"
	return bits.OnesCount8(n)
}

// ----------------------- //
//    bits.ReverseBytes    //
// ----------------------- //

func ReverseBytes(n uint) uint {
	// amd64:"BSWAPQ"
	// s390x:"MOVDBR"
	// arm64:"REV"
	return bits.ReverseBytes(n)
}

func ReverseBytes64(n uint64) uint64 {
	// amd64:"BSWAPQ"
	// s390x:"MOVDBR"
	// arm64:"REV"
	return bits.ReverseBytes64(n)
}

func ReverseBytes32(n uint32) uint32 {
	// amd64:"BSWAPL"
	// s390x:"MOVWBR"
	// arm64:"REVW"
	return bits.ReverseBytes32(n)
}

func ReverseBytes16(n uint16) uint16 {
	// amd64:"ROLW"
	return bits.ReverseBytes16(n)
}

// --------------------- //
//    bits.RotateLeft    //
// --------------------- //

func RotateLeft64(n uint64) uint64 {
	// amd64:"ROLQ"
	// arm64:"ROR"
	// ppc64:"ROTL"
	// ppc64le:"ROTL"
	// s390x:"RLLG"
	return bits.RotateLeft64(n, 37)
}

func RotateLeft32(n uint32) uint32 {
	// amd64:"ROLL" 386:"ROLL"
	// arm64:"RORW"
	// ppc64:"ROTLW"
	// ppc64le:"ROTLW"
	// s390x:"RLL"
	return bits.RotateLeft32(n, 9)
}

func RotateLeft16(n uint16) uint16 {
	// amd64:"ROLW" 386:"ROLW"
	return bits.RotateLeft16(n, 5)
}

func RotateLeft8(n uint8) uint8 {
	// amd64:"ROLB" 386:"ROLB"
	return bits.RotateLeft8(n, 5)
}

func RotateLeftVariable(n uint, m int) uint {
	// amd64:"ROLQ"
	// arm64:"ROR"
	// ppc64:"ROTL"
	// ppc64le:"ROTL"
	// s390x:"RLLG"
	return bits.RotateLeft(n, m)
}

func RotateLeftVariable64(n uint64, m int) uint64 {
	// amd64:"ROLQ"
	// arm64:"ROR"
	// ppc64:"ROTL"
	// ppc64le:"ROTL"
	// s390x:"RLLG"
	return bits.RotateLeft64(n, m)
}

func RotateLeftVariable32(n uint32, m int) uint32 {
	// amd64:"ROLL"
	// arm64:"RORW"
	// ppc64:"ROTLW"
	// ppc64le:"ROTLW"
	// s390x:"RLL"
	return bits.RotateLeft32(n, m)
}

// ------------------------ //
//    bits.TrailingZeros    //
// ------------------------ //

func TrailingZeros(n uint) int {
	// amd64:"BSFQ","MOVL\t\\$64","CMOVQEQ"
	// s390x:"FLOGR"
	// ppc64:"ANDN","POPCNTD"
	// ppc64le:"ANDN","POPCNTD"
	return bits.TrailingZeros(n)
}

func TrailingZeros64(n uint64) int {
	// amd64:"BSFQ","MOVL\t\\$64","CMOVQEQ"
	// s390x:"FLOGR"
	// ppc64:"ANDN","POPCNTD"
	// ppc64le:"ANDN","POPCNTD"
	return bits.TrailingZeros64(n)
}

func TrailingZeros32(n uint32) int {
	// amd64:"BTSQ\\t\\$32","BSFQ"
	// s390x:"FLOGR","MOVWZ"
	// ppc64:"ANDN","POPCNTW"
	// ppc64le:"ANDN","POPCNTW"
	return bits.TrailingZeros32(n)
}

func TrailingZeros16(n uint16) int {
	// amd64:"BSFL","BTSL\\t\\$16"
	// s390x:"FLOGR","OR\t\\$65536"
	// ppc64:"POPCNTD","OR\\t\\$65536"
	// ppc64le:"POPCNTD","OR\\t\\$65536"
	return bits.TrailingZeros16(n)
}

func TrailingZeros8(n uint8) int {
	// amd64:"BSFL","BTSL\\t\\$8"
	// s390x:"FLOGR","OR\t\\$256"
	return bits.TrailingZeros8(n)
}

// IterateBitsNN checks special handling of TrailingZerosNN when the input is known to be non-zero.

func IterateBits(n uint) int {
	i := 0
	for n != 0 {
		// amd64:"BSFQ",-"CMOVEQ"
		i += bits.TrailingZeros(n)
		n &= n - 1
	}
	return i
}

func IterateBits64(n uint64) int {
	i := 0
	for n != 0 {
		// amd64:"BSFQ",-"CMOVEQ"
		i += bits.TrailingZeros64(n)
		n &= n - 1
	}
	return i
}

func IterateBits32(n uint32) int {
	i := 0
	for n != 0 {
		// amd64:"BSFL",-"BTSQ"
		i += bits.TrailingZeros32(n)
		n &= n - 1
	}
	return i
}

func IterateBits16(n uint16) int {
	i := 0
	for n != 0 {
		// amd64:"BSFL",-"BTSL"
		i += bits.TrailingZeros16(n)
		n &= n - 1
	}
	return i
}

func IterateBits8(n uint8) int {
	i := 0
	for n != 0 {
		// amd64:"BSFL",-"BTSL"
		i += bits.TrailingZeros8(n)
		n &= n - 1
	}
	return i
}

// --------------- //
//    bits.Add*    //
// --------------- //

func Add(x, y, ci uint) (r, co uint) {
	// amd64:"NEGL","ADCQ","SBBQ","NEGQ"
	return bits.Add(x, y, ci)
}

func AddC(x, ci uint) (r, co uint) {
	// amd64:"NEGL","ADCQ","SBBQ","NEGQ"
	return bits.Add(x, 7, ci)
}

func AddZ(x, y uint) (r, co uint) {
	// amd64:"ADDQ","SBBQ","NEGQ",-"NEGL",-"ADCQ"
	return bits.Add(x, y, 0)
}

func AddR(x, y, ci uint) uint {
	// amd64:"NEGL","ADCQ",-"SBBQ",-"NEGQ"
	r, _ := bits.Add(x, y, ci)
	return r
}
func AddM(p, q, r *[3]uint) {
	var c uint
	r[0], c = bits.Add(p[0], q[0], c)
	// amd64:"ADCQ",-"NEGL",-"SBBQ",-"NEGQ"
	r[1], c = bits.Add(p[1], q[1], c)
	r[2], c = bits.Add(p[2], q[2], c)
}

func Add64(x, y, ci uint64) (r, co uint64) {
	// amd64:"NEGL","ADCQ","SBBQ","NEGQ"
	return bits.Add64(x, y, ci)
}

func Add64C(x, ci uint64) (r, co uint64) {
	// amd64:"NEGL","ADCQ","SBBQ","NEGQ"
	return bits.Add64(x, 7, ci)
}

func Add64Z(x, y uint64) (r, co uint64) {
	// amd64:"ADDQ","SBBQ","NEGQ",-"NEGL",-"ADCQ"
	return bits.Add64(x, y, 0)
}

func Add64R(x, y, ci uint64) uint64 {
	// amd64:"NEGL","ADCQ",-"SBBQ",-"NEGQ"
	r, _ := bits.Add64(x, y, ci)
	return r
}
func Add64M(p, q, r *[3]uint64) {
	var c uint64
	r[0], c = bits.Add64(p[0], q[0], c)
	// amd64:"ADCQ",-"NEGL",-"SBBQ",-"NEGQ"
	r[1], c = bits.Add64(p[1], q[1], c)
	r[2], c = bits.Add64(p[2], q[2], c)
}

// --------------- //
//    bits.Sub*    //
// --------------- //

func Sub(x, y, ci uint) (r, co uint) {
	// amd64:"NEGL","SBBQ","NEGQ"
	return bits.Sub(x, y, ci)
}

func SubC(x, ci uint) (r, co uint) {
	// amd64:"NEGL","SBBQ","NEGQ"
	return bits.Sub(x, 7, ci)
}

func SubZ(x, y uint) (r, co uint) {
	// amd64:"SUBQ","SBBQ","NEGQ",-"NEGL"
	return bits.Sub(x, y, 0)
}

func SubR(x, y, ci uint) uint {
	// amd64:"NEGL","SBBQ",-"NEGQ"
	r, _ := bits.Sub(x, y, ci)
	return r
}
func SubM(p, q, r *[3]uint) {
	var c uint
	r[0], c = bits.Sub(p[0], q[0], c)
	// amd64:"SBBQ",-"NEGL",-"NEGQ"
	r[1], c = bits.Sub(p[1], q[1], c)
	r[2], c = bits.Sub(p[2], q[2], c)
}

func Sub64(x, y, ci uint64) (r, co uint64) {
	// amd64:"NEGL","SBBQ","NEGQ"
	return bits.Sub64(x, y, ci)
}

func Sub64C(x, ci uint64) (r, co uint64) {
	// amd64:"NEGL","SBBQ","NEGQ"
	return bits.Sub64(x, 7, ci)
}

func Sub64Z(x, y uint64) (r, co uint64) {
	// amd64:"SUBQ","SBBQ","NEGQ",-"NEGL"
	return bits.Sub64(x, y, 0)
}

func Sub64R(x, y, ci uint64) uint64 {
	// amd64:"NEGL","SBBQ",-"NEGQ"
	r, _ := bits.Sub64(x, y, ci)
	return r
}
func Sub64M(p, q, r *[3]uint64) {
	var c uint64
	r[0], c = bits.Sub64(p[0], q[0], c)
	// amd64:"SBBQ",-"NEGL",-"NEGQ"
	r[1], c = bits.Sub64(p[1], q[1], c)
	r[2], c = bits.Sub64(p[2], q[2], c)
}

// --------------- //
//    bits.Mul*    //
// --------------- //

func Mul(x, y uint) (hi, lo uint) {
	// amd64:"MULQ"
	// arm64:"UMULH","MUL"
	// ppc64:"MULHDU","MULLD"
	// ppc64le:"MULHDU","MULLD"
	return bits.Mul(x, y)
}

func Mul64(x, y uint64) (hi, lo uint64) {
	// amd64:"MULQ"
	// arm64:"UMULH","MUL"
	// ppc64:"MULHDU","MULLD"
	// ppc64le:"MULHDU","MULLD"
	return bits.Mul64(x, y)
}
