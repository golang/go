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
	// amd64/v1,amd64/v2:"BSRQ"
	// amd64/v3:"LZCNTQ", -"BSRQ"
	// s390x:"FLOGR"
	// arm:"CLZ" arm64:"CLZ"
	// mips:"CLZ"
	// wasm:"I64Clz"
	// ppc64le:"CNTLZD"
	// ppc64:"CNTLZD"
	return bits.LeadingZeros(n)
}

func LeadingZeros64(n uint64) int {
	// amd64/v1,amd64/v2:"BSRQ"
	// amd64/v3:"LZCNTQ", -"BSRQ"
	// s390x:"FLOGR"
	// arm:"CLZ" arm64:"CLZ"
	// mips:"CLZ"
	// wasm:"I64Clz"
	// ppc64le:"CNTLZD"
	// ppc64:"CNTLZD"
	return bits.LeadingZeros64(n)
}

func LeadingZeros32(n uint32) int {
	// amd64/v1,amd64/v2:"BSRQ","LEAQ",-"CMOVQEQ"
	// amd64/v3: "LZCNTL",- "BSRL"
	// s390x:"FLOGR"
	// arm:"CLZ" arm64:"CLZW"
	// mips:"CLZ"
	// wasm:"I64Clz"
	// ppc64le:"CNTLZW"
	// ppc64:"CNTLZW"
	return bits.LeadingZeros32(n)
}

func LeadingZeros16(n uint16) int {
	// amd64/v1,amd64/v2:"BSRL","LEAL",-"CMOVQEQ"
	// amd64/v3: "LZCNTL",- "BSRL"
	// s390x:"FLOGR"
	// arm:"CLZ" arm64:"CLZ"
	// mips:"CLZ"
	// wasm:"I64Clz"
	// ppc64le:"CNTLZD"
	// ppc64:"CNTLZD"
	return bits.LeadingZeros16(n)
}

func LeadingZeros8(n uint8) int {
	// amd64/v1,amd64/v2:"BSRL","LEAL",-"CMOVQEQ"
	// amd64/v3: "LZCNTL",- "BSRL"
	// s390x:"FLOGR"
	// arm:"CLZ" arm64:"CLZ"
	// mips:"CLZ"
	// wasm:"I64Clz"
	// ppc64le:"CNTLZD"
	// ppc64:"CNTLZD"
	return bits.LeadingZeros8(n)
}

// --------------- //
//    bits.Len*    //
// --------------- //

func Len(n uint) int {
	// amd64/v1,amd64/v2:"BSRQ"
	// amd64/v3: "LZCNTQ"
	// s390x:"FLOGR"
	// arm:"CLZ" arm64:"CLZ"
	// mips:"CLZ"
	// wasm:"I64Clz"
	// ppc64le:"SUBC","CNTLZD"
	// ppc64:"SUBC","CNTLZD"
	return bits.Len(n)
}

func Len64(n uint64) int {
	// amd64/v1,amd64/v2:"BSRQ"
	// amd64/v3: "LZCNTQ"
	// s390x:"FLOGR"
	// arm:"CLZ" arm64:"CLZ"
	// mips:"CLZ"
	// wasm:"I64Clz"
	// ppc64le:"SUBC","CNTLZD"
	// ppc64:"SUBC","CNTLZD"
	return bits.Len64(n)
}

func SubFromLen64(n uint64) int {
	// ppc64le:"CNTLZD",-"SUBC"
	// ppc64:"CNTLZD",-"SUBC"
	return 64 - bits.Len64(n)
}

func Len32(n uint32) int {
	// amd64/v1,amd64/v2:"BSRQ","LEAQ",-"CMOVQEQ"
	// amd64/v3: "LZCNTL"
	// s390x:"FLOGR"
	// arm:"CLZ" arm64:"CLZ"
	// mips:"CLZ"
	// wasm:"I64Clz"
	// ppc64: "CNTLZW"
	// ppc64le: "CNTLZW"
	return bits.Len32(n)
}

func Len16(n uint16) int {
	// amd64/v1,amd64/v2:"BSRL","LEAL",-"CMOVQEQ"
	// amd64/v3: "LZCNTL"
	// s390x:"FLOGR"
	// arm:"CLZ" arm64:"CLZ"
	// mips:"CLZ"
	// wasm:"I64Clz"
	// ppc64le:"SUBC","CNTLZD"
	// ppc64:"SUBC","CNTLZD"
	return bits.Len16(n)
}

func Len8(n uint8) int {
	// amd64/v1,amd64/v2:"BSRL","LEAL",-"CMOVQEQ"
	// amd64/v3: "LZCNTL"
	// s390x:"FLOGR"
	// arm:"CLZ" arm64:"CLZ"
	// mips:"CLZ"
	// wasm:"I64Clz"
	// ppc64le:"SUBC","CNTLZD"
	// ppc64:"SUBC","CNTLZD"
	return bits.Len8(n)
}

// -------------------- //
//    bits.OnesCount    //
// -------------------- //

// TODO(register args) Restore a m d 6 4 / v 1 :.*x86HasPOPCNT when only one ABI is tested.
func OnesCount(n uint) int {
	// amd64/v2:-".*x86HasPOPCNT" amd64/v3:-".*x86HasPOPCNT"
	// amd64:"POPCNTQ"
	// arm64:"VCNT","VUADDLV"
	// s390x:"POPCNT"
	// ppc64:"POPCNTD"
	// ppc64le:"POPCNTD"
	// wasm:"I64Popcnt"
	return bits.OnesCount(n)
}

func OnesCount64(n uint64) int {
	// amd64/v2:-".*x86HasPOPCNT" amd64/v3:-".*x86HasPOPCNT"
	// amd64:"POPCNTQ"
	// arm64:"VCNT","VUADDLV"
	// s390x:"POPCNT"
	// ppc64:"POPCNTD"
	// ppc64le:"POPCNTD"
	// wasm:"I64Popcnt"
	return bits.OnesCount64(n)
}

func OnesCount32(n uint32) int {
	// amd64/v2:-".*x86HasPOPCNT" amd64/v3:-".*x86HasPOPCNT"
	// amd64:"POPCNTL"
	// arm64:"VCNT","VUADDLV"
	// s390x:"POPCNT"
	// ppc64:"POPCNTW"
	// ppc64le:"POPCNTW"
	// wasm:"I64Popcnt"
	return bits.OnesCount32(n)
}

func OnesCount16(n uint16) int {
	// amd64/v2:-".*x86HasPOPCNT" amd64/v3:-".*x86HasPOPCNT"
	// amd64:"POPCNTL"
	// arm64:"VCNT","VUADDLV"
	// s390x:"POPCNT"
	// ppc64:"POPCNTW"
	// ppc64le:"POPCNTW"
	// wasm:"I64Popcnt"
	return bits.OnesCount16(n)
}

func OnesCount8(n uint8) int {
	// s390x:"POPCNT"
	// ppc64:"POPCNTB"
	// ppc64le:"POPCNTB"
	// wasm:"I64Popcnt"
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
	// arm64:"REV16W",-"UBFX",-"ORR"
	// arm/5:"SLL","SRL","ORR"
	// arm/6:"REV16"
	// arm/7:"REV16"
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
	// s390x:"RISBGZ\t[$]0, [$]63, [$]37, "
	// wasm:"I64Rotl"
	return bits.RotateLeft64(n, 37)
}

func RotateLeft32(n uint32) uint32 {
	// amd64:"ROLL" 386:"ROLL"
	// arm:`MOVW\tR[0-9]+@>23`
	// arm64:"RORW"
	// ppc64:"ROTLW"
	// ppc64le:"ROTLW"
	// s390x:"RLL"
	// wasm:"I32Rotl"
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
	// wasm:"I64Rotl"
	return bits.RotateLeft(n, m)
}

func RotateLeftVariable64(n uint64, m int) uint64 {
	// amd64:"ROLQ"
	// arm64:"ROR"
	// ppc64:"ROTL"
	// ppc64le:"ROTL"
	// s390x:"RLLG"
	// wasm:"I64Rotl"
	return bits.RotateLeft64(n, m)
}

func RotateLeftVariable32(n uint32, m int) uint32 {
	// arm:`MOVW\tR[0-9]+@>R[0-9]+`
	// amd64:"ROLL"
	// arm64:"RORW"
	// ppc64:"ROTLW"
	// ppc64le:"ROTLW"
	// s390x:"RLL"
	// wasm:"I32Rotl"
	return bits.RotateLeft32(n, m)
}

// ------------------------ //
//    bits.TrailingZeros    //
// ------------------------ //

func TrailingZeros(n uint) int {
	// amd64/v1,amd64/v2:"BSFQ","MOVL\t\\$64","CMOVQEQ"
	// amd64/v3:"TZCNTQ"
	// arm:"CLZ"
	// arm64:"RBIT","CLZ"
	// s390x:"FLOGR"
	// ppc64/power8:"ANDN","POPCNTD"
	// ppc64le/power8:"ANDN","POPCNTD"
	// ppc64/power9: "CNTTZD"
	// ppc64le/power9: "CNTTZD"
	// wasm:"I64Ctz"
	return bits.TrailingZeros(n)
}

func TrailingZeros64(n uint64) int {
	// amd64/v1,amd64/v2:"BSFQ","MOVL\t\\$64","CMOVQEQ"
	// amd64/v3:"TZCNTQ"
	// arm64:"RBIT","CLZ"
	// s390x:"FLOGR"
	// ppc64/power8:"ANDN","POPCNTD"
	// ppc64le/power8:"ANDN","POPCNTD"
	// ppc64/power9: "CNTTZD"
	// ppc64le/power9: "CNTTZD"
	// wasm:"I64Ctz"
	return bits.TrailingZeros64(n)
}

func TrailingZeros64Subtract(n uint64) int {
	// ppc64le/power8:"NEG","SUBC","ANDN","POPCNTD"
	// ppc64le/power9:"SUBC","CNTTZD"
	return bits.TrailingZeros64(1 - n)
}

func TrailingZeros32(n uint32) int {
	// amd64/v1,amd64/v2:"BTSQ\\t\\$32","BSFQ"
	// amd64/v3:"TZCNTL"
	// arm:"CLZ"
	// arm64:"RBITW","CLZW"
	// s390x:"FLOGR","MOVWZ"
	// ppc64/power8:"ANDN","POPCNTW"
	// ppc64le/power8:"ANDN","POPCNTW"
	// ppc64/power9: "CNTTZW"
	// ppc64le/power9: "CNTTZW"
	// wasm:"I64Ctz"
	return bits.TrailingZeros32(n)
}

func TrailingZeros16(n uint16) int {
	// amd64:"BSFL","BTSL\\t\\$16"
	// 386:"BSFL\t"
	// arm:"ORR\t\\$65536","CLZ",-"MOVHU\tR"
	// arm64:"ORR\t\\$65536","RBITW","CLZW",-"MOVHU\tR",-"RBIT\t",-"CLZ\t"
	// s390x:"FLOGR","OR\t\\$65536"
	// ppc64/power8:"POPCNTD","OR\\t\\$65536"
	// ppc64le/power8:"POPCNTD","OR\\t\\$65536"
	// ppc64/power9:"CNTTZD","OR\\t\\$65536"
	// ppc64le/power9:"CNTTZD","OR\\t\\$65536"
	// wasm:"I64Ctz"
	return bits.TrailingZeros16(n)
}

func TrailingZeros8(n uint8) int {
	// amd64:"BSFL","BTSL\\t\\$8"
	// arm:"ORR\t\\$256","CLZ",-"MOVBU\tR"
	// arm64:"ORR\t\\$256","RBITW","CLZW",-"MOVBU\tR",-"RBIT\t",-"CLZ\t"
	// s390x:"FLOGR","OR\t\\$256"
	// wasm:"I64Ctz"
	return bits.TrailingZeros8(n)
}

// IterateBitsNN checks special handling of TrailingZerosNN when the input is known to be non-zero.

func IterateBits(n uint) int {
	i := 0
	for n != 0 {
		// amd64/v1,amd64/v2:"BSFQ",-"CMOVEQ"
		// amd64/v3:"TZCNTQ"
		i += bits.TrailingZeros(n)
		n &= n - 1
	}
	return i
}

func IterateBits64(n uint64) int {
	i := 0
	for n != 0 {
		// amd64/v1,amd64/v2:"BSFQ",-"CMOVEQ"
		// amd64/v3:"TZCNTQ"
		i += bits.TrailingZeros64(n)
		n &= n - 1
	}
	return i
}

func IterateBits32(n uint32) int {
	i := 0
	for n != 0 {
		// amd64/v1,amd64/v2:"BSFL",-"BTSQ"
		// amd64/v3:"TZCNTL"
		i += bits.TrailingZeros32(n)
		n &= n - 1
	}
	return i
}

func IterateBits16(n uint16) int {
	i := 0
	for n != 0 {
		// amd64/v1,amd64/v2:"BSFL",-"BTSL"
		// amd64/v3:"TZCNTL"
		// arm64:"RBITW","CLZW",-"ORR"
		i += bits.TrailingZeros16(n)
		n &= n - 1
	}
	return i
}

func IterateBits8(n uint8) int {
	i := 0
	for n != 0 {
		// amd64/v1,amd64/v2:"BSFL",-"BTSL"
		// amd64/v3:"TZCNTL"
		// arm64:"RBITW","CLZW",-"ORR"
		i += bits.TrailingZeros8(n)
		n &= n - 1
	}
	return i
}

// --------------- //
//    bits.Add*    //
// --------------- //

func Add(x, y, ci uint) (r, co uint) {
	// arm64:"ADDS","ADCS","ADC",-"ADD\t",-"CMP"
	// amd64:"NEGL","ADCQ","SBBQ","NEGQ"
	// ppc64: "ADDC", "ADDE", "ADDZE"
	// ppc64le: "ADDC", "ADDE", "ADDZE"
	// s390x:"ADDE","ADDC\t[$]-1,"
	// riscv64: "ADD","SLTU"
	return bits.Add(x, y, ci)
}

func AddC(x, ci uint) (r, co uint) {
	// arm64:"ADDS","ADCS","ADC",-"ADD\t",-"CMP"
	// amd64:"NEGL","ADCQ","SBBQ","NEGQ"
	// ppc64: "ADDC", "ADDE", "ADDZE"
	// ppc64le: "ADDC", "ADDE", "ADDZE"
	// s390x:"ADDE","ADDC\t[$]-1,"
	// riscv64: "ADD","SLTU"
	return bits.Add(x, 7, ci)
}

func AddZ(x, y uint) (r, co uint) {
	// arm64:"ADDS","ADC",-"ADCS",-"ADD\t",-"CMP"
	// amd64:"ADDQ","SBBQ","NEGQ",-"NEGL",-"ADCQ"
	// ppc64: "ADDC", -"ADDE", "ADDZE"
	// ppc64le: "ADDC", -"ADDE", "ADDZE"
	// s390x:"ADDC",-"ADDC\t[$]-1,"
	// riscv64: "ADD","SLTU"
	return bits.Add(x, y, 0)
}

func AddR(x, y, ci uint) uint {
	// arm64:"ADDS","ADCS",-"ADD\t",-"CMP"
	// amd64:"NEGL","ADCQ",-"SBBQ",-"NEGQ"
	// ppc64: "ADDC", "ADDE", -"ADDZE"
	// ppc64le: "ADDC", "ADDE", -"ADDZE"
	// s390x:"ADDE","ADDC\t[$]-1,"
	// riscv64: "ADD",-"SLTU"
	r, _ := bits.Add(x, y, ci)
	return r
}

func AddM(p, q, r *[3]uint) {
	var c uint
	r[0], c = bits.Add(p[0], q[0], c)
	// arm64:"ADCS",-"ADD\t",-"CMP"
	// amd64:"ADCQ",-"NEGL",-"SBBQ",-"NEGQ"
	// s390x:"ADDE",-"ADDC\t[$]-1,"
	r[1], c = bits.Add(p[1], q[1], c)
	r[2], c = bits.Add(p[2], q[2], c)
}

func Add64(x, y, ci uint64) (r, co uint64) {
	// arm64:"ADDS","ADCS","ADC",-"ADD\t",-"CMP"
	// amd64:"NEGL","ADCQ","SBBQ","NEGQ"
	// ppc64: "ADDC", "ADDE", "ADDZE"
	// ppc64le: "ADDC", "ADDE", "ADDZE"
	// s390x:"ADDE","ADDC\t[$]-1,"
	// riscv64: "ADD","SLTU"
	return bits.Add64(x, y, ci)
}

func Add64C(x, ci uint64) (r, co uint64) {
	// arm64:"ADDS","ADCS","ADC",-"ADD\t",-"CMP"
	// amd64:"NEGL","ADCQ","SBBQ","NEGQ"
	// ppc64: "ADDC", "ADDE", "ADDZE"
	// ppc64le: "ADDC", "ADDE", "ADDZE"
	// s390x:"ADDE","ADDC\t[$]-1,"
	// riscv64: "ADD","SLTU"
	return bits.Add64(x, 7, ci)
}

func Add64Z(x, y uint64) (r, co uint64) {
	// arm64:"ADDS","ADC",-"ADCS",-"ADD\t",-"CMP"
	// amd64:"ADDQ","SBBQ","NEGQ",-"NEGL",-"ADCQ"
	// ppc64: "ADDC", -"ADDE", "ADDZE"
	// ppc64le: "ADDC", -"ADDE", "ADDZE"
	// s390x:"ADDC",-"ADDC\t[$]-1,"
	// riscv64: "ADD","SLTU"
	return bits.Add64(x, y, 0)
}

func Add64R(x, y, ci uint64) uint64 {
	// arm64:"ADDS","ADCS",-"ADD\t",-"CMP"
	// amd64:"NEGL","ADCQ",-"SBBQ",-"NEGQ"
	// ppc64: "ADDC", "ADDE", -"ADDZE"
	// ppc64le: "ADDC", "ADDE", -"ADDZE"
	// s390x:"ADDE","ADDC\t[$]-1,"
	// riscv64: "ADD",-"SLTU"
	r, _ := bits.Add64(x, y, ci)
	return r
}
func Add64M(p, q, r *[3]uint64) {
	var c uint64
	r[0], c = bits.Add64(p[0], q[0], c)
	// arm64:"ADCS",-"ADD\t",-"CMP"
	// amd64:"ADCQ",-"NEGL",-"SBBQ",-"NEGQ"
	// ppc64: -"ADDC", "ADDE", -"ADDZE"
	// ppc64le: -"ADDC", "ADDE", -"ADDZE"
	// s390x:"ADDE",-"ADDC\t[$]-1,"
	r[1], c = bits.Add64(p[1], q[1], c)
	r[2], c = bits.Add64(p[2], q[2], c)
}

func Add64MSaveC(p, q, r, c *[2]uint64) {
	// ppc64: "ADDC\tR", "ADDZE"
	// ppc64le: "ADDC\tR", "ADDZE"
	r[0], c[0] = bits.Add64(p[0], q[0], 0)
	// ppc64: "ADDC\t[$]-1", "ADDE", "ADDZE"
	// ppc64le: "ADDC\t[$]-1", "ADDE", "ADDZE"
	r[1], c[1] = bits.Add64(p[1], q[1], c[0])
}

func Add64PanicOnOverflowEQ(a, b uint64) uint64 {
	r, c := bits.Add64(a, b, 0)
	// s390x:"BRC\t[$]3,",-"ADDE"
	if c == 1 {
		panic("overflow")
	}
	return r
}

func Add64PanicOnOverflowNE(a, b uint64) uint64 {
	r, c := bits.Add64(a, b, 0)
	// s390x:"BRC\t[$]3,",-"ADDE"
	if c != 0 {
		panic("overflow")
	}
	return r
}

func Add64PanicOnOverflowGT(a, b uint64) uint64 {
	r, c := bits.Add64(a, b, 0)
	// s390x:"BRC\t[$]3,",-"ADDE"
	if c > 0 {
		panic("overflow")
	}
	return r
}

func Add64MPanicOnOverflowEQ(a, b [2]uint64) [2]uint64 {
	var r [2]uint64
	var c uint64
	r[0], c = bits.Add64(a[0], b[0], c)
	r[1], c = bits.Add64(a[1], b[1], c)
	// s390x:"BRC\t[$]3,"
	if c == 1 {
		panic("overflow")
	}
	return r
}

func Add64MPanicOnOverflowNE(a, b [2]uint64) [2]uint64 {
	var r [2]uint64
	var c uint64
	r[0], c = bits.Add64(a[0], b[0], c)
	r[1], c = bits.Add64(a[1], b[1], c)
	// s390x:"BRC\t[$]3,"
	if c != 0 {
		panic("overflow")
	}
	return r
}

func Add64MPanicOnOverflowGT(a, b [2]uint64) [2]uint64 {
	var r [2]uint64
	var c uint64
	r[0], c = bits.Add64(a[0], b[0], c)
	r[1], c = bits.Add64(a[1], b[1], c)
	// s390x:"BRC\t[$]3,"
	if c > 0 {
		panic("overflow")
	}
	return r
}

// --------------- //
//    bits.Sub*    //
// --------------- //

func Sub(x, y, ci uint) (r, co uint) {
	// amd64:"NEGL","SBBQ","NEGQ"
	// arm64:"NEGS","SBCS","NGC","NEG",-"ADD",-"SUB",-"CMP"
	// ppc64:"SUBC", "SUBE", "SUBZE", "NEG"
	// ppc64le:"SUBC", "SUBE", "SUBZE", "NEG"
	// s390x:"SUBE"
	return bits.Sub(x, y, ci)
}

func SubC(x, ci uint) (r, co uint) {
	// amd64:"NEGL","SBBQ","NEGQ"
	// arm64:"NEGS","SBCS","NGC","NEG",-"ADD",-"SUB",-"CMP"
	// ppc64:"SUBC", "SUBE", "SUBZE", "NEG"
	// ppc64le:"SUBC", "SUBE", "SUBZE", "NEG"
	// s390x:"SUBE"
	return bits.Sub(x, 7, ci)
}

func SubZ(x, y uint) (r, co uint) {
	// amd64:"SUBQ","SBBQ","NEGQ",-"NEGL"
	// arm64:"SUBS","NGC","NEG",-"SBCS",-"ADD",-"SUB\t",-"CMP"
	// ppc64:"SUBC", -"SUBE", "SUBZE", "NEG"
	// ppc64le:"SUBC", -"SUBE", "SUBZE", "NEG"
	// s390x:"SUBC"
	return bits.Sub(x, y, 0)
}

func SubR(x, y, ci uint) uint {
	// amd64:"NEGL","SBBQ",-"NEGQ"
	// arm64:"NEGS","SBCS",-"NGC",-"NEG\t",-"ADD",-"SUB",-"CMP"
	// ppc64:"SUBC", "SUBE", -"SUBZE", -"NEG"
	// ppc64le:"SUBC", "SUBE", -"SUBZE", -"NEG"
	// s390x:"SUBE"
	r, _ := bits.Sub(x, y, ci)
	return r
}
func SubM(p, q, r *[3]uint) {
	var c uint
	r[0], c = bits.Sub(p[0], q[0], c)
	// amd64:"SBBQ",-"NEGL",-"NEGQ"
	// arm64:"SBCS",-"NEGS",-"NGC",-"NEG",-"ADD",-"SUB",-"CMP"
	// ppc64:-"SUBC", "SUBE", -"SUBZE", -"NEG"
	// ppc64le:-"SUBC", "SUBE", -"SUBZE", -"NEG"
	// s390x:"SUBE"
	r[1], c = bits.Sub(p[1], q[1], c)
	r[2], c = bits.Sub(p[2], q[2], c)
}

func Sub64(x, y, ci uint64) (r, co uint64) {
	// amd64:"NEGL","SBBQ","NEGQ"
	// arm64:"NEGS","SBCS","NGC","NEG",-"ADD",-"SUB",-"CMP"
	// ppc64:"SUBC", "SUBE", "SUBZE", "NEG"
	// ppc64le:"SUBC", "SUBE", "SUBZE", "NEG"
	// s390x:"SUBE"
	return bits.Sub64(x, y, ci)
}

func Sub64C(x, ci uint64) (r, co uint64) {
	// amd64:"NEGL","SBBQ","NEGQ"
	// arm64:"NEGS","SBCS","NGC","NEG",-"ADD",-"SUB",-"CMP"
	// ppc64:"SUBC", "SUBE", "SUBZE", "NEG"
	// ppc64le:"SUBC", "SUBE", "SUBZE", "NEG"
	// s390x:"SUBE"
	return bits.Sub64(x, 7, ci)
}

func Sub64Z(x, y uint64) (r, co uint64) {
	// amd64:"SUBQ","SBBQ","NEGQ",-"NEGL"
	// arm64:"SUBS","NGC","NEG",-"SBCS",-"ADD",-"SUB\t",-"CMP"
	// ppc64:"SUBC", -"SUBE", "SUBZE", "NEG"
	// ppc64le:"SUBC", -"SUBE", "SUBZE", "NEG"
	// s390x:"SUBC"
	return bits.Sub64(x, y, 0)
}

func Sub64R(x, y, ci uint64) uint64 {
	// amd64:"NEGL","SBBQ",-"NEGQ"
	// arm64:"NEGS","SBCS",-"NGC",-"NEG\t",-"ADD",-"SUB",-"CMP"
	// ppc64:"SUBC", "SUBE", -"SUBZE", -"NEG"
	// ppc64le:"SUBC", "SUBE", -"SUBZE", -"NEG"
	// s390x:"SUBE"
	r, _ := bits.Sub64(x, y, ci)
	return r
}
func Sub64M(p, q, r *[3]uint64) {
	var c uint64
	r[0], c = bits.Sub64(p[0], q[0], c)
	// amd64:"SBBQ",-"NEGL",-"NEGQ"
	// arm64:"SBCS",-"NEGS",-"NGC",-"NEG",-"ADD",-"SUB",-"CMP"
	// s390x:"SUBE"
	r[1], c = bits.Sub64(p[1], q[1], c)
	r[2], c = bits.Sub64(p[2], q[2], c)
}

func Sub64MSaveC(p, q, r, c *[2]uint64) {
	// ppc64:"SUBC\tR\\d+, R\\d+,", "SUBZE", "NEG"
	// ppc64le:"SUBC\tR\\d+, R\\d+,", "SUBZE", "NEG"
	r[0], c[0] = bits.Sub64(p[0], q[0], 0)
	// ppc64:"SUBC\tR\\d+, [$]0,", "SUBE", "SUBZE", "NEG"
	// ppc64le:"SUBC\tR\\d+, [$]0,", "SUBE", "SUBZE", "NEG"
	r[1], c[1] = bits.Sub64(p[1], q[1], c[0])
}

func Sub64PanicOnOverflowEQ(a, b uint64) uint64 {
	r, b := bits.Sub64(a, b, 0)
	// s390x:"BRC\t[$]12,",-"ADDE",-"SUBE"
	if b == 1 {
		panic("overflow")
	}
	return r
}

func Sub64PanicOnOverflowNE(a, b uint64) uint64 {
	r, b := bits.Sub64(a, b, 0)
	// s390x:"BRC\t[$]12,",-"ADDE",-"SUBE"
	if b != 0 {
		panic("overflow")
	}
	return r
}

func Sub64PanicOnOverflowGT(a, b uint64) uint64 {
	r, b := bits.Sub64(a, b, 0)
	// s390x:"BRC\t[$]12,",-"ADDE",-"SUBE"
	if b > 0 {
		panic("overflow")
	}
	return r
}

func Sub64MPanicOnOverflowEQ(a, b [2]uint64) [2]uint64 {
	var r [2]uint64
	var c uint64
	r[0], c = bits.Sub64(a[0], b[0], c)
	r[1], c = bits.Sub64(a[1], b[1], c)
	// s390x:"BRC\t[$]12,"
	if c == 1 {
		panic("overflow")
	}
	return r
}

func Sub64MPanicOnOverflowNE(a, b [2]uint64) [2]uint64 {
	var r [2]uint64
	var c uint64
	r[0], c = bits.Sub64(a[0], b[0], c)
	r[1], c = bits.Sub64(a[1], b[1], c)
	// s390x:"BRC\t[$]12,"
	if c != 0 {
		panic("overflow")
	}
	return r
}

func Sub64MPanicOnOverflowGT(a, b [2]uint64) [2]uint64 {
	var r [2]uint64
	var c uint64
	r[0], c = bits.Sub64(a[0], b[0], c)
	r[1], c = bits.Sub64(a[1], b[1], c)
	// s390x:"BRC\t[$]12,"
	if c > 0 {
		panic("overflow")
	}
	return r
}

// --------------- //
//    bits.Mul*    //
// --------------- //

func Mul(x, y uint) (hi, lo uint) {
	// amd64:"MULQ"
	// arm64:"UMULH","MUL"
	// ppc64:"MULHDU","MULLD"
	// ppc64le:"MULHDU","MULLD"
	// s390x:"MLGR"
	// mips64: "MULVU"
	return bits.Mul(x, y)
}

func Mul64(x, y uint64) (hi, lo uint64) {
	// amd64:"MULQ"
	// arm64:"UMULH","MUL"
	// ppc64:"MULHDU","MULLD"
	// ppc64le:"MULHDU","MULLD"
	// s390x:"MLGR"
	// mips64: "MULVU"
	// riscv64:"MULHU","MUL"
	return bits.Mul64(x, y)
}

func Mul64HiOnly(x, y uint64) uint64 {
	// arm64:"UMULH",-"MUL"
	// riscv64:"MULHU",-"MUL\t"
	hi, _ := bits.Mul64(x, y)
	return hi
}

func Mul64LoOnly(x, y uint64) uint64 {
	// arm64:"MUL",-"UMULH"
	// riscv64:"MUL\t",-"MULHU"
	_, lo := bits.Mul64(x, y)
	return lo
}

// --------------- //
//    bits.Div*    //
// --------------- //

func Div(hi, lo, x uint) (q, r uint) {
	// amd64:"DIVQ"
	return bits.Div(hi, lo, x)
}

func Div32(hi, lo, x uint32) (q, r uint32) {
	// arm64:"ORR","UDIV","MSUB",-"UREM"
	return bits.Div32(hi, lo, x)
}

func Div64(hi, lo, x uint64) (q, r uint64) {
	// amd64:"DIVQ"
	return bits.Div64(hi, lo, x)
}

func Div64degenerate(x uint64) (q, r uint64) {
	// amd64:-"DIVQ"
	return bits.Div64(0, x, 5)
}
