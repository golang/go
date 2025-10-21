// asmcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

func cmovint(c int) int {
	x := c + 4
	if x < 0 {
		x = 182
	}
	// amd64:"CMOVQLT"
	// arm64:"CSEL LT"
	// ppc64x:"ISEL [$]0"
	// wasm:"Select"
	// riscv64/rva20u64, riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
	// riscv64/rva23u64:`CZEROEQZ`, `CZERONEZ`, `OR`
	return x
}

func cmovchan(x, y chan int) chan int {
	if x != y {
		x = y
	}
	// amd64:"CMOVQNE"
	// arm64:"CSEL NE"
	// ppc64x:"ISEL [$]2"
	// wasm:"Select"
	// riscv64/rva20u64, riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
	// riscv64/rva23u64:`CZEROEQZ`, `CZERONEZ`, `OR`
	return x
}

func cmovuintptr(x, y uintptr) uintptr {
	if x < y {
		x = -y
	}
	// amd64:"CMOVQ(HI|CS)"
	// arm64:"CSNEG LS"
	// ppc64x:"ISEL [$]1"
	// wasm:"Select"
	// riscv64/rva20u64, riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
	// riscv64/rva23u64:`CZEROEQZ`, `CZERONEZ`, `OR`
	return x
}

func cmov32bit(x, y uint32) uint32 {
	if x < y {
		x = -y
	}
	// amd64:"CMOVL(HI|CS)"
	// arm64:"CSNEG (LS|HS)"
	// ppc64x:"ISEL [$]1"
	// wasm:"Select"
	// riscv64/rva20u64, riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
	// riscv64/rva23u64:`CZEROEQZ`, `CZERONEZ`, `OR`
	return x
}

func cmov16bit(x, y uint16) uint16 {
	if x < y {
		x = -y
	}
	// amd64:"CMOVW(HI|CS)"
	// arm64:"CSNEG (LS|HS)"
	// ppc64x:"ISEL [$][01]"
	// wasm:"Select"
	// riscv64/rva20u64, riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
	// riscv64/rva23u64:`CZEROEQZ`, `CZERONEZ`, `OR`
	return x
}

// Floating point comparison. For EQ/NE, we must
// generate special code to handle NaNs.
func cmovfloateq(x, y float64) int {
	a := 128
	if x == y {
		a = 256
	}
	// amd64:"CMOVQNE" "CMOVQPC"
	// arm64:"CSEL EQ"
	// ppc64x:"ISEL [$]2"
	// wasm:"Select"
	// riscv64/rva20u64, riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
	// riscv64/rva23u64:`CZEROEQZ`, `CZERONEZ`, `OR`
	return a
}

func cmovfloatne(x, y float64) int {
	a := 128
	if x != y {
		a = 256
	}
	// amd64:"CMOVQNE" "CMOVQPS"
	// arm64:"CSEL NE"
	// ppc64x:"ISEL [$]2"
	// wasm:"Select"
	// riscv64/rva20u64, riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
	// riscv64/rva23u64:`CZEROEQZ`, `CZERONEZ`, `OR`
	return a
}

//go:noinline
func frexp(f float64) (frac float64, exp int) {
	return 1.0, 4
}

//go:noinline
func ldexp(frac float64, exp int) float64 {
	return 1.0
}

// Generate a CMOV with a floating comparison and integer move.
func cmovfloatint2(x, y float64) float64 {
	yfr, yexp := 4.0, 5

	r := x
	for r >= y {
		rfr, rexp := frexp(r)
		if rfr < yfr {
			rexp = rexp - 42
		}
		// amd64:"CMOVQHI"
		// arm64:"CSEL MI"
		// ppc64x:"ISEL [$]0"
		// wasm:"Select"
		// riscv64/rva20u64, riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
		// riscv64/rva23u64:`CZEROEQZ`, `CZERONEZ`, `OR`
		r = r - ldexp(y, rexp-yexp)
	}
	return r
}

func cmovloaded(x [4]int, y int) int {
	if x[2] != 0 {
		y = x[2]
	} else {
		y = y >> 2
	}
	// amd64:"CMOVQNE"
	// arm64:"CSEL NE"
	// ppc64x:"ISEL [$]2"
	// wasm:"Select"
	// riscv64/rva20u64, riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
	// riscv64/rva23u64:`CZEROEQZ`, `CZERONEZ`, `OR`, -`SNEZ`
	return y
}

func cmovuintptr2(x, y uintptr) uintptr {
	a := x * 2
	if a == 0 {
		a = 256
	}
	// amd64:"CMOVQEQ"
	// arm64:"CSEL EQ"
	// ppc64x:"ISEL [$]2"
	// wasm:"Select"
	// riscv64/rva20u64, riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
	// riscv64/rva23u64:`CZEROEQZ`, `CZERONEZ`, `OR`, -`SEQZ`
	return a
}

// Floating point CMOVs are not supported by amd64/arm64/ppc64x
func cmovfloatmove(x, y int) float64 {
	a := 1.0
	if x <= y {
		a = 2.0
	}
	// amd64:-"CMOV"
	// arm64:-"CSEL"
	// ppc64x:-"ISEL"
	// wasm:-"Select"
	return a
}

// On amd64, the following patterns trigger comparison inversion.
// Test that we correctly invert the CMOV condition
var gsink int64
var gusink uint64

func cmovinvert1(x, y int64) int64 {
	if x < gsink {
		y = -y
	}
	// amd64:"CMOVQGT"
	// riscv64/rva20u64, riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
	// riscv64/rva23u64:`CZEROEQZ`, `CZERONEZ`, `OR`
	return y
}
func cmovinvert2(x, y int64) int64 {
	if x <= gsink {
		y = -y
	}
	// amd64:"CMOVQGE"
	return y
}
func cmovinvert3(x, y int64) int64 {
	if x == gsink {
		y = -y
	}
	// amd64:"CMOVQEQ"
	// riscv64/rva20u64, riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
	// riscv64/rva23u64:`CZEROEQZ`, `CZERONEZ`, `OR`
	return y
}
func cmovinvert4(x, y int64) int64 {
	if x != gsink {
		y = -y
	}
	// amd64:"CMOVQNE"
	// riscv64/rva20u64, riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
	// riscv64/rva23u64:`CZEROEQZ`, `CZERONEZ`, `OR`
	return y
}
func cmovinvert5(x, y uint64) uint64 {
	if x > gusink {
		y = -y
	}
	// amd64:"CMOVQCS"
	// riscv64/rva20u64, riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
	// riscv64/rva23u64:`CZEROEQZ`, `CZERONEZ`, `OR`
	return y
}
func cmovinvert6(x, y uint64) uint64 {
	if x >= gusink {
		y = -y
	}
	// amd64:"CMOVQLS"
	// riscv64/rva20u64, riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
	// riscv64/rva23u64:`CZEROEQZ`, `CZERONEZ`, `OR`
	return y
}

func cmovload(a []int, i int, b bool) int {
	if b {
		i += 42
	}
	// See issue 26306
	// amd64:-"CMOVQNE"
	return a[i]
}

func cmovstore(a []int, i int, b bool) {
	if b {
		i += 42
	}
	// amd64:"CMOVQNE"
	// riscv64/rva20u64, riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
	// riscv64/rva23u64:`CZEROEQZ`, `CZERONEZ`, `OR`
	a[i] = 7
}

var r0, r1, r2, r3, r4, r5 int

func cmovinc(cond bool, a, b, c int) {
	var x0, x1 int

	if cond {
		x0 = a
	} else {
		x0 = b + 1
	}
	// arm64:"CSINC NE", -"CSEL"
	// riscv64/rva20u64, riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
	// riscv64/rva23u64:`CZEROEQZ`, `CZERONEZ`, `OR`
	r0 = x0

	if cond {
		x1 = b + 1
	} else {
		x1 = a
	}
	// arm64:"CSINC EQ", -"CSEL"
	// riscv64/rva20u64, riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
	// riscv64/rva23u64:`CZEROEQZ`, `CZERONEZ`, `OR`
	r1 = x1

	if cond {
		c++
	}
	// arm64:"CSINC EQ", -"CSEL"
	r2 = c
}

func cmovinv(cond bool, a, b int) {
	var x0, x1 int

	if cond {
		x0 = a
	} else {
		x0 = ^b
	}
	// arm64:"CSINV NE", -"CSEL"
	// riscv64/rva20u64, riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
	// riscv64/rva23u64:`CZEROEQZ`, `CZERONEZ`, `OR`
	r0 = x0

	if cond {
		x1 = ^b
	} else {
		x1 = a
	}
	// arm64:"CSINV EQ", -"CSEL"
	// riscv64/rva20u64, riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
	// riscv64/rva23u64:`CZEROEQZ`, `CZERONEZ`, `OR`
	r1 = x1
}

func cmovneg(cond bool, a, b, c int) {
	var x0, x1 int

	if cond {
		x0 = a
	} else {
		x0 = -b
	}
	// arm64:"CSNEG NE", -"CSEL"
	// riscv64/rva20u64, riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
	// riscv64/rva23u64:`CZEROEQZ`, `CZERONEZ`, `OR`
	r0 = x0

	if cond {
		x1 = -b
	} else {
		x1 = a
	}
	// arm64:"CSNEG EQ", -"CSEL"
	// riscv64/rva20u64, riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
	// riscv64/rva23u64:`CZEROEQZ`, `CZERONEZ`, `OR`
	r1 = x1
}

func cmovsetm(cond bool, x int) {
	var x0, x1 int

	if cond {
		x0 = -1
	} else {
		x0 = 0
	}
	// arm64:"CSETM NE", -"CSEL"
	// riscv64/rva20u64, riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
	// riscv64/rva23u64:`CZEROEQZ`, -`CZERONEZ`
	r0 = x0

	if cond {
		x1 = 0
	} else {
		x1 = -1
	}
	// arm64:"CSETM EQ", -"CSEL"
	// riscv64/rva20u64, riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
	// riscv64/rva23u64:`CZERONEZ`, -`CZEROEQZ`
	r1 = x1
}

func cmovFcmp0(s, t float64, a, b int) {
	var x0, x1, x2, x3, x4, x5 int

	if s < t {
		x0 = a
	} else {
		x0 = b + 1
	}
	// arm64:"CSINC MI", -"CSEL"
	// riscv64/rva20u64, riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
	// riscv64/rva23u64:`CZEROEQZ`, `CZERONEZ`, `OR`
	r0 = x0

	if s <= t {
		x1 = a
	} else {
		x1 = ^b
	}
	// arm64:"CSINV LS", -"CSEL"
	// riscv64/rva20u64, riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
	// riscv64/rva23u64:`CZEROEQZ`, `CZERONEZ`, `OR`
	r1 = x1

	if s > t {
		x2 = a
	} else {
		x2 = -b
	}
	// arm64:"CSNEG MI", -"CSEL"
	// riscv64/rva20u64, riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
	// riscv64/rva23u64:`CZEROEQZ`, `CZERONEZ`, `OR`
	r2 = x2

	if s >= t {
		x3 = -1
	} else {
		x3 = 0
	}
	// arm64:"CSETM LS", -"CSEL"
	// riscv64/rva20u64, riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
	// riscv64/rva23u64:`CZEROEQZ`, -`CZERONEZ`
	r3 = x3

	if s == t {
		x4 = a
	} else {
		x4 = b + 1
	}
	// arm64:"CSINC EQ", -"CSEL"
	// riscv64/rva20u64, riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
	// riscv64/rva23u64:`CZEROEQZ`, `CZERONEZ`, `OR`
	r4 = x4

	if s != t {
		x5 = a
	} else {
		x5 = b + 1
	}
	// arm64:"CSINC NE", -"CSEL"
	// riscv64/rva20u64, riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
	// riscv64/rva23u64:`CZEROEQZ`, `CZERONEZ`, `OR`
	r5 = x5
}

func cmovFcmp1(s, t float64, a, b int) {
	var x0, x1, x2, x3, x4, x5 int

	if s < t {
		x0 = b + 1
	} else {
		x0 = a
	}
	// arm64:"CSINC PL", -"CSEL"
	// riscv64/rva20u64, riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
	// riscv64/rva23u64:`CZEROEQZ`, `CZERONEZ`, `OR`
	r0 = x0

	if s <= t {
		x1 = ^b
	} else {
		x1 = a
	}
	// arm64:"CSINV HI", -"CSEL"
	// riscv64/rva20u64, riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
	// riscv64/rva23u64:`CZEROEQZ`, `CZERONEZ`, `OR`
	r1 = x1

	if s > t {
		x2 = -b
	} else {
		x2 = a
	}
	// arm64:"CSNEG PL", -"CSEL"
	// riscv64/rva20u64, riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
	// riscv64/rva23u64:`CZEROEQZ`, `CZERONEZ`, `OR`
	r2 = x2

	if s >= t {
		x3 = 0
	} else {
		x3 = -1
	}
	// arm64:"CSETM HI", -"CSEL"
	// riscv64/rva20u64, riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
	// riscv64/rva23u64:`CZERONEZ`,-`CZEROEQZ`
	r3 = x3

	if s == t {
		x4 = b + 1
	} else {
		x4 = a
	}
	// arm64:"CSINC NE", -"CSEL"
	// riscv64/rva20u64, riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
	// riscv64/rva23u64:`CZEROEQZ`, `CZERONEZ`, `OR`
	r4 = x4

	if s != t {
		x5 = b + 1
	} else {
		x5 = a
	}
	// arm64:"CSINC EQ", -"CSEL"
	// riscv64/rva20u64, riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
	// riscv64/rva23u64:`CZEROEQZ`, `CZERONEZ`, `OR`
	r5 = x5
}

func cmovzero1(c bool) int {
	var x int
	if c {
		x = 182
	}
	// loong64:"MASKEQZ", -"MASKNEZ"
	// riscv64/rva20u64, riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
	// riscv64/rva23u64:`CZEROEQZ`, -`CZERONEZ`
	return x
}

func cmovzero2(c bool) int {
	var x int
	if !c {
		x = 182
	}
	// loong64:"MASKNEZ", -"MASKEQZ"
	// riscv64/rva20u64, riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
	// riscv64/rva23u64:`CZERONEZ`, -`CZEROEQZ`
	return x
}

// Conditionally selecting between a value or 0 can be done without
// an extra load of 0 to a register on PPC64 by using R0 (which always
// holds the value $0) instead. Verify both cases where either arg1
// or arg2 is zero.
func cmovzeroregZero(a, b int) int {
	x := 0
	if a == b {
		x = a
	}
	// ppc64x:"ISEL [$]2, R[0-9]+, R0, R[0-9]+"
	// riscv64/rva20u64, riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
	// riscv64/rva23u64:`CZEROEQZ`,-`CZERONEZ`
	return x
}

func cmovzeroreg1(a, b int) int {
	x := a
	if a == b {
		x = 0
	}
	// ppc64x:"ISEL [$]2, R0, R[0-9]+, R[0-9]+"
	// riscv64/rva20u64, riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
	// riscv64/rva23u64:`CZERONEZ`, -`CZEROEQZ`
	return x
}

func cmovmathadd(a uint, b bool) uint {
	if b {
		a++
	}
	// amd64:"ADDQ", -"CMOV"
	// arm64:"CSINC", -"CSEL"
	// ppc64x:"ADD", -"ISEL"
	// wasm:"I64Add", -"Select"
	return a
}

func cmovmathsub(a uint, b bool) uint {
	if b {
		a--
	}
	// amd64:"SUBQ", -"CMOV"
	// arm64:"SUB", -"CSEL"
	// ppc64x:"SUB", -"ISEL"
	// wasm:"I64Sub", -"Select"
	return a
}

func cmovmathdouble(a uint, b bool) uint {
	if b {
		a *= 2
	}
	// amd64:"SHL", -"CMOV"
	// amd64/v3:"SHL", -"CMOV", -"MOV"
	// arm64:"LSL", -"CSEL"
	// wasm:"I64Shl", -"Select"
	return a
}

func cmovmathhalvei(a int, b bool) int {
	if b {
		// For some reason the compiler attributes the shift to inside this block rather than where the Phi node is.
		// arm64:"ASR", -"CSEL"
		// wasm:"I64ShrS", -"Select"
		a /= 2
	}
	// arm64:-"CSEL"
	// wasm:-"Select"
	return a
}

func cmovmathhalveu(a uint, b bool) uint {
	if b {
		a /= 2
	}
	// amd64:"SHR", -"CMOV"
	// amd64/v3:"SHR", -"CMOV", -"MOV"
	// arm64:"LSR", -"CSEL"
	// wasm:"I64ShrU", -"Select"
	return a
}

func branchlessBoolToUint8(b bool) (r uint8) {
	if b {
		r = 1
	}
	return
}

func cmovFromMulFromFlags64(x uint64, b bool) uint64 {
	// amd64:-"MOVB.ZX"
	r := uint64(branchlessBoolToUint8(b))
	// amd64:"CMOV",-"MOVB.ZX",-"MUL"
	return x * r
}
func cmovFromMulFromFlags64sext(x int64, b bool) int64 {
	// amd64:-"MOVB.ZX"
	r := int64(int8(branchlessBoolToUint8(b)))
	// amd64:"CMOV",-"MOVB.ZX",-"MUL"
	return x * r
}

func branchlessBoolToUint8(b bool) (r uint8) {
	if b {
		r = 1
	}
	return
}

func cmovFromMulFromFlags64(x uint64, b bool) uint64 {
	// amd64:-"MOVB.ZX"
	r := uint64(branchlessBoolToUint8(b))
	// amd64:"CMOV",-"MOVB.ZX",-"MUL"
	return x * r
}
func cmovFromMulFromFlags64sext(x int64, b bool) int64 {
	// amd64:-"MOVB.ZX"
	r := int64(int8(branchlessBoolToUint8(b)))
	// amd64:"CMOV",-"MOVB.ZX",-"MUL"
	return x * r
}

func cmoveAddZero(cond, a, b int) int {
	if cond == 0 {
		a += b
	}
	// riscv64/rva23u64:`CZERONEZ`, `ADD`, -`SEQZ`, -`CZEROEQZ`, -`OR`
	return a
}

func cmoveAddNonZero(cond, a, b int) int {
	if cond != 0 {
		a += b
	}
	// riscv64/rva23u64:`CZEROEQZ`, `ADD`, -`SNEZ`, -`CZERONEZ`, -`OR`
	return a
}

func cmoveSubZero(cond, a, b int) int {
	if cond == 0 {
		a -= b
	}
	// riscv64/rva23u64:`CZERONEZ`, `SUB`, -`SEQZ`, -`CZEROEQZ`, -`OR`
	return a
}

func cmoveSubNonZero(cond, a, b int) int {
	if cond != 0 {
		a -= b
	}
	// riscv64/rva23u64:`CZEROEQZ`, `SUB`, -`SNEZ`, -`CZERONEZ`, -`OR`
	return a
}

func cmoveOrZero(cond, a, b int) int {
	if cond == 0 {
		a |= b
	}
	// riscv64/rva23u64:`CZERONEZ`, `OR`, -`SEQZ`, -`CZEROEQZ`
	return a
}

func cmoveOrNonZero(cond, a, b int) int {
	if cond != 0 {
		a |= b
	}
	// riscv64/rva23u64:`CZEROEQZ`, `OR`, -`SNEZ`, -`CZERONEZ`
	return a
}

func cmoveXorZero(cond, a, b int) int {
	if cond == 0 {
		a ^= b
	}
	// riscv64/rva23u64:`CZERONEZ`, `XOR`, -`SEQZ`, -`CZEROEQZ`, -`OR`
	return a
}

func cmoveXorNonZero(cond, a, b int) int {
	if cond != 0 {
		a ^= b
	}
	// riscv64/rva23u64:`CZEROEQZ`, `XOR`, -`SNEZ`, -`CZERONEZ`, -`OR`
	return a
}

func cmoveAndZero(cond, a, b int) int {
	if cond == 0 {
		a &= b
	}
	// riscv64/rva23u64:`CZEROEQZ`, `AND`, `OR`, -`SEQZ`, -`CZERONEZ`
	return a
}

func CondAndNonZero(cond, a, b int) int {
	if cond != 0 {
		a &= b
	}
	// riscv64/rva23u64:`CZERONEZ`, `AND`, `OR`, -`SNEZ`, -`CZEROEQZ`
	return a
}

// Immediate variants tests
func cmoveAddiZero(cond, a int) int {
	if cond == 0 {
		a += 42
	}
	// riscv64/rva20u64, riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
	// riscv64/rva23u64:`CZERONEZ`, `ADD`, -`SEQZ`, -`CZEROEQZ`, -`OR`
	return a
}

func cmoveAddiNonZero(cond, a int) int {
	if cond != 0 {
		a += 42
	}
	// riscv64/rva20u64, riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
	// riscv64/rva23u64:`CZEROEQZ`, `ADD`, -`SNEZ`, -`CZERONEZ`, -`OR`
	return a
}

func cmoveOriZero(cond, a int) int {
	if cond == 0 {
		a |= 0xFF
	}
	// riscv64/rva20u64, riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
	// riscv64/rva23u64:`CZERONEZ`, -`SEQZ`, -`CZEROEQZ`
	return a
}

func cmoveOriNonZero(cond, a int) int {
	if cond != 0 {
		a |= 0xFF
	}
	// riscv64/rva23u64:`CZEROEQZ`, -`SNEZ`, -`CZERONEZ`
	return a
}

func cmoveXoriZero(cond, a int) int {
	if cond == 0 {
		a ^= 0xFFFF
	}
	// riscv64/rva23u64:`CZERONEZ`, `XOR`, -`SEQZ`, -`CZEROEQZ`, -`OR`
	return a
}

func cmoveXoriNonZero(cond, a int) int {
	if cond != 0 {
		a ^= 0xFFFF
	}
	// riscv64/rva23u64:`CZEROEQZ`, `XOR`, -`SNEZ`, -`CZERONEZ`, -`OR`
	return a
}

func cmoveAndiZero(cond, a int) int {
	if cond == 0 {
		a &= 0xFF
	}
	// riscv64/rva23u64:`CZEROEQZ`, `AND`, `OR`, -`SEQZ`, -`CZERONEZ`
	return a
}

func cmoveAndiNonZero(cond, a int) int {
	if cond != 0 {
		a &= 0xFF
	}
	// riscv64/rva23u64:`CZERONEZ`, `AND`, `OR`, -`SNEZ`, -`CZEROEQZ`
	return a
}

// 32-bit immediate variant tests
func cmoveAddiwZero(cond int32, a int32) int32 {
	if cond == 0 {
		a += 42
	}
	// riscv64/rva20u64, riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
	// riscv64/rva23u64:`CZERONEZ`, `ADD`, -`SEQZ`, -`CZEROEQZ`, -`OR`
	return a
}

func cmoveAddiwNonZero(cond int32, a int32) int32 {
	if cond != 0 {
		a += 42
	}
	// riscv64/rva20u64, riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
	// riscv64/rva23u64:`CZEROEQZ`, `ADD`, -`SNEZ`, -`CZERONEZ`, -`OR`
	return a
}

func cmoveAddwZero(cond int32, a, b int32) int32 {
	if cond == 0 {
		a += b
	}
	// riscv64/rva20u64, riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
	// riscv64/rva23u64:`CZERONEZ`, `ADD`, -`SEQZ`, -`CZEROEQZ`, -`OR`
	return a
}

func cmoveAddwNonZero(cond int32, a, b int32) int32 {
	if cond != 0 {
		a += b
	}
	// riscv64/rva20u64, riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
	// riscv64/rva23u64:`CZEROEQZ`, `ADD`, -`SNEZ`, -`CZERONEZ`, -`OR`
	return a
}

func cmoveSubwZero(cond int32, a, b int32) int32 {
	if cond == 0 {
		a -= b
	}
	// riscv64/rva20u64, riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
	// riscv64/rva23u64:`CZERONEZ`, `SUB`, -`SEQZ`, -`CZEROEQZ`, -`OR`
	return a
}

func cmoveSubwNonZero(cond int32, a, b int32) int32 {
	if cond != 0 {
		a -= b
	}
	// riscv64/rva20u64, riscv64/rva22u64:-`CZEROEQZ`, -`CZERONEZ`
	// riscv64/rva23u64:`CZEROEQZ`, `SUB`, -`SNEZ`, -`CZERONEZ`, -`OR`
	return a
}
