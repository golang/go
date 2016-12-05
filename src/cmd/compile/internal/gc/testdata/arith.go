// run

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests arithmetic expressions

package main

import "fmt"

const (
	y = 0x0fffFFFF
)

//go:noinline
func lshNop1(x uint64) uint64 {
	// two outer shifts should be removed
	return (((x << 5) >> 2) << 2)
}

//go:noinline
func lshNop2(x uint64) uint64 {
	return (((x << 5) >> 2) << 3)
}

//go:noinline
func lshNop3(x uint64) uint64 {
	return (((x << 5) >> 2) << 6)
}

//go:noinline
func lshNotNop(x uint64) uint64 {
	// outer shift can't be removed
	return (((x << 5) >> 2) << 1)
}

//go:noinline
func rshNop1(x uint64) uint64 {
	return (((x >> 5) << 2) >> 2)
}

//go:noinline
func rshNop2(x uint64) uint64 {
	return (((x >> 5) << 2) >> 3)
}

//go:noinline
func rshNop3(x uint64) uint64 {
	return (((x >> 5) << 2) >> 6)
}

//go:noinline
func rshNotNop(x uint64) uint64 {
	return (((x >> 5) << 2) >> 1)
}

func testShiftRemoval() {
	allSet := ^uint64(0)
	if want, got := uint64(0x7ffffffffffffff), rshNop1(allSet); want != got {
		println("testShiftRemoval rshNop1 failed, wanted", want, "got", got)
		failed = true
	}
	if want, got := uint64(0x3ffffffffffffff), rshNop2(allSet); want != got {
		println("testShiftRemoval rshNop2 failed, wanted", want, "got", got)
		failed = true
	}
	if want, got := uint64(0x7fffffffffffff), rshNop3(allSet); want != got {
		println("testShiftRemoval rshNop3 failed, wanted", want, "got", got)
		failed = true
	}
	if want, got := uint64(0xffffffffffffffe), rshNotNop(allSet); want != got {
		println("testShiftRemoval rshNotNop failed, wanted", want, "got", got)
		failed = true
	}
	if want, got := uint64(0xffffffffffffffe0), lshNop1(allSet); want != got {
		println("testShiftRemoval lshNop1 failed, wanted", want, "got", got)
		failed = true
	}
	if want, got := uint64(0xffffffffffffffc0), lshNop2(allSet); want != got {
		println("testShiftRemoval lshNop2 failed, wanted", want, "got", got)
		failed = true
	}
	if want, got := uint64(0xfffffffffffffe00), lshNop3(allSet); want != got {
		println("testShiftRemoval lshNop3 failed, wanted", want, "got", got)
		failed = true
	}
	if want, got := uint64(0x7ffffffffffffff0), lshNotNop(allSet); want != got {
		println("testShiftRemoval lshNotNop failed, wanted", want, "got", got)
		failed = true
	}
}

//go:noinline
func parseLE64(b []byte) uint64 {
	// skip the first two bytes, and parse the remaining 8 as a uint64
	return uint64(b[2]) | uint64(b[3])<<8 | uint64(b[4])<<16 | uint64(b[5])<<24 |
		uint64(b[6])<<32 | uint64(b[7])<<40 | uint64(b[8])<<48 | uint64(b[9])<<56
}

//go:noinline
func parseLE32(b []byte) uint32 {
	return uint32(b[2]) | uint32(b[3])<<8 | uint32(b[4])<<16 | uint32(b[5])<<24
}

//go:noinline
func parseLE16(b []byte) uint16 {
	return uint16(b[2]) | uint16(b[3])<<8
}

// testLoadCombine tests for issue #14694 where load combining didn't respect the pointer offset.
func testLoadCombine() {
	testData := []byte{0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09}
	if want, got := uint64(0x0908070605040302), parseLE64(testData); want != got {
		println("testLoadCombine failed, wanted", want, "got", got)
		failed = true
	}
	if want, got := uint32(0x05040302), parseLE32(testData); want != got {
		println("testLoadCombine failed, wanted", want, "got", got)
		failed = true
	}
	if want, got := uint16(0x0302), parseLE16(testData); want != got {
		println("testLoadCombine failed, wanted", want, "got", got)
		failed = true
	}
}

var loadSymData = [...]byte{0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08}

func testLoadSymCombine() {
	w2 := uint16(0x0201)
	g2 := uint16(loadSymData[0]) | uint16(loadSymData[1])<<8
	if g2 != w2 {
		println("testLoadSymCombine failed, wanted", w2, "got", g2)
		failed = true
	}
	w4 := uint32(0x04030201)
	g4 := uint32(loadSymData[0]) | uint32(loadSymData[1])<<8 |
		uint32(loadSymData[2])<<16 | uint32(loadSymData[3])<<24
	if g4 != w4 {
		println("testLoadSymCombine failed, wanted", w4, "got", g4)
		failed = true
	}
	w8 := uint64(0x0807060504030201)
	g8 := uint64(loadSymData[0]) | uint64(loadSymData[1])<<8 |
		uint64(loadSymData[2])<<16 | uint64(loadSymData[3])<<24 |
		uint64(loadSymData[4])<<32 | uint64(loadSymData[5])<<40 |
		uint64(loadSymData[6])<<48 | uint64(loadSymData[7])<<56
	if g8 != w8 {
		println("testLoadSymCombine failed, wanted", w8, "got", g8)
		failed = true
	}
}

//go:noinline
func invalidAdd_ssa(x uint32) uint32 {
	return x + y + y + y + y + y + y + y + y + y + y + y + y + y + y + y + y + y
}

//go:noinline
func invalidSub_ssa(x uint32) uint32 {
	return x - y - y - y - y - y - y - y - y - y - y - y - y - y - y - y - y - y
}

//go:noinline
func invalidMul_ssa(x uint32) uint32 {
	return x * y * y * y * y * y * y * y * y * y * y * y * y * y * y * y * y * y
}

// testLargeConst tests a situation where larger than 32 bit consts were passed to ADDL
// causing an invalid instruction error.
func testLargeConst() {
	if want, got := uint32(268435440), invalidAdd_ssa(1); want != got {
		println("testLargeConst add failed, wanted", want, "got", got)
		failed = true
	}
	if want, got := uint32(4026531858), invalidSub_ssa(1); want != got {
		println("testLargeConst sub failed, wanted", want, "got", got)
		failed = true
	}
	if want, got := uint32(268435455), invalidMul_ssa(1); want != got {
		println("testLargeConst mul failed, wanted", want, "got", got)
		failed = true
	}
}

// testArithRshConst ensures that "const >> const" right shifts correctly perform
// sign extension on the lhs constant
func testArithRshConst() {
	wantu := uint64(0x4000000000000000)
	if got := arithRshuConst_ssa(); got != wantu {
		println("arithRshuConst failed, wanted", wantu, "got", got)
		failed = true
	}

	wants := int64(-0x4000000000000000)
	if got := arithRshConst_ssa(); got != wants {
		println("arithRshuConst failed, wanted", wants, "got", got)
		failed = true
	}
}

//go:noinline
func arithRshuConst_ssa() uint64 {
	y := uint64(0x8000000000000001)
	z := uint64(1)
	return uint64(y >> z)
}

//go:noinline
func arithRshConst_ssa() int64 {
	y := int64(-0x8000000000000000)
	z := uint64(1)
	return int64(y >> z)
}

//go:noinline
func arithConstShift_ssa(x int64) int64 {
	return x >> 100
}

// testArithConstShift tests that right shift by large constants preserve
// the sign of the input.
func testArithConstShift() {
	want := int64(-1)
	if got := arithConstShift_ssa(-1); want != got {
		println("arithConstShift_ssa(-1) failed, wanted", want, "got", got)
		failed = true
	}
	want = 0
	if got := arithConstShift_ssa(1); want != got {
		println("arithConstShift_ssa(1) failed, wanted", want, "got", got)
		failed = true
	}
}

// overflowConstShift_ssa verifes that constant folding for shift
// doesn't wrap (i.e. x << MAX_INT << 1 doesn't get folded to x << 0).
//go:noinline
func overflowConstShift64_ssa(x int64) int64 {
	return x << uint64(0xffffffffffffffff) << uint64(1)
}

//go:noinline
func overflowConstShift32_ssa(x int64) int32 {
	return int32(x) << uint32(0xffffffff) << uint32(1)
}

//go:noinline
func overflowConstShift16_ssa(x int64) int16 {
	return int16(x) << uint16(0xffff) << uint16(1)
}

//go:noinline
func overflowConstShift8_ssa(x int64) int8 {
	return int8(x) << uint8(0xff) << uint8(1)
}

func testOverflowConstShift() {
	want := int64(0)
	for x := int64(-127); x < int64(127); x++ {
		got := overflowConstShift64_ssa(x)
		if want != got {
			fmt.Printf("overflowShift64 failed, wanted %d got %d\n", want, got)
		}
		got = int64(overflowConstShift32_ssa(x))
		if want != got {
			fmt.Printf("overflowShift32 failed, wanted %d got %d\n", want, got)
		}
		got = int64(overflowConstShift16_ssa(x))
		if want != got {
			fmt.Printf("overflowShift16 failed, wanted %d got %d\n", want, got)
		}
		got = int64(overflowConstShift8_ssa(x))
		if want != got {
			fmt.Printf("overflowShift8 failed, wanted %d got %d\n", want, got)
		}
	}
}

// test64BitConstMult tests that rewrite rules don't fold 64 bit constants
// into multiply instructions.
func test64BitConstMult() {
	want := int64(103079215109)
	if got := test64BitConstMult_ssa(1, 2); want != got {
		println("test64BitConstMult failed, wanted", want, "got", got)
		failed = true
	}
}

//go:noinline
func test64BitConstMult_ssa(a, b int64) int64 {
	return 34359738369*a + b*34359738370
}

// test64BitConstAdd tests that rewrite rules don't fold 64 bit constants
// into add instructions.
func test64BitConstAdd() {
	want := int64(3567671782835376650)
	if got := test64BitConstAdd_ssa(1, 2); want != got {
		println("test64BitConstAdd failed, wanted", want, "got", got)
		failed = true
	}
}

//go:noinline
func test64BitConstAdd_ssa(a, b int64) int64 {
	return a + 575815584948629622 + b + 2991856197886747025
}

// testRegallocCVSpill tests that regalloc spills a value whose last use is the
// current value.
func testRegallocCVSpill() {
	want := int8(-9)
	if got := testRegallocCVSpill_ssa(1, 2, 3, 4); want != got {
		println("testRegallocCVSpill failed, wanted", want, "got", got)
		failed = true
	}
}

//go:noinline
func testRegallocCVSpill_ssa(a, b, c, d int8) int8 {
	return a + -32 + b + 63*c*-87*d
}

func testBitwiseLogic() {
	a, b := uint32(57623283), uint32(1314713839)
	if want, got := uint32(38551779), testBitwiseAnd_ssa(a, b); want != got {
		println("testBitwiseAnd failed, wanted", want, "got", got)
		failed = true
	}
	if want, got := uint32(1333785343), testBitwiseOr_ssa(a, b); want != got {
		println("testBitwiseOr failed, wanted", want, "got", got)
		failed = true
	}
	if want, got := uint32(1295233564), testBitwiseXor_ssa(a, b); want != got {
		println("testBitwiseXor failed, wanted", want, "got", got)
		failed = true
	}
	if want, got := int32(832), testBitwiseLsh_ssa(13, 4, 2); want != got {
		println("testBitwiseLsh failed, wanted", want, "got", got)
		failed = true
	}
	if want, got := int32(0), testBitwiseLsh_ssa(13, 25, 15); want != got {
		println("testBitwiseLsh failed, wanted", want, "got", got)
		failed = true
	}
	if want, got := int32(0), testBitwiseLsh_ssa(-13, 25, 15); want != got {
		println("testBitwiseLsh failed, wanted", want, "got", got)
		failed = true
	}
	if want, got := int32(-13), testBitwiseRsh_ssa(-832, 4, 2); want != got {
		println("testBitwiseRsh failed, wanted", want, "got", got)
		failed = true
	}
	if want, got := int32(0), testBitwiseRsh_ssa(13, 25, 15); want != got {
		println("testBitwiseRsh failed, wanted", want, "got", got)
		failed = true
	}
	if want, got := int32(-1), testBitwiseRsh_ssa(-13, 25, 15); want != got {
		println("testBitwiseRsh failed, wanted", want, "got", got)
		failed = true
	}
	if want, got := uint32(0x3ffffff), testBitwiseRshU_ssa(0xffffffff, 4, 2); want != got {
		println("testBitwiseRshU failed, wanted", want, "got", got)
		failed = true
	}
	if want, got := uint32(0), testBitwiseRshU_ssa(13, 25, 15); want != got {
		println("testBitwiseRshU failed, wanted", want, "got", got)
		failed = true
	}
	if want, got := uint32(0), testBitwiseRshU_ssa(0x8aaaaaaa, 25, 15); want != got {
		println("testBitwiseRshU failed, wanted", want, "got", got)
		failed = true
	}
}

//go:noinline
func testBitwiseAnd_ssa(a, b uint32) uint32 {
	return a & b
}

//go:noinline
func testBitwiseOr_ssa(a, b uint32) uint32 {
	return a | b
}

//go:noinline
func testBitwiseXor_ssa(a, b uint32) uint32 {
	return a ^ b
}

//go:noinline
func testBitwiseLsh_ssa(a int32, b, c uint32) int32 {
	return a << b << c
}

//go:noinline
func testBitwiseRsh_ssa(a int32, b, c uint32) int32 {
	return a >> b >> c
}

//go:noinline
func testBitwiseRshU_ssa(a uint32, b, c uint32) uint32 {
	return a >> b >> c
}

//go:noinline
func testShiftCX_ssa() int {
	v1 := uint8(3)
	v4 := (v1 * v1) ^ v1 | v1 - v1 - v1&v1 ^ uint8(3+2) + v1*1>>0 - v1 | 1 | v1<<(2*3|0-0*0^1)
	v5 := v4>>(3-0-uint(3)) | v1 | v1 + v1 ^ v4<<(0+1|3&1)<<(uint64(1)<<0*2*0<<0) ^ v1
	v6 := v5 ^ (v1+v1)*v1 | v1 | v1*v1>>(v1&v1)>>(uint(1)<<0*uint(3)>>1)*v1<<2*v1<<v1 - v1>>2 | (v4 - v1) ^ v1 + v1 ^ v1>>1 | v1 + v1 - v1 ^ v1
	v7 := v6 & v5 << 0
	v1++
	v11 := 2&1 ^ 0 + 3 | int(0^0)<<1>>(1*0*3) ^ 0*0 ^ 3&0*3&3 ^ 3*3 ^ 1 ^ int(2)<<(2*3) + 2 | 2 | 2 ^ 2 + 1 | 3 | 0 ^ int(1)>>1 ^ 2 // int
	v7--
	return int(uint64(2*1)<<(3-2)<<uint(3>>v7)-2)&v11 | v11 - int(2)<<0>>(2-1)*(v11*0&v11<<1<<(uint8(2)+v4))
}

func testShiftCX() {
	want := 141
	if got := testShiftCX_ssa(); want != got {
		println("testShiftCX failed, wanted", want, "got", got)
		failed = true
	}
}

// testSubqToNegq ensures that the SUBQ -> NEGQ translation works correctly.
func testSubqToNegq() {
	want := int64(-318294940372190156)
	if got := testSubqToNegq_ssa(1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2); want != got {
		println("testSubqToNegq failed, wanted", want, "got", got)
		failed = true
	}
}

//go:noinline
func testSubqToNegq_ssa(a, b, c, d, e, f, g, h, i, j, k int64) int64 {
	return a + 8207351403619448057 - b - 1779494519303207690 + c*8810076340510052032*d - 4465874067674546219 - e*4361839741470334295 - f + 8688847565426072650*g*8065564729145417479
}

func testOcom() {
	want1, want2 := int32(0x55555555), int32(-0x55555556)
	if got1, got2 := testOcom_ssa(0x55555555, 0x55555555); want1 != got1 || want2 != got2 {
		println("testSubqToNegq failed, wanted", want1, "and", want2,
			"got", got1, "and", got2)
		failed = true
	}
}

//go:noinline
func testOcom_ssa(a, b int32) (int32, int32) {
	return ^^^^a, ^^^^^b
}

func lrot1_ssa(w uint8, x uint16, y uint32, z uint64) (a uint8, b uint16, c uint32, d uint64) {
	a = (w << 5) | (w >> 3)
	b = (x << 13) | (x >> 3)
	c = (y << 29) | (y >> 3)
	d = (z << 61) | (z >> 3)
	return
}

//go:noinline
func lrot2_ssa(w, n uint32) uint32 {
	// Want to be sure that a "rotate by 32" which
	// is really 0 | (w >> 0) == w
	// is correctly compiled.
	return (w << n) | (w >> (32 - n))
}

//go:noinline
func lrot3_ssa(w uint32) uint32 {
	// Want to be sure that a "rotate by 32" which
	// is really 0 | (w >> 0) == w
	// is correctly compiled.
	return (w << 32) | (w >> (32 - 32))
}

func testLrot() {
	wantA, wantB, wantC, wantD := uint8(0xe1), uint16(0xe001),
		uint32(0xe0000001), uint64(0xe000000000000001)
	a, b, c, d := lrot1_ssa(0xf, 0xf, 0xf, 0xf)
	if a != wantA || b != wantB || c != wantC || d != wantD {
		println("lrot1_ssa(0xf, 0xf, 0xf, 0xf)=",
			wantA, wantB, wantC, wantD, ", got", a, b, c, d)
		failed = true
	}
	x := lrot2_ssa(0xb0000001, 32)
	wantX := uint32(0xb0000001)
	if x != wantX {
		println("lrot2_ssa(0xb0000001, 32)=",
			wantX, ", got", x)
		failed = true
	}
	x = lrot3_ssa(0xb0000001)
	if x != wantX {
		println("lrot3_ssa(0xb0000001)=",
			wantX, ", got", x)
		failed = true
	}

}

//go:noinline
func sub1_ssa() uint64 {
	v1 := uint64(3) // uint64
	return v1*v1 - (v1&v1)&v1
}

//go:noinline
func sub2_ssa() uint8 {
	v1 := uint8(0)
	v3 := v1 + v1 + v1 ^ v1 | 3 + v1 ^ v1 | v1 ^ v1
	v1-- // dev.ssa doesn't see this one
	return v1 ^ v1*v1 - v3
}

func testSubConst() {
	x1 := sub1_ssa()
	want1 := uint64(6)
	if x1 != want1 {
		println("sub1_ssa()=", want1, ", got", x1)
		failed = true
	}
	x2 := sub2_ssa()
	want2 := uint8(251)
	if x2 != want2 {
		println("sub2_ssa()=", want2, ", got", x2)
		failed = true
	}
}

//go:noinline
func orPhi_ssa(a bool, x int) int {
	v := 0
	if a {
		v = -1
	} else {
		v = -1
	}
	return x | v
}

func testOrPhi() {
	if want, got := -1, orPhi_ssa(true, 4); got != want {
		println("orPhi_ssa(true, 4)=", got, " want ", want)
	}
	if want, got := -1, orPhi_ssa(false, 0); got != want {
		println("orPhi_ssa(false, 0)=", got, " want ", want)
	}
}

//go:noinline
func addshiftLL_ssa(a, b uint32) uint32 {
	return a + b<<3
}

//go:noinline
func subshiftLL_ssa(a, b uint32) uint32 {
	return a - b<<3
}

//go:noinline
func rsbshiftLL_ssa(a, b uint32) uint32 {
	return a<<3 - b
}

//go:noinline
func andshiftLL_ssa(a, b uint32) uint32 {
	return a & (b << 3)
}

//go:noinline
func orshiftLL_ssa(a, b uint32) uint32 {
	return a | b<<3
}

//go:noinline
func xorshiftLL_ssa(a, b uint32) uint32 {
	return a ^ b<<3
}

//go:noinline
func bicshiftLL_ssa(a, b uint32) uint32 {
	return a &^ (b << 3)
}

//go:noinline
func notshiftLL_ssa(a uint32) uint32 {
	return ^(a << 3)
}

//go:noinline
func addshiftRL_ssa(a, b uint32) uint32 {
	return a + b>>3
}

//go:noinline
func subshiftRL_ssa(a, b uint32) uint32 {
	return a - b>>3
}

//go:noinline
func rsbshiftRL_ssa(a, b uint32) uint32 {
	return a>>3 - b
}

//go:noinline
func andshiftRL_ssa(a, b uint32) uint32 {
	return a & (b >> 3)
}

//go:noinline
func orshiftRL_ssa(a, b uint32) uint32 {
	return a | b>>3
}

//go:noinline
func xorshiftRL_ssa(a, b uint32) uint32 {
	return a ^ b>>3
}

//go:noinline
func bicshiftRL_ssa(a, b uint32) uint32 {
	return a &^ (b >> 3)
}

//go:noinline
func notshiftRL_ssa(a uint32) uint32 {
	return ^(a >> 3)
}

//go:noinline
func addshiftRA_ssa(a, b int32) int32 {
	return a + b>>3
}

//go:noinline
func subshiftRA_ssa(a, b int32) int32 {
	return a - b>>3
}

//go:noinline
func rsbshiftRA_ssa(a, b int32) int32 {
	return a>>3 - b
}

//go:noinline
func andshiftRA_ssa(a, b int32) int32 {
	return a & (b >> 3)
}

//go:noinline
func orshiftRA_ssa(a, b int32) int32 {
	return a | b>>3
}

//go:noinline
func xorshiftRA_ssa(a, b int32) int32 {
	return a ^ b>>3
}

//go:noinline
func bicshiftRA_ssa(a, b int32) int32 {
	return a &^ (b >> 3)
}

//go:noinline
func notshiftRA_ssa(a int32) int32 {
	return ^(a >> 3)
}

//go:noinline
func addshiftLLreg_ssa(a, b uint32, s uint8) uint32 {
	return a + b<<s
}

//go:noinline
func subshiftLLreg_ssa(a, b uint32, s uint8) uint32 {
	return a - b<<s
}

//go:noinline
func rsbshiftLLreg_ssa(a, b uint32, s uint8) uint32 {
	return a<<s - b
}

//go:noinline
func andshiftLLreg_ssa(a, b uint32, s uint8) uint32 {
	return a & (b << s)
}

//go:noinline
func orshiftLLreg_ssa(a, b uint32, s uint8) uint32 {
	return a | b<<s
}

//go:noinline
func xorshiftLLreg_ssa(a, b uint32, s uint8) uint32 {
	return a ^ b<<s
}

//go:noinline
func bicshiftLLreg_ssa(a, b uint32, s uint8) uint32 {
	return a &^ (b << s)
}

//go:noinline
func notshiftLLreg_ssa(a uint32, s uint8) uint32 {
	return ^(a << s)
}

//go:noinline
func addshiftRLreg_ssa(a, b uint32, s uint8) uint32 {
	return a + b>>s
}

//go:noinline
func subshiftRLreg_ssa(a, b uint32, s uint8) uint32 {
	return a - b>>s
}

//go:noinline
func rsbshiftRLreg_ssa(a, b uint32, s uint8) uint32 {
	return a>>s - b
}

//go:noinline
func andshiftRLreg_ssa(a, b uint32, s uint8) uint32 {
	return a & (b >> s)
}

//go:noinline
func orshiftRLreg_ssa(a, b uint32, s uint8) uint32 {
	return a | b>>s
}

//go:noinline
func xorshiftRLreg_ssa(a, b uint32, s uint8) uint32 {
	return a ^ b>>s
}

//go:noinline
func bicshiftRLreg_ssa(a, b uint32, s uint8) uint32 {
	return a &^ (b >> s)
}

//go:noinline
func notshiftRLreg_ssa(a uint32, s uint8) uint32 {
	return ^(a >> s)
}

//go:noinline
func addshiftRAreg_ssa(a, b int32, s uint8) int32 {
	return a + b>>s
}

//go:noinline
func subshiftRAreg_ssa(a, b int32, s uint8) int32 {
	return a - b>>s
}

//go:noinline
func rsbshiftRAreg_ssa(a, b int32, s uint8) int32 {
	return a>>s - b
}

//go:noinline
func andshiftRAreg_ssa(a, b int32, s uint8) int32 {
	return a & (b >> s)
}

//go:noinline
func orshiftRAreg_ssa(a, b int32, s uint8) int32 {
	return a | b>>s
}

//go:noinline
func xorshiftRAreg_ssa(a, b int32, s uint8) int32 {
	return a ^ b>>s
}

//go:noinline
func bicshiftRAreg_ssa(a, b int32, s uint8) int32 {
	return a &^ (b >> s)
}

//go:noinline
func notshiftRAreg_ssa(a int32, s uint8) int32 {
	return ^(a >> s)
}

// test ARM shifted ops
func testShiftedOps() {
	a, b := uint32(10), uint32(42)
	if want, got := a+b<<3, addshiftLL_ssa(a, b); got != want {
		println("addshiftLL_ssa(10, 42) =", got, " want ", want)
		failed = true
	}
	if want, got := a-b<<3, subshiftLL_ssa(a, b); got != want {
		println("subshiftLL_ssa(10, 42) =", got, " want ", want)
		failed = true
	}
	if want, got := a<<3-b, rsbshiftLL_ssa(a, b); got != want {
		println("rsbshiftLL_ssa(10, 42) =", got, " want ", want)
		failed = true
	}
	if want, got := a&(b<<3), andshiftLL_ssa(a, b); got != want {
		println("andshiftLL_ssa(10, 42) =", got, " want ", want)
		failed = true
	}
	if want, got := a|b<<3, orshiftLL_ssa(a, b); got != want {
		println("orshiftLL_ssa(10, 42) =", got, " want ", want)
		failed = true
	}
	if want, got := a^b<<3, xorshiftLL_ssa(a, b); got != want {
		println("xorshiftLL_ssa(10, 42) =", got, " want ", want)
		failed = true
	}
	if want, got := a&^(b<<3), bicshiftLL_ssa(a, b); got != want {
		println("bicshiftLL_ssa(10, 42) =", got, " want ", want)
		failed = true
	}
	if want, got := ^(a << 3), notshiftLL_ssa(a); got != want {
		println("notshiftLL_ssa(10) =", got, " want ", want)
		failed = true
	}
	if want, got := a+b>>3, addshiftRL_ssa(a, b); got != want {
		println("addshiftRL_ssa(10, 42) =", got, " want ", want)
		failed = true
	}
	if want, got := a-b>>3, subshiftRL_ssa(a, b); got != want {
		println("subshiftRL_ssa(10, 42) =", got, " want ", want)
		failed = true
	}
	if want, got := a>>3-b, rsbshiftRL_ssa(a, b); got != want {
		println("rsbshiftRL_ssa(10, 42) =", got, " want ", want)
		failed = true
	}
	if want, got := a&(b>>3), andshiftRL_ssa(a, b); got != want {
		println("andshiftRL_ssa(10, 42) =", got, " want ", want)
		failed = true
	}
	if want, got := a|b>>3, orshiftRL_ssa(a, b); got != want {
		println("orshiftRL_ssa(10, 42) =", got, " want ", want)
		failed = true
	}
	if want, got := a^b>>3, xorshiftRL_ssa(a, b); got != want {
		println("xorshiftRL_ssa(10, 42) =", got, " want ", want)
		failed = true
	}
	if want, got := a&^(b>>3), bicshiftRL_ssa(a, b); got != want {
		println("bicshiftRL_ssa(10, 42) =", got, " want ", want)
		failed = true
	}
	if want, got := ^(a >> 3), notshiftRL_ssa(a); got != want {
		println("notshiftRL_ssa(10) =", got, " want ", want)
		failed = true
	}
	c, d := int32(10), int32(-42)
	if want, got := c+d>>3, addshiftRA_ssa(c, d); got != want {
		println("addshiftRA_ssa(10, -42) =", got, " want ", want)
		failed = true
	}
	if want, got := c-d>>3, subshiftRA_ssa(c, d); got != want {
		println("subshiftRA_ssa(10, -42) =", got, " want ", want)
		failed = true
	}
	if want, got := c>>3-d, rsbshiftRA_ssa(c, d); got != want {
		println("rsbshiftRA_ssa(10, -42) =", got, " want ", want)
		failed = true
	}
	if want, got := c&(d>>3), andshiftRA_ssa(c, d); got != want {
		println("andshiftRA_ssa(10, -42) =", got, " want ", want)
		failed = true
	}
	if want, got := c|d>>3, orshiftRA_ssa(c, d); got != want {
		println("orshiftRA_ssa(10, -42) =", got, " want ", want)
		failed = true
	}
	if want, got := c^d>>3, xorshiftRA_ssa(c, d); got != want {
		println("xorshiftRA_ssa(10, -42) =", got, " want ", want)
		failed = true
	}
	if want, got := c&^(d>>3), bicshiftRA_ssa(c, d); got != want {
		println("bicshiftRA_ssa(10, -42) =", got, " want ", want)
		failed = true
	}
	if want, got := ^(d >> 3), notshiftRA_ssa(d); got != want {
		println("notshiftRA_ssa(-42) =", got, " want ", want)
		failed = true
	}
	s := uint8(3)
	if want, got := a+b<<s, addshiftLLreg_ssa(a, b, s); got != want {
		println("addshiftLLreg_ssa(10, 42, 3) =", got, " want ", want)
		failed = true
	}
	if want, got := a-b<<s, subshiftLLreg_ssa(a, b, s); got != want {
		println("subshiftLLreg_ssa(10, 42, 3) =", got, " want ", want)
		failed = true
	}
	if want, got := a<<s-b, rsbshiftLLreg_ssa(a, b, s); got != want {
		println("rsbshiftLLreg_ssa(10, 42, 3) =", got, " want ", want)
		failed = true
	}
	if want, got := a&(b<<s), andshiftLLreg_ssa(a, b, s); got != want {
		println("andshiftLLreg_ssa(10, 42, 3) =", got, " want ", want)
		failed = true
	}
	if want, got := a|b<<s, orshiftLLreg_ssa(a, b, s); got != want {
		println("orshiftLLreg_ssa(10, 42, 3) =", got, " want ", want)
		failed = true
	}
	if want, got := a^b<<s, xorshiftLLreg_ssa(a, b, s); got != want {
		println("xorshiftLLreg_ssa(10, 42, 3) =", got, " want ", want)
		failed = true
	}
	if want, got := a&^(b<<s), bicshiftLLreg_ssa(a, b, s); got != want {
		println("bicshiftLLreg_ssa(10, 42, 3) =", got, " want ", want)
		failed = true
	}
	if want, got := ^(a << s), notshiftLLreg_ssa(a, s); got != want {
		println("notshiftLLreg_ssa(10) =", got, " want ", want)
		failed = true
	}
	if want, got := a+b>>s, addshiftRLreg_ssa(a, b, s); got != want {
		println("addshiftRLreg_ssa(10, 42, 3) =", got, " want ", want)
		failed = true
	}
	if want, got := a-b>>s, subshiftRLreg_ssa(a, b, s); got != want {
		println("subshiftRLreg_ssa(10, 42, 3) =", got, " want ", want)
		failed = true
	}
	if want, got := a>>s-b, rsbshiftRLreg_ssa(a, b, s); got != want {
		println("rsbshiftRLreg_ssa(10, 42, 3) =", got, " want ", want)
		failed = true
	}
	if want, got := a&(b>>s), andshiftRLreg_ssa(a, b, s); got != want {
		println("andshiftRLreg_ssa(10, 42, 3) =", got, " want ", want)
		failed = true
	}
	if want, got := a|b>>s, orshiftRLreg_ssa(a, b, s); got != want {
		println("orshiftRLreg_ssa(10, 42, 3) =", got, " want ", want)
		failed = true
	}
	if want, got := a^b>>s, xorshiftRLreg_ssa(a, b, s); got != want {
		println("xorshiftRLreg_ssa(10, 42, 3) =", got, " want ", want)
		failed = true
	}
	if want, got := a&^(b>>s), bicshiftRLreg_ssa(a, b, s); got != want {
		println("bicshiftRLreg_ssa(10, 42, 3) =", got, " want ", want)
		failed = true
	}
	if want, got := ^(a >> s), notshiftRLreg_ssa(a, s); got != want {
		println("notshiftRLreg_ssa(10) =", got, " want ", want)
		failed = true
	}
	if want, got := c+d>>s, addshiftRAreg_ssa(c, d, s); got != want {
		println("addshiftRAreg_ssa(10, -42, 3) =", got, " want ", want)
		failed = true
	}
	if want, got := c-d>>s, subshiftRAreg_ssa(c, d, s); got != want {
		println("subshiftRAreg_ssa(10, -42, 3) =", got, " want ", want)
		failed = true
	}
	if want, got := c>>s-d, rsbshiftRAreg_ssa(c, d, s); got != want {
		println("rsbshiftRAreg_ssa(10, -42, 3) =", got, " want ", want)
		failed = true
	}
	if want, got := c&(d>>s), andshiftRAreg_ssa(c, d, s); got != want {
		println("andshiftRAreg_ssa(10, -42, 3) =", got, " want ", want)
		failed = true
	}
	if want, got := c|d>>s, orshiftRAreg_ssa(c, d, s); got != want {
		println("orshiftRAreg_ssa(10, -42, 3) =", got, " want ", want)
		failed = true
	}
	if want, got := c^d>>s, xorshiftRAreg_ssa(c, d, s); got != want {
		println("xorshiftRAreg_ssa(10, -42, 3) =", got, " want ", want)
		failed = true
	}
	if want, got := c&^(d>>s), bicshiftRAreg_ssa(c, d, s); got != want {
		println("bicshiftRAreg_ssa(10, -42, 3) =", got, " want ", want)
		failed = true
	}
	if want, got := ^(d >> s), notshiftRAreg_ssa(d, s); got != want {
		println("notshiftRAreg_ssa(-42, 3) =", got, " want ", want)
		failed = true
	}
}

var failed = false

func main() {

	test64BitConstMult()
	test64BitConstAdd()
	testRegallocCVSpill()
	testSubqToNegq()
	testBitwiseLogic()
	testOcom()
	testLrot()
	testShiftCX()
	testSubConst()
	testOverflowConstShift()
	testArithConstShift()
	testArithRshConst()
	testLargeConst()
	testLoadCombine()
	testLoadSymCombine()
	testShiftRemoval()
	testShiftedOps()

	if failed {
		panic("failed")
	}
}
