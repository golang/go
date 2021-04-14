// asmcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

// ------------------ //
//   masked shifts    //
// ------------------ //

func lshMask64x64(v int64, s uint64) int64 {
	// s390x:-".*AND",-".*MOVDGE"
	// ppc64le:"ANDCC",-"ORN",-"ISEL"
	// ppc64:"ANDCC",-"ORN",-"ISEL"
	return v << (s & 63)
}

func rshMask64Ux64(v uint64, s uint64) uint64 {
	// s390x:-".*AND",-".*MOVDGE"
	// ppc64le:"ANDCC",-"ORN",-"ISEL"
	// ppc64:"ANDCC",-"ORN",-"ISEL"
	return v >> (s & 63)
}

func rshMask64x64(v int64, s uint64) int64 {
	// s390x:-".*AND",-".*MOVDGE"
	// ppc64le:"ANDCC",-ORN",-"ISEL"
	// ppc64:"ANDCC",-"ORN",-"ISEL"
	return v >> (s & 63)
}

func lshMask32x64(v int32, s uint64) int32 {
	// s390x:-".*AND",-".*MOVDGE"
	// ppc64le:"ISEL",-"ORN"
	// ppc64:"ISEL",-"ORN"
	return v << (s & 63)
}

func rshMask32Ux64(v uint32, s uint64) uint32 {
	// s390x:-".*AND",-".*MOVDGE"
	// ppc64le:"ISEL",-"ORN"
	// ppc64:"ISEL",-"ORN"
	return v >> (s & 63)
}

func rshMask32x64(v int32, s uint64) int32 {
	// s390x:-".*AND",-".*MOVDGE"
	// ppc64le:"ISEL",-"ORN"
	// ppc64:"ISEL",-"ORN"
	return v >> (s & 63)
}

func lshMask64x32(v int64, s uint32) int64 {
	// s390x:-".*AND",-".*MOVDGE"
	// ppc64le:"ANDCC",-"ORN"
	// ppc64:"ANDCC",-"ORN"
	return v << (s & 63)
}

func rshMask64Ux32(v uint64, s uint32) uint64 {
	// s390x:-".*AND",-".*MOVDGE"
	// ppc64le:"ANDCC",-"ORN"
	// ppc64:"ANDCC",-"ORN"
	return v >> (s & 63)
}

func rshMask64x32(v int64, s uint32) int64 {
	// s390x:-".*AND",-".*MOVDGE"
	// ppc64le:"ANDCC",-"ORN",-"ISEL"
	// ppc64:"ANDCC",-"ORN",-"ISEL"
	return v >> (s & 63)
}

func lshMask64x32Ext(v int64, s int32) int64 {
	// s390x:-".*AND",-".*MOVDGE"
	// ppc64le:"ANDCC",-"ORN",-"ISEL"
	// ppc64:"ANDCC",-"ORN",-"ISEL"
	return v << uint(s&63)
}

func rshMask64Ux32Ext(v uint64, s int32) uint64 {
	// s390x:-".*AND",-".*MOVDGE"
	// ppc64le:"ANDCC",-"ORN",-"ISEL"
	// ppc64:"ANDCC",-"ORN",-"ISEL"
	return v >> uint(s&63)
}

func rshMask64x32Ext(v int64, s int32) int64 {
	// s390x:-".*AND",-".*MOVDGE"
	// ppc64le:"ANDCC",-"ORN",-"ISEL"
	// ppc64:"ANDCC",-"ORN",-"ISEL"
	return v >> uint(s&63)
}

// --------------- //
//  signed shifts  //
// --------------- //

// We do want to generate a test + panicshift for these cases.
func lshSigned(v8 int8, v16 int16, v32 int32, v64 int64, x int) {
	// amd64:"TESTB"
	_ = x << v8
	// amd64:"TESTW"
	_ = x << v16
	// amd64:"TESTL"
	_ = x << v32
	// amd64:"TESTQ"
	_ = x << v64
}

// We want to avoid generating a test + panicshift for these cases.
func lshSignedMasked(v8 int8, v16 int16, v32 int32, v64 int64, x int) {
	// amd64:-"TESTB"
	_ = x << (v8 & 7)
	// amd64:-"TESTW"
	_ = x << (v16 & 15)
	// amd64:-"TESTL"
	_ = x << (v32 & 31)
	// amd64:-"TESTQ"
	_ = x << (v64 & 63)
}

// ------------------ //
//   bounded shifts   //
// ------------------ //

func rshGuarded64(v int64, s uint) int64 {
	if s < 64 {
		// s390x:-".*AND",-".*MOVDGE" wasm:-"Select",-".*LtU"
		return v >> s
	}
	panic("shift too large")
}

func rshGuarded64U(v uint64, s uint) uint64 {
	if s < 64 {
		// s390x:-".*AND",-".*MOVDGE" wasm:-"Select",-".*LtU"
		return v >> s
	}
	panic("shift too large")
}

func lshGuarded64(v int64, s uint) int64 {
	if s < 64 {
		// s390x:-".*AND",-".*MOVDGE" wasm:-"Select",-".*LtU"
		return v << s
	}
	panic("shift too large")
}

func checkWidenAfterShift(v int64, u uint64) (int64, uint64) {

	// ppc64le:-".*MOVW"
	f := int32(v>>32)
	// ppc64le:".*MOVW"
	f += int32(v>>31)
	// ppc64le:-".*MOVH"
	g := int16(v>>48)
	// ppc64le:".*MOVH"
	g += int16(v>>30)
	// ppc64le:-".*MOVH"
	g += int16(f>>16)
	// ppc64le:-".*MOVB"
	h := int8(v>>56)
	// ppc64le:".*MOVB"
	h += int8(v>>28)
	// ppc64le:-".*MOVB"
	h += int8(f>>24)
	// ppc64le:".*MOVB"
	h += int8(f>>16)
	return int64(h),uint64(g)
}
