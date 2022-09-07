// asmcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

// ------------------ //
//   constant shifts  //
// ------------------ //

func lshConst64x64(v int64) int64 {
	// riscv64:"SLLI",-"AND",-"SLTIU"
	// ppc64le:"SLD"
	// ppc64:"SLD"
	return v << uint64(33)
}

func rshConst64Ux64(v uint64) uint64 {
	// riscv64:"SRLI",-"AND",-"SLTIU"
	// ppc64le:"SRD"
	// ppc64:"SRD"
	return v >> uint64(33)
}

func rshConst64x64(v int64) int64 {
	// riscv64:"SRAI",-"OR",-"SLTIU"
	// ppc64le:"SRAD"
	// ppc64:"SRAD"
	return v >> uint64(33)
}

func lshConst32x64(v int32) int32 {
	// riscv64:"SLLI",-"AND",-"SLTIU"
	// ppc64le:"SLW"
	// ppc64:"SLW"
	return v << uint64(29)
}

func rshConst32Ux64(v uint32) uint32 {
	// riscv64:"SRLI",-"AND",-"SLTIU"
	// ppc64le:"SRW"
	// ppc64:"SRW"
	return v >> uint64(29)
}

func rshConst32x64(v int32) int32 {
	// riscv64:"SRAI",-"OR",-"SLTIU"
	// ppc64le:"SRAW"
	// ppc64:"SRAW"
	return v >> uint64(29)
}

func lshConst64x32(v int64) int64 {
	// riscv64:"SLLI",-"AND",-"SLTIU"
	// ppc64le:"SLD"
	// ppc64:"SLD"
	return v << uint32(33)
}

func rshConst64Ux32(v uint64) uint64 {
	// riscv64:"SRLI",-"AND",-"SLTIU"
	// ppc64le:"SRD"
	// ppc64:"SRD"
	return v >> uint32(33)
}

func rshConst64x32(v int64) int64 {
	// riscv64:"SRAI",-"OR",-"SLTIU"
	// ppc64le:"SRAD"
	// ppc64:"SRAD"
	return v >> uint32(33)
}

// ------------------ //
//   masked shifts    //
// ------------------ //

func lshMask64x64(v int64, s uint64) int64 {
	// ppc64:"ANDCC",-"ORN",-"ISEL"
	// ppc64le:"ANDCC",-"ORN",-"ISEL"
	// riscv64:"SLL",-"AND\t",-"SLTIU"
	// s390x:-"RISBGZ",-"AND",-"LOCGR"
	// arm64:"LSL",-"AND"
	return v << (s & 63)
}

func rshMask64Ux64(v uint64, s uint64) uint64 {
	// ppc64:"ANDCC",-"ORN",-"ISEL"
	// ppc64le:"ANDCC",-"ORN",-"ISEL"
	// riscv64:"SRL",-"AND\t",-"SLTIU"
	// s390x:-"RISBGZ",-"AND",-"LOCGR"
	// arm64:"LSR",-"AND"
	return v >> (s & 63)
}

func rshMask64x64(v int64, s uint64) int64 {
	// ppc64:"ANDCC",-"ORN",-"ISEL"
	// ppc64le:"ANDCC",-ORN",-"ISEL"
	// riscv64:"SRA",-"OR",-"SLTIU"
	// s390x:-"RISBGZ",-"AND",-"LOCGR"
	// arm64:"ASR",-"AND"
	return v >> (s & 63)
}

func lshMask32x64(v int32, s uint64) int32 {
	// ppc64:"ISEL",-"ORN"
	// ppc64le:"ISEL",-"ORN"
	// riscv64:"SLL","AND","SLTIU"
	// s390x:-"RISBGZ",-"AND",-"LOCGR"
	// arm64:"LSL",-"AND"
	return v << (s & 63)
}

func rshMask32Ux64(v uint32, s uint64) uint32 {
	// ppc64:"ISEL",-"ORN"
	// ppc64le:"ISEL",-"ORN"
	// riscv64:"SRL","AND","SLTIU"
	// s390x:-"RISBGZ",-"AND",-"LOCGR"
	// arm64:"LSR",-"AND"
	return v >> (s & 63)
}

func rshMask32x64(v int32, s uint64) int32 {
	// ppc64:"ISEL",-"ORN"
	// ppc64le:"ISEL",-"ORN"
	// riscv64:"SRA","OR","SLTIU"
	// s390x:-"RISBGZ",-"AND",-"LOCGR"
	// arm64:"ASR",-"AND"
	return v >> (s & 63)
}

func lshMask64x32(v int64, s uint32) int64 {
	// ppc64:"ANDCC",-"ORN"
	// ppc64le:"ANDCC",-"ORN"
	// riscv64:"SLL",-"AND\t",-"SLTIU"
	// s390x:-"RISBGZ",-"AND",-"LOCGR"
	// arm64:"LSL",-"AND"
	return v << (s & 63)
}

func rshMask64Ux32(v uint64, s uint32) uint64 {
	// ppc64:"ANDCC",-"ORN"
	// ppc64le:"ANDCC",-"ORN"
	// riscv64:"SRL",-"AND\t",-"SLTIU"
	// s390x:-"RISBGZ",-"AND",-"LOCGR"
	// arm64:"LSR",-"AND"
	return v >> (s & 63)
}

func rshMask64x32(v int64, s uint32) int64 {
	// ppc64:"ANDCC",-"ORN",-"ISEL"
	// ppc64le:"ANDCC",-"ORN",-"ISEL"
	// riscv64:"SRA",-"OR",-"SLTIU"
	// s390x:-"RISBGZ",-"AND",-"LOCGR"
	// arm64:"ASR",-"AND"
	return v >> (s & 63)
}

func lshMask64x32Ext(v int64, s int32) int64 {
	// ppc64:"ANDCC",-"ORN",-"ISEL"
	// ppc64le:"ANDCC",-"ORN",-"ISEL"
	// riscv64:"SLL","AND","SLTIU"
	// s390x:-"RISBGZ",-"AND",-"LOCGR"
	return v << uint(s&63)
}

func rshMask64Ux32Ext(v uint64, s int32) uint64 {
	// ppc64:"ANDCC",-"ORN",-"ISEL"
	// ppc64le:"ANDCC",-"ORN",-"ISEL"
	// riscv64:"SRL","AND","SLTIU"
	// s390x:-"RISBGZ",-"AND",-"LOCGR"
	return v >> uint(s&63)
}

func rshMask64x32Ext(v int64, s int32) int64 {
	// ppc64:"ANDCC",-"ORN",-"ISEL"
	// ppc64le:"ANDCC",-"ORN",-"ISEL"
	// riscv64:"SRA","OR","SLTIU"
	// s390x:-"RISBGZ",-"AND",-"LOCGR"
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

func lshGuarded64(v int64, s uint) int64 {
	if s < 64 {
		// riscv64:"SLL",-"AND",-"SLTIU"
		// s390x:-"RISBGZ",-"AND",-"LOCGR"
		// wasm:-"Select",-".*LtU"
		return v << s
	}
	panic("shift too large")
}

func rshGuarded64U(v uint64, s uint) uint64 {
	if s < 64 {
		// riscv64:"SRL",-"AND",-"SLTIU"
		// s390x:-"RISBGZ",-"AND",-"LOCGR"
		// wasm:-"Select",-".*LtU"
		return v >> s
	}
	panic("shift too large")
}

func rshGuarded64(v int64, s uint) int64 {
	if s < 64 {
		// riscv64:"SRA",-"OR",-"SLTIU"
		// s390x:-"RISBGZ",-"AND",-"LOCGR"
		// wasm:-"Select",-".*LtU"
		return v >> s
	}
	panic("shift too large")
}

func checkUnneededTrunc(tab *[100000]uint32, d uint64, v uint32, h uint16, b byte) (uint32, uint64) {

	// ppc64le:-".*RLWINM",-".*RLDICR",".*CLRLSLDI"
	// ppc64:-".*RLWINM",-".*RLDICR",".*CLRLSLDI"
	f := tab[byte(v)^b]
	// ppc64le:-".*RLWINM",-".*RLDICR",".*CLRLSLDI"
	// ppc64:-".*RLWINM",-".*RLDICR",".*CLRLSLDI"
	f += tab[byte(v)&b]
	// ppc64le:-".*RLWINM",-".*RLDICR",".*CLRLSLDI"
	// ppc64:-".*RLWINM",-".*RLDICR",".*CLRLSLDI"
	f += tab[byte(v)|b]
	// ppc64le:-".*RLWINM",-".*RLDICR",".*CLRLSLDI"
	// ppc64:-".*RLWINM",-".*RLDICR",".*CLRLSLDI"
	f += tab[uint16(v)&h]
	// ppc64le:-".*RLWINM",-".*RLDICR",".*CLRLSLDI"
	// ppc64:-".*RLWINM",-".*RLDICR",".*CLRLSLDI"
	f += tab[uint16(v)^h]
	// ppc64le:-".*RLWINM",-".*RLDICR",".*CLRLSLDI"
	// ppc64:-".*RLWINM",-".*RLDICR",".*CLRLSLDI"
	f += tab[uint16(v)|h]
	// ppc64le:-".*AND",-"RLDICR",".*CLRLSLDI"
	// ppc64:-".*AND",-"RLDICR",".*CLRLSLDI"
	f += tab[v&0xff]
	// ppc64le:-".*AND",".*CLRLSLWI"
	// ppc64:-".*AND",".*CLRLSLWI"
	f += 2 * uint32(uint16(d))
	// ppc64le:-".*AND",-"RLDICR",".*CLRLSLDI"
	// ppc64:-".*AND",-"RLDICR",".*CLRLSLDI"
	g := 2 * uint64(uint32(d))
	return f, g
}

func checkCombinedShifts(v8 uint8, v16 uint16, v32 uint32, x32 int32, v64 uint64) (uint8, uint16, uint32, uint64, int64) {

	// ppc64le:-"AND","CLRLSLWI"
	// ppc64:-"AND","CLRLSLWI"
	f := (v8 & 0xF) << 2
	// ppc64le:"CLRLSLWI"
	// ppc64:"CLRLSLWI"
	f += byte(v16) << 3
	// ppc64le:-"AND","CLRLSLWI"
	// ppc64:-"AND","CLRLSLWI"
	g := (v16 & 0xFF) << 3
	// ppc64le:-"AND","CLRLSLWI"
	// ppc64:-"AND","CLRLSLWI"
	h := (v32 & 0xFFFFF) << 2
	// ppc64le:"CLRLSLDI"
	// ppc64:"CLRLSLDI"
	i := (v64 & 0xFFFFFFFF) << 5
	// ppc64le:-"CLRLSLDI"
	// ppc64:-"CLRLSLDI"
	i += (v64 & 0xFFFFFFF) << 38
	// ppc64le/power9:-"CLRLSLDI"
	// ppc64/power9:-"CLRLSLDI"
	i += (v64 & 0xFFFF00) << 10
	// ppc64le/power9:-"SLD","EXTSWSLI"
	// ppc64/power9:-"SLD","EXTSWSLI"
	j := int64(x32+32) * 8
	return f, g, h, i, j
}

func checkWidenAfterShift(v int64, u uint64) (int64, uint64) {

	// ppc64le:-".*MOVW"
	f := int32(v >> 32)
	// ppc64le:".*MOVW"
	f += int32(v >> 31)
	// ppc64le:-".*MOVH"
	g := int16(v >> 48)
	// ppc64le:".*MOVH"
	g += int16(v >> 30)
	// ppc64le:-".*MOVH"
	g += int16(f >> 16)
	// ppc64le:-".*MOVB"
	h := int8(v >> 56)
	// ppc64le:".*MOVB"
	h += int8(v >> 28)
	// ppc64le:-".*MOVB"
	h += int8(f >> 24)
	// ppc64le:".*MOVB"
	h += int8(f >> 16)
	return int64(h), uint64(g)
}

func checkShiftAndMask32(v []uint32) {
	i := 0

	// ppc64le: "RLWNM\t[$]24, R[0-9]+, [$]12, [$]19, R[0-9]+"
	// ppc64: "RLWNM\t[$]24, R[0-9]+, [$]12, [$]19, R[0-9]+"
	v[i] = (v[i] & 0xFF00000) >> 8
	i++
	// ppc64le: "RLWNM\t[$]26, R[0-9]+, [$]22, [$]29, R[0-9]+"
	// ppc64: "RLWNM\t[$]26, R[0-9]+, [$]22, [$]29, R[0-9]+"
	v[i] = (v[i] & 0xFF00) >> 6
	i++
	// ppc64le: "MOVW\tR0"
	// ppc64: "MOVW\tR0"
	v[i] = (v[i] & 0xFF) >> 8
	i++
	// ppc64le: "MOVW\tR0"
	// ppc64: "MOVW\tR0"
	v[i] = (v[i] & 0xF000000) >> 28
	i++
	// ppc64le: "RLWNM\t[$]26, R[0-9]+, [$]24, [$]31, R[0-9]+"
	// ppc64: "RLWNM\t[$]26, R[0-9]+, [$]24, [$]31, R[0-9]+"
	v[i] = (v[i] >> 6) & 0xFF
	i++
	// ppc64le: "RLWNM\t[$]26, R[0-9]+, [$]12, [$]19, R[0-9]+"
	// ppc64: "RLWNM\t[$]26, R[0-9]+, [$]12, [$]19, R[0-9]+"
	v[i] = (v[i] >> 6) & 0xFF000
	i++
	// ppc64le: "MOVW\tR0"
	// ppc64: "MOVW\tR0"
	v[i] = (v[i] >> 20) & 0xFF000
	i++
	// ppc64le: "MOVW\tR0"
	// ppc64: "MOVW\tR0"
	v[i] = (v[i] >> 24) & 0xFF00
	i++
}

func checkMergedShifts32(a [256]uint32, b [256]uint64, u uint32, v uint32) {
	// ppc64le: -"CLRLSLDI", "RLWNM\t[$]10, R[0-9]+, [$]22, [$]29, R[0-9]+"
	// ppc64: -"CLRLSLDI", "RLWNM\t[$]10, R[0-9]+, [$]22, [$]29, R[0-9]+"
	a[0] = a[uint8(v>>24)]
	// ppc64le: -"CLRLSLDI", "RLWNM\t[$]11, R[0-9]+, [$]21, [$]28, R[0-9]+"
	// ppc64: -"CLRLSLDI", "RLWNM\t[$]11, R[0-9]+, [$]21, [$]28, R[0-9]+"
	b[0] = b[uint8(v>>24)]
	// ppc64le: -"CLRLSLDI", "RLWNM\t[$]15, R[0-9]+, [$]21, [$]28, R[0-9]+"
	// ppc64: -"CLRLSLDI", "RLWNM\t[$]15, R[0-9]+, [$]21, [$]28, R[0-9]+"
	b[1] = b[(v>>20)&0xFF]
	// ppc64le: -"SLD", "RLWNM\t[$]10, R[0-9]+, [$]22, [$]28, R[0-9]+"
	// ppc64: -"SLD", "RLWNM\t[$]10, R[0-9]+, [$]22, [$]28, R[0-9]+"
	b[2] = b[v>>25]
}

// 128 bit shifts

func check128bitShifts(x, y uint64, bits uint) (uint64, uint64) {
	s := bits & 63
	ŝ := (64 - bits) & 63
	// check that the shift operation has two commas (three operands)
	// amd64:"SHRQ.*,.*,"
	shr := x>>s | y<<ŝ
	// amd64:"SHLQ.*,.*,"
	shl := x<<s | y>>ŝ
	return shr, shl
}

func checkShiftToMask(u []uint64, s []int64) {
	// amd64:-"SHR",-"SHL","ANDQ"
	u[0] = u[0] >> 5 << 5
	// amd64:-"SAR",-"SHL","ANDQ"
	s[0] = s[0] >> 5 << 5
	// amd64:-"SHR",-"SHL","ANDQ"
	u[1] = u[1] << 5 >> 5
}
