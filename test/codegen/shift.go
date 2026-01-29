// asmcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

// ------------------ //
//   constant shifts  //
// ------------------ //

func lshConst64x64(v int64) int64 {
	// loong64:"SLLV"
	// ppc64x:"SLD"
	// riscv64:"SLLI" -"AND" -"SLTIU"
	return v << uint64(33)
}

func rshConst64Ux64(v uint64) uint64 {
	// loong64:"SRLV"
	// ppc64x:"SRD"
	// riscv64:"SRLI " -"AND" -"SLTIU"
	return v >> uint64(33)
}

func rshConst64Ux64Overflow32(v uint32) uint64 {
	// loong64:"MOVV R0," -"SRL "
	// riscv64:"MOV [$]0," -"SRL"
	return uint64(v) >> 32
}

func rshConst64Ux64Overflow16(v uint16) uint64 {
	// loong64:"MOVV R0," -"SRLV"
	// riscv64:"MOV [$]0," -"SRL"
	return uint64(v) >> 16
}

func rshConst64Ux64Overflow8(v uint8) uint64 {
	// loong64:"MOVV R0," -"SRLV"
	// riscv64:"MOV [$]0," -"SRL"
	return uint64(v) >> 8
}

func rshConst64x64(v int64) int64 {
	// loong64:"SRAV"
	// ppc64x:"SRAD"
	// riscv64:"SRAI " -"OR" -"SLTIU"
	return v >> uint64(33)
}

func rshConst64x64Overflow32(v int32) int64 {
	// loong64:"SRA [$]31"
	// riscv64:"SRAIW" -"SLLI" -"SRAI "
	return int64(v) >> 32
}

func rshConst64x64Overflow16(v int16) int64 {
	// loong64:"SLLV [$]48" "SRAV [$]63"
	// riscv64:"SLLI" "SRAI" -"SRAIW"
	return int64(v) >> 16
}

func rshConst64x64Overflow8(v int8) int64 {
	// loong64:"SLLV [$]56" "SRAV [$]63"
	// riscv64:"SLLI" "SRAI" -"SRAIW"
	return int64(v) >> 8
}

func lshConst32x1(v int32) int32 {
	// amd64:"ADDL", -"SHLL"
	return v << 1
}

func lshConst64x1(v int64) int64 {
	// amd64:"ADDQ", -"SHLQ"
	return v << 1
}

func lshConst32x64(v int32) int32 {
	// loong64:"SLL "
	// ppc64x:"SLW"
	// riscv64:"SLLI" -"AND" -"SLTIU", -"MOVW"
	return v << uint64(29)
}

func rshConst32Ux64(v uint32) uint32 {
	// loong64:"SRL "
	// ppc64x:"SRW"
	// riscv64:"SRLIW" -"AND" -"SLTIU", -"MOVW"
	return v >> uint64(29)
}

func rshConst32x64(v int32) int32 {
	// loong64:"SRA "
	// ppc64x:"SRAW"
	// riscv64:"SRAIW" -"OR" -"SLTIU", -"MOVW"
	return v >> uint64(29)
}

func lshConst64x32(v int64) int64 {
	// loong64:"SLLV"
	// ppc64x:"SLD"
	// riscv64:"SLLI" -"AND" -"SLTIU"
	return v << uint32(33)
}

func rshConst64Ux32(v uint64) uint64 {
	// loong64:"SRLV"
	// ppc64x:"SRD"
	// riscv64:"SRLI " -"AND" -"SLTIU"
	return v >> uint32(33)
}

func rshConst64x32(v int64) int64 {
	// loong64:"SRAV"
	// ppc64x:"SRAD"
	// riscv64:"SRAI " -"OR" -"SLTIU"
	return v >> uint32(33)
}

func lshConst32x1Add(x int32) int32 {
	// amd64:"SHLL [$]2"
	// loong64:"SLL [$]2"
	// riscv64:"SLLI [$]2"
	return (x + x) << 1
}

func lshConst64x1Add(x int64) int64 {
	// amd64:"SHLQ [$]2"
	// loong64:"SLLV [$]2"
	// riscv64:"SLLI [$]2"
	return (x + x) << 1
}

func lshConst32x2Add(x int32) int32 {
	// amd64:"SHLL [$]3"
	// loong64:"SLL [$]3"
	// riscv64:"SLLI [$]3"
	return (x + x) << 2
}

func lshConst64x2Add(x int64) int64 {
	// amd64:"SHLQ [$]3"
	// loong64:"SLLV [$]3"
	// riscv64:"SLLI [$]3"
	return (x + x) << 2
}

func lshConst32x31Add(x int32) int32 {
	// loong64:-"SLL " "MOVV R0"
	// riscv64:-"SLLI" "MOV [$]0"
	return (x + x) << 31
}

func lshConst64x63Add(x int64) int64 {
	// loong64:-"SLLV" "MOVV R0"
	// riscv64:-"SLLI" "MOV [$]0"
	return (x + x) << 63
}

// ------------------ //
//   masked shifts    //
// ------------------ //

func lshMask64x64(v int64, s uint64) int64 {
	// arm64:"LSL" -"AND"
	// loong64:"SLLV" -"AND"
	// ppc64x:"RLDICL" -"ORN" -"ISEL"
	// riscv64:"SLL" -"AND " -"SLTIU"
	// s390x:-"RISBGZ" -"AND" -"LOCGR"
	return v << (s & 63)
}

func rshMask64Ux64(v uint64, s uint64) uint64 {
	// arm64:"LSR" -"AND" -"CSEL"
	// loong64:"SRLV" -"AND"
	// ppc64x:"RLDICL" -"ORN" -"ISEL"
	// riscv64:"SRL " -"AND " -"SLTIU"
	// s390x:-"RISBGZ" -"AND" -"LOCGR"
	return v >> (s & 63)
}

func rshMask64x64(v int64, s uint64) int64 {
	// arm64:"ASR" -"AND" -"CSEL"
	// loong64:"SRAV" -"AND"
	// ppc64x:"RLDICL" -"ORN" -"ISEL"
	// riscv64:"SRA " -"OR" -"SLTIU"
	// s390x:-"RISBGZ" -"AND" -"LOCGR"
	return v >> (s & 63)
}

func lshMask32x64(v int32, s uint64) int32 {
	// arm64:"LSL" -"AND"
	// loong64:"SLL " "AND" "SGTU" "MASKEQZ"
	// ppc64x:"ISEL" -"ORN"
	// riscv64:"SLL" -"AND " -"SLTIU"
	// s390x:-"RISBGZ" -"AND" -"LOCGR"
	return v << (s & 63)
}

func lsh5Mask32x64(v int32, s uint64) int32 {
	// loong64:"SLL " -"AND"
	return v << (s & 31)
}

func rshMask32Ux64(v uint32, s uint64) uint32 {
	// arm64:"LSR" -"AND"
	// loong64:"SRL " "AND" "SGTU" "MASKEQZ"
	// ppc64x:"ISEL" -"ORN"
	// riscv64:"SRLW" "SLTIU" "NEG" "AND " -"SRL "
	// s390x:-"RISBGZ" -"AND" -"LOCGR"
	return v >> (s & 63)
}

func rsh5Mask32Ux64(v uint32, s uint64) uint32 {
	// loong64:"SRL " -"AND"
	// riscv64:"SRLW" -"AND " -"SLTIU" -"SRL "
	return v >> (s & 31)
}

func rshMask32x64(v int32, s uint64) int32 {
	// arm64:"ASR" -"AND"
	// loong64:"SRA " "AND" "SGTU" "SUBVU" "OR"
	// ppc64x:"ISEL" -"ORN"
	// riscv64:"SRAW" "OR" "SLTIU"
	// s390x:-"RISBGZ" -"AND" -"LOCGR"
	return v >> (s & 63)
}

func rsh5Mask32x64(v int32, s uint64) int32 {
	// loong64:"SRA " -"AND"
	// riscv64:"SRAW" -"OR" -"SLTIU"
	return v >> (s & 31)
}

func lshMask64x32(v int64, s uint32) int64 {
	// arm64:"LSL" -"AND"
	// loong64:"SLLV" -"AND"
	// ppc64x:"RLDICL" -"ORN"
	// riscv64:"SLL" -"AND " -"SLTIU"
	// s390x:-"RISBGZ" -"AND" -"LOCGR"
	return v << (s & 63)
}

func rshMask64Ux32(v uint64, s uint32) uint64 {
	// arm64:"LSR" -"AND" -"CSEL"
	// loong64:"SRLV" -"AND"
	// ppc64x:"RLDICL" -"ORN"
	// riscv64:"SRL " -"AND " -"SLTIU"
	// s390x:-"RISBGZ" -"AND" -"LOCGR"
	return v >> (s & 63)
}

func rshMask64x32(v int64, s uint32) int64 {
	// arm64:"ASR" -"AND" -"CSEL"
	// loong64:"SRAV" -"AND"
	// ppc64x:"RLDICL" -"ORN" -"ISEL"
	// riscv64:"SRA " -"OR" -"SLTIU"
	// s390x:-"RISBGZ" -"AND" -"LOCGR"
	return v >> (s & 63)
}

func lshMask64x32Ext(v int64, s int32) int64 {
	// ppc64x:"RLDICL" -"ORN" -"ISEL"
	// riscv64:"SLL" -"AND " -"SLTIU"
	// s390x:-"RISBGZ" -"AND" -"LOCGR"
	return v << uint(s&63)
}

func rshMask64Ux32Ext(v uint64, s int32) uint64 {
	// ppc64x:"RLDICL" -"ORN" -"ISEL"
	// riscv64:"SRL " -"AND " -"SLTIU"
	// s390x:-"RISBGZ" -"AND" -"LOCGR"
	return v >> uint(s&63)
}

func rshMask64x32Ext(v int64, s int32) int64 {
	// ppc64x:"RLDICL" -"ORN" -"ISEL"
	// riscv64:"SRA " -"OR" -"SLTIU"
	// s390x:-"RISBGZ" -"AND" -"LOCGR"
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
		// riscv64:"SLL" -"AND" -"SLTIU"
		// s390x:-"RISBGZ" -"AND" -"LOCGR"
		// wasm:-"Select" -".*LtU"
		// arm64:"LSL" -"CSEL"
		return v << s
	}
	panic("shift too large")
}

func rshGuarded64U(v uint64, s uint) uint64 {
	if s < 64 {
		// riscv64:"SRL " -"AND" -"SLTIU"
		// s390x:-"RISBGZ" -"AND" -"LOCGR"
		// wasm:-"Select" -".*LtU"
		// arm64:"LSR" -"CSEL"
		return v >> s
	}
	panic("shift too large")
}

func rshGuarded64(v int64, s uint) int64 {
	if s < 64 {
		// riscv64:"SRA " -"OR" -"SLTIU"
		// s390x:-"RISBGZ" -"AND" -"LOCGR"
		// wasm:-"Select" -".*LtU"
		// arm64:"ASR" -"CSEL"
		return v >> s
	}
	panic("shift too large")
}

func provedUnsignedShiftLeft(val64 uint64, val32 uint32, val16 uint16, val8 uint8, shift int) (r1 uint64, r2 uint32, r3 uint16, r4 uint8) {
	if shift >= 0 && shift < 64 {
		// arm64:"LSL" -"CSEL"
		r1 = val64 << shift
	}
	if shift >= 0 && shift < 32 {
		// arm64:"LSL" -"CSEL"
		r2 = val32 << shift
	}
	if shift >= 0 && shift < 16 {
		// arm64:"LSL" -"CSEL"
		r3 = val16 << shift
	}
	if shift >= 0 && shift < 8 {
		// arm64:"LSL" -"CSEL"
		r4 = val8 << shift
	}
	return r1, r2, r3, r4
}

func provedSignedShiftLeft(val64 int64, val32 int32, val16 int16, val8 int8, shift int) (r1 int64, r2 int32, r3 int16, r4 int8) {
	if shift >= 0 && shift < 64 {
		// arm64:"LSL" -"CSEL"
		r1 = val64 << shift
	}
	if shift >= 0 && shift < 32 {
		// arm64:"LSL" -"CSEL"
		r2 = val32 << shift
	}
	if shift >= 0 && shift < 16 {
		// arm64:"LSL" -"CSEL"
		r3 = val16 << shift
	}
	if shift >= 0 && shift < 8 {
		// arm64:"LSL" -"CSEL"
		r4 = val8 << shift
	}
	return r1, r2, r3, r4
}

func provedUnsignedShiftRight(val64 uint64, val32 uint32, val16 uint16, val8 uint8, shift int) (r1 uint64, r2 uint32, r3 uint16, r4 uint8) {
	if shift >= 0 && shift < 64 {
		// arm64:"LSR" -"CSEL"
		r1 = val64 >> shift
	}
	if shift >= 0 && shift < 32 {
		// arm64:"LSR" -"CSEL"
		r2 = val32 >> shift
	}
	if shift >= 0 && shift < 16 {
		// arm64:"LSR" -"CSEL"
		r3 = val16 >> shift
	}
	if shift >= 0 && shift < 8 {
		// arm64:"LSR" -"CSEL"
		r4 = val8 >> shift
	}
	return r1, r2, r3, r4
}

func provedSignedShiftRight(val64 int64, val32 int32, val16 int16, val8 int8, shift int) (r1 int64, r2 int32, r3 int16, r4 int8) {
	if shift >= 0 && shift < 64 {
		// arm64:"ASR" -"CSEL"
		r1 = val64 >> shift
	}
	if shift >= 0 && shift < 32 {
		// arm64:"ASR" -"CSEL"
		r2 = val32 >> shift
	}
	if shift >= 0 && shift < 16 {
		// arm64:"ASR" -"CSEL"
		r3 = val16 >> shift
	}
	if shift >= 0 && shift < 8 {
		// arm64:"ASR" -"CSEL"
		r4 = val8 >> shift
	}
	return r1, r2, r3, r4
}

func checkUnneededTrunc(tab *[100000]uint32, d uint64, v uint32, h uint16, b byte) (uint32, uint64) {

	// ppc64x:-".*RLWINM" -".*RLDICR" ".*CLRLSLDI"
	f := tab[byte(v)^b]
	// ppc64x:-".*RLWINM" -".*RLDICR" ".*CLRLSLDI"
	f += tab[byte(v)&b]
	// ppc64x:-".*RLWINM" -".*RLDICR" ".*CLRLSLDI"
	f += tab[byte(v)|b]
	// ppc64x:-".*RLWINM" -".*RLDICR" ".*CLRLSLDI"
	f += tab[uint16(v)&h]
	// ppc64x:-".*RLWINM" -".*RLDICR" ".*CLRLSLDI"
	f += tab[uint16(v)^h]
	// ppc64x:-".*RLWINM" -".*RLDICR" ".*CLRLSLDI"
	f += tab[uint16(v)|h]
	// ppc64x:-".*AND" -"RLDICR" ".*CLRLSLDI"
	f += tab[v&0xff]
	// ppc64x:-".*AND" ".*CLRLSLWI"
	f += 2 * uint32(uint16(d))
	// ppc64x:-".*AND" -"RLDICR" ".*CLRLSLDI"
	g := 2 * uint64(uint32(d))
	return f, g
}

func checkCombinedShifts(v8 uint8, v16 uint16, v32 uint32, x32 int32, v64 uint64) (uint8, uint16, uint32, uint64, int64) {

	// ppc64x:-"AND" "CLRLSLWI"
	f := (v8 & 0xF) << 2
	// ppc64x:"CLRLSLWI"
	f += byte(v16) << 3
	// ppc64x:-"AND" "CLRLSLWI"
	g := (v16 & 0xFF) << 3
	// ppc64x:-"AND" "CLRLSLWI"
	h := (v32 & 0xFFFFF) << 2
	// ppc64x:"CLRLSLDI"
	i := (v64 & 0xFFFFFFFF) << 5
	// ppc64x:-"CLRLSLDI"
	i += (v64 & 0xFFFFFFF) << 38
	// ppc64x/power9:-"CLRLSLDI"
	i += (v64 & 0xFFFF00) << 10
	// ppc64x/power9:-"SLD" "EXTSWSLI"
	j := int64(x32+32) * 8
	return f, g, h, i, j
}

func checkWidenAfterShift(v int64, u uint64) (int64, uint64) {

	// ppc64x:-".*MOVW"
	f := int32(v >> 32)
	// ppc64x:".*MOVW"
	f += int32(v >> 31)
	// ppc64x:-".*MOVH"
	g := int16(v >> 48)
	// ppc64x:".*MOVH"
	g += int16(v >> 30)
	// ppc64x:-".*MOVH"
	g += int16(f >> 16)
	// ppc64x:-".*MOVB"
	h := int8(v >> 56)
	// ppc64x:".*MOVB"
	h += int8(v >> 28)
	// ppc64x:-".*MOVB"
	h += int8(f >> 24)
	// ppc64x:".*MOVB"
	h += int8(f >> 16)
	return int64(h), uint64(g)
}

func checkShiftAndMask32(v []uint32) {
	i := 0

	// ppc64x: "RLWNM [$]24, R[0-9]+, [$]12, [$]19, R[0-9]+"
	v[i] = (v[i] & 0xFF00000) >> 8
	i++
	// ppc64x: "RLWNM [$]26, R[0-9]+, [$]22, [$]29, R[0-9]+"
	v[i] = (v[i] & 0xFF00) >> 6
	i++
	// ppc64x: "MOVW R0"
	v[i] = (v[i] & 0xFF) >> 8
	i++
	// ppc64x: "MOVW R0"
	v[i] = (v[i] & 0xF000000) >> 28
	i++
	// ppc64x: "RLWNM [$]26, R[0-9]+, [$]24, [$]31, R[0-9]+"
	v[i] = (v[i] >> 6) & 0xFF
	i++
	// ppc64x: "RLWNM [$]26, R[0-9]+, [$]12, [$]19, R[0-9]+"
	v[i] = (v[i] >> 6) & 0xFF000
	i++
	// ppc64x: "MOVW R0"
	v[i] = (v[i] >> 20) & 0xFF000
	i++
	// ppc64x: "MOVW R0"
	v[i] = (v[i] >> 24) & 0xFF00
	i++
}

func checkMergedShifts32(a [256]uint32, b [256]uint64, u uint32, v uint32) {
	// ppc64x: -"CLRLSLDI", "RLWNM [$]10, R[0-9]+, [$]22, [$]29, R[0-9]+"
	a[0] = a[uint8(v>>24)]
	// ppc64x: -"CLRLSLDI", "RLWNM [$]11, R[0-9]+, [$]21, [$]28, R[0-9]+"
	b[0] = b[uint8(v>>24)]
	// ppc64x: -"CLRLSLDI", "RLWNM [$]15, R[0-9]+, [$]21, [$]28, R[0-9]+"
	b[1] = b[(v>>20)&0xFF]
	// ppc64x: -"SLD", "RLWNM [$]10, R[0-9]+, [$]22, [$]28, R[0-9]+"
	b[2] = b[v>>25]
}

func checkMergedShifts64(a [256]uint32, b [256]uint64, c [256]byte, v uint64) {
	// ppc64x: -"CLRLSLDI", "RLWNM [$]10, R[0-9]+, [$]22, [$]29, R[0-9]+"
	a[0] = a[uint8(v>>24)]
	// ppc64x: "SRD", "CLRLSLDI", -"RLWNM"
	a[1] = a[uint8(v>>25)]
	// ppc64x: -"CLRLSLDI", "RLWNM [$]9, R[0-9]+, [$]23, [$]29, R[0-9]+"
	a[2] = a[v>>25&0x7F]
	// ppc64x: -"CLRLSLDI", "RLWNM [$]3, R[0-9]+, [$]29, [$]29, R[0-9]+"
	a[3] = a[(v>>31)&0x01]
	// ppc64x: -"CLRLSLDI", "RLWNM [$]12, R[0-9]+, [$]21, [$]28, R[0-9]+"
	b[0] = b[uint8(v>>23)]
	// ppc64x: -"CLRLSLDI", "RLWNM [$]15, R[0-9]+, [$]21, [$]28, R[0-9]+"
	b[1] = b[(v>>20)&0xFF]
	// ppc64x: "RLWNM", -"SLD"
	b[2] = b[((uint64((uint32(v) >> 21)) & 0x3f) << 4)]
	// ppc64x: -"RLWNM"
	b[3] = (b[3] << 24) & 0xFFFFFF000000
	// ppc64x: "RLWNM [$]24, R[0-9]+, [$]0, [$]7,"
	b[4] = (b[4] << 24) & 0xFF000000
	// ppc64x: "RLWNM [$]24, R[0-9]+, [$]0, [$]7,"
	b[5] = (b[5] << 24) & 0xFF00000F
	// ppc64x: -"RLWNM"
	b[6] = (b[6] << 0) & 0xFF00000F
	// ppc64x: "RLWNM [$]4, R[0-9]+, [$]28, [$]31,"
	b[7] = (b[7] >> 28) & 0xF
	// ppc64x: "RLWNM [$]11, R[0-9]+, [$]10, [$]15"
	c[0] = c[((v>>5)&0x3F)<<16]
	// ppc64x: "ANDCC [$]8064,"
	c[1] = c[((v>>7)&0x3F)<<7]
}

func checkShiftMask(a uint32, b uint64, z []uint32, y []uint64) {
	_ = y[128]
	_ = z[128]
	// ppc64x: -"MOVBZ", -"SRW", "RLWNM"
	z[0] = uint32(uint8(a >> 5))
	// ppc64x: -"MOVBZ", -"SRW", "RLWNM"
	z[1] = uint32(uint8((a >> 4) & 0x7e))
	// ppc64x: "RLWNM [$]25, R[0-9]+, [$]27, [$]29, R[0-9]+"
	z[2] = uint32(uint8(a>>7)) & 0x1c
	// ppc64x: -"MOVWZ"
	y[0] = uint64((a >> 6) & 0x1c)
	// ppc64x: -"MOVWZ"
	y[1] = uint64(uint32(b)<<6) + 1
	// ppc64x: -"MOVHZ", -"MOVWZ"
	y[2] = uint64((uint16(a) >> 9) & 0x1F)
	// ppc64x: -"MOVHZ", -"MOVWZ", -"ANDCC"
	y[3] = uint64(((uint16(a) & 0xFF0) >> 9) & 0x1F)
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
	// amd64:-"SHR" -"SHL" "ANDQ"
	u[0] = u[0] >> 5 << 5
	// amd64:-"SAR" -"SHL" "ANDQ"
	s[0] = s[0] >> 5 << 5
	// amd64:-"SHR" -"SHL" "ANDQ"
	u[1] = u[1] << 5 >> 5
}

//
// Left shift with addition.
//

func checkLeftShiftWithAddition(a int64, b int64) int64 {
	// riscv64/rva20u64: "SLLI" "ADD"
	// riscv64/rva22u64,riscv64/rva23u64: "SH1ADD"
	a = a + b<<1
	// riscv64/rva20u64: "SLLI" "ADD"
	// riscv64/rva22u64,riscv64/rva23u64: "SH2ADD"
	a = a + b<<2
	// riscv64/rva20u64: "SLLI" "ADD"
	// riscv64/rva22u64,riscv64/rva23u64: "SH3ADD"
	a = a + b<<3
	return a
}

//
// Convert and shift.
//

func rsh64Uto32U(v uint64) uint32 {
	x := uint32(v)
	// riscv64:"MOVWU"
	if x > 8 {
		// riscv64:"SRLIW" -"MOVWU" -"SLLI"
		x >>= 2
	}
	return x
}

func rsh64Uto16U(v uint64) uint16 {
	x := uint16(v)
	// riscv64:"MOVHU"
	if x > 8 {
		// riscv64:"SLLI" "SRLI"
		x >>= 2
	}
	return x
}

func rsh64Uto8U(v uint64) uint8 {
	x := uint8(v)
	// riscv64:"MOVBU"
	if x > 8 {
		// riscv64:"SLLI" "SRLI"
		x >>= 2
	}
	return x
}

func rsh64to32(v int64) int32 {
	x := int32(v)
	// riscv64:"MOVW"
	if x > 8 {
		// riscv64:"SRLIW" -"MOVW" -"SLLI"
		x >>= 2
	}
	return x
}

func rsh64to16(v int64) int16 {
	x := int16(v)
	// riscv64:"MOVH"
	if x > 8 {
		// riscv64:"SLLI" "SRLI"
		x >>= 2
	}
	return x
}

func rsh64to8(v int64) int8 {
	x := int8(v)
	// riscv64:"MOVB"
	if x > 8 {
		// riscv64:"SLLI" "SRLI"
		x >>= 2
	}
	return x
}

// We don't need to worry about shifting
// more than the type size.
// (There is still a negative shift test, but
// no shift-too-big test.)
func signedModShift(i int) int64 {
	// arm64:-"CMP" -"CSEL"
	return 1 << (i % 64)
}
