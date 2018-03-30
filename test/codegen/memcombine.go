// asmcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

import "encoding/binary"

var sink64 uint64
var sink32 uint32
var sink16 uint16

// ------------- //
//    Loading    //
// ------------- //

func load_le64(b []byte) {
	// amd64:`MOVQ\s\(.*\),`
	// s390x:`MOVDBR\s\(.*\),`
	// arm64:`MOVD\s\(R[0-9]+\),`
	// ppc64le:`MOVD\s`,-`MOV[BHW]Z`
	sink64 = binary.LittleEndian.Uint64(b)
}

func load_le64_idx(b []byte, idx int) {
	// amd64:`MOVQ\s\(.*\)\(.*\*1\),`
	// s390x:`MOVDBR\s\(.*\)\(.*\*1\),`
	// arm64:`MOVD\s\(R[0-9]+\),`
	// ppc64le:`MOVD\s`,-`MOV[BHW]Z\s`
	sink64 = binary.LittleEndian.Uint64(b[idx:])
}

func load_le32(b []byte) {
	// amd64:`MOVL\s\(.*\),`           386:`MOVL\s\(.*\),`
	// s390x:`MOVWBR\s\(.*\),`
	// arm64:`MOVWU\s\(R[0-9]+\),`
	// ppc64le:`MOVWZ\s`
	sink32 = binary.LittleEndian.Uint32(b)
}

func load_le32_idx(b []byte, idx int) {
	// amd64:`MOVL\s\(.*\)\(.*\*1\),`  386:`MOVL\s\(.*\)\(.*\*1\),`
	// s390x:`MOVWBR\s\(.*\)\(.*\*1\),`
	// arm64:`MOVWU\s\(R[0-9]+\),`
	// ppc64le:`MOVWZ\s`
	sink32 = binary.LittleEndian.Uint32(b[idx:])
}

func load_le16(b []byte) {
	// amd64:`MOVWLZX\s\(.*\),`
	// ppc64le:`MOVHZ\s`
	sink16 = binary.LittleEndian.Uint16(b)
}

func load_le16_idx(b []byte, idx int) {
	// amd64:`MOVWLZX\s\(.*\),`
	// ppc64le:`MOVHZ\s`
	sink16 = binary.LittleEndian.Uint16(b[idx:])
}

func load_be64(b []byte) {
	// amd64:`BSWAPQ`
	// s390x:`MOVD\s\(.*\),`
	// arm64:`REV`
	sink64 = binary.BigEndian.Uint64(b)
}

func load_be64_idx(b []byte, idx int) {
	// amd64:`BSWAPQ`
	// s390x:`MOVD\s\(.*\)\(.*\*1\),`
	// arm64:`REV`
	sink64 = binary.BigEndian.Uint64(b[idx:])
}

func load_be32(b []byte) {
	// amd64:`BSWAPL`
	// s390x:`MOVWZ\s\(.*\),`
	// arm64:`REVW`
	sink32 = binary.BigEndian.Uint32(b)
}

func load_be32_idx(b []byte, idx int) {
	// amd64:`BSWAPL`
	// s390x:`MOVWZ\s\(.*\)\(.*\*1\),`
	// arm64:`REVW`
	sink32 = binary.BigEndian.Uint32(b[idx:])
}

func load_be16(b []byte) {
	// amd64:`ROLW\s\$8`
	sink16 = binary.BigEndian.Uint16(b)
}

func load_be16_idx(b []byte, idx int) {
	// amd64:`ROLW\s\$8`
	sink16 = binary.BigEndian.Uint16(b[idx:])
}

// ------------- //
//    Storing    //
// ------------- //

func store_le64(b []byte) {
	// amd64:`MOVQ\s.*\(.*\)$`,-`SHR.`
	// arm64:`MOVD`,-`MOV[WBH]`
	// ppc64le:`MOVD\s`,-`MOV[BHW]\s`
	binary.LittleEndian.PutUint64(b, sink64)
}

func store_le64_idx(b []byte, idx int) {
	// amd64:`MOVQ\s.*\(.*\)\(.*\*1\)$`,-`SHR.`
	// arm64:`MOVD`,-`MOV[WBH]`
	// ppc64le:`MOVD\s`,-`MOV[BHW]\s`
	binary.LittleEndian.PutUint64(b[idx:], sink64)
}

func store_le32(b []byte) {
	// amd64:`MOVL\s`
	// arm64:`MOVW`,-`MOV[BH]`
	// ppc64le:`MOVW\s`
	binary.LittleEndian.PutUint32(b, sink32)
}

func store_le32_idx(b []byte, idx int) {
	// amd64:`MOVL\s`
	// arm64:`MOVW`,-`MOV[BH]`
	// ppc64le:`MOVW\s`
	binary.LittleEndian.PutUint32(b[idx:], sink32)
}

func store_le16(b []byte) {
	// amd64:`MOVW\s`
	// arm64:`MOVH`,-`MOVB`
	// ppc64le(DISABLED):`MOVH\s`
	binary.LittleEndian.PutUint16(b, sink16)
}

func store_le16_idx(b []byte, idx int) {
	// amd64:`MOVW\s`
	// arm64:`MOVH`,-`MOVB`
	// ppc64le(DISABLED):`MOVH\s`
	binary.LittleEndian.PutUint16(b[idx:], sink16)
}

func store_be64(b []byte) {
	// amd64:`BSWAPQ`,-`SHR.`
	// arm64:`MOVD`,`REV`,-`MOV[WBH]`
	binary.BigEndian.PutUint64(b, sink64)
}

func store_be64_idx(b []byte, idx int) {
	// amd64:`BSWAPQ`,-`SHR.`
	// arm64:`MOVD`,`REV`,-`MOV[WBH]`
	binary.BigEndian.PutUint64(b[idx:], sink64)
}

func store_be32(b []byte) {
	// amd64:`BSWAPL`,-`SHR.`
	// arm64:`MOVW`,`REVW`,-`MOV[BH]`
	binary.BigEndian.PutUint32(b, sink32)
}

func store_be32_idx(b []byte, idx int) {
	// amd64:`BSWAPL`,-`SHR.`
	// arm64:`MOVW`,`REVW`,-`MOV[BH]`
	binary.BigEndian.PutUint32(b[idx:], sink32)
}

func store_be16(b []byte) {
	// amd64:`ROLW\s\$8`,-`SHR.`
	// arm64:`MOVH`,`REV16W`,-`MOVB`
	binary.BigEndian.PutUint16(b, sink16)
}

func store_be16_idx(b []byte, idx int) {
	// amd64:`ROLW\s\$8`,-`SHR.`
	// arm64:`MOVH`,`REV16W`,-`MOVB`
	binary.BigEndian.PutUint16(b[idx:], sink16)
}

// ------------- //
//    Zeroing    //
// ------------- //

// Check that zero stores are combined into larger stores

func zero_2(b1, b2 []byte) {
	// bounds checks to guarantee safety of writes below
	_, _ = b1[1], b2[1]
	b1[0], b1[1] = 0, 0 // arm64:"MOVH\tZR",-"MOVB"
	b2[1], b2[0] = 0, 0 // arm64:"MOVH\tZR",-"MOVB"
}

func zero_4(b1, b2 []byte) {
	_, _ = b1[3], b2[3]
	b1[0], b1[1], b1[2], b1[3] = 0, 0, 0, 0 // arm64:"MOVW\tZR",-"MOVB",-"MOVH"
	b2[2], b2[3], b2[1], b2[0] = 0, 0, 0, 0 // arm64:"MOVW\tZR",-"MOVB",-"MOVH"
}

func zero_8(b []byte) {
	_ = b[7]
	b[0], b[1], b[2], b[3] = 0, 0, 0, 0
	b[4], b[5], b[6], b[7] = 0, 0, 0, 0 // arm64:"MOVD\tZR",-"MOVB",-"MOVH",-"MOVW"
}

func zero_16(b []byte) {
	_ = b[15]
	b[0], b[1], b[2], b[3] = 0, 0, 0, 0
	b[4], b[5], b[6], b[7] = 0, 0, 0, 0
	b[8], b[9], b[10], b[11] = 0, 0, 0, 0
	b[12], b[13], b[14], b[15] = 0, 0, 0, 0 // arm64:"STP",-"MOVB",-"MOVH",-"MOVW"
}
