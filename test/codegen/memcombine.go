// asmcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

import (
	"encoding/binary"
	"runtime"
)

// ------------- //
//    Loading    //
// ------------- //

func load_le64(b []byte) uint64 {
	// amd64:`MOVQ\s\(.*\),`,-`MOV[BWL]\t[^$]`,-`OR`
	// s390x:`MOVDBR\s\(.*\),`
	// arm64:`MOVD\s\(R[0-9]+\),`,-`MOV[BHW]`
	// ppc64le:`MOVD\s`,-`MOV[BHW]Z`
	// ppc64:`MOVDBR\s`,-`MOV[BHW]Z`
	return binary.LittleEndian.Uint64(b)
}

func load_le64_idx(b []byte, idx int) uint64 {
	// amd64:`MOVQ\s\(.*\)\(.*\*1\),`,-`MOV[BWL]\t[^$]`,-`OR`
	// s390x:`MOVDBR\s\(.*\)\(.*\*1\),`
	// arm64:`MOVD\s\(R[0-9]+\)\(R[0-9]+\),`,-`MOV[BHW]`
	// ppc64le:`MOVD\s`,-`MOV[BHW]Z\s`
	// ppc64:`MOVDBR\s`,-`MOV[BHW]Z\s`
	return binary.LittleEndian.Uint64(b[idx:])
}

func load_le32(b []byte) uint32 {
	// amd64:`MOVL\s\(.*\),`,-`MOV[BW]`,-`OR`
	// 386:`MOVL\s\(.*\),`,-`MOV[BW]`,-`OR`
	// s390x:`MOVWBR\s\(.*\),`
	// arm64:`MOVWU\s\(R[0-9]+\),`,-`MOV[BH]`
	// ppc64le:`MOVWZ\s`,-`MOV[BH]Z\s`
	// ppc64:`MOVWBR\s`,-`MOV[BH]Z\s`
	return binary.LittleEndian.Uint32(b)
}

func load_le32_idx(b []byte, idx int) uint32 {
	// amd64:`MOVL\s\(.*\)\(.*\*1\),`,-`MOV[BW]`,-`OR`
	// 386:`MOVL\s\(.*\)\(.*\*1\),`,-`MOV[BW]`,-`OR`
	// s390x:`MOVWBR\s\(.*\)\(.*\*1\),`
	// arm64:`MOVWU\s\(R[0-9]+\)\(R[0-9]+\),`,-`MOV[BH]`
	// ppc64le:`MOVWZ\s`,-`MOV[BH]Z\s`
	// ppc64:`MOVWBR\s`,-`MOV[BH]Z\s'
	return binary.LittleEndian.Uint32(b[idx:])
}

func load_le16(b []byte) uint16 {
	// amd64:`MOVWLZX\s\(.*\),`,-`MOVB`,-`OR`
	// ppc64le:`MOVHZ\s`,-`MOVBZ`
	// arm64:`MOVHU\s\(R[0-9]+\),`,-`MOVB`
	// s390x:`MOVHBR\s\(.*\),`
	// ppc64:`MOVHBR\s`,-`MOVBZ`
	return binary.LittleEndian.Uint16(b)
}

func load_le16_idx(b []byte, idx int) uint16 {
	// amd64:`MOVWLZX\s\(.*\),`,-`MOVB`,-`OR`
	// ppc64le:`MOVHZ\s`,-`MOVBZ`
	// ppc64:`MOVHBR\s`,-`MOVBZ`
	// arm64:`MOVHU\s\(R[0-9]+\)\(R[0-9]+\),`,-`MOVB`
	// s390x:`MOVHBR\s\(.*\)\(.*\*1\),`
	return binary.LittleEndian.Uint16(b[idx:])
}

func load_be64(b []byte) uint64 {
	// amd64/v1,amd64/v2:`BSWAPQ`,-`MOV[BWL]\t[^$]`,-`OR`
	// amd64/v3:`MOVBEQ`
	// s390x:`MOVD\s\(.*\),`
	// arm64:`REV`,`MOVD\s\(R[0-9]+\),`,-`MOV[BHW]`,-`REVW`,-`REV16W`
	// ppc64le:`MOVDBR`,-`MOV[BHW]Z`
	// ppc64:`MOVD`,-`MOV[BHW]Z`
	return binary.BigEndian.Uint64(b)
}

func load_be64_idx(b []byte, idx int) uint64 {
	// amd64/v1,amd64/v2:`BSWAPQ`,-`MOV[BWL]\t[^$]`,-`OR`
	// amd64/v3: `MOVBEQ\t\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*1\), [A-Z]+[0-9]*`
	// s390x:`MOVD\s\(.*\)\(.*\*1\),`
	// arm64:`REV`,`MOVD\s\(R[0-9]+\)\(R[0-9]+\),`,-`MOV[WHB]`,-`REVW`,-`REV16W`
	// ppc64le:`MOVDBR`,-`MOV[BHW]Z`
	// ppc64:`MOVD`,-`MOV[BHW]Z`
	return binary.BigEndian.Uint64(b[idx:])
}

func load_be32(b []byte) uint32 {
	// amd64/v1,amd64/v2:`BSWAPL`,-`MOV[BW]`,-`OR`
	// amd64/v3: `MOVBEL`
	// s390x:`MOVWZ\s\(.*\),`
	// arm64:`REVW`,`MOVWU\s\(R[0-9]+\),`,-`MOV[BH]`,-`REV16W`
	// ppc64le:`MOVWBR`,-`MOV[BH]Z`
	// ppc64:`MOVWZ`,-MOV[BH]Z`
	return binary.BigEndian.Uint32(b)
}

func load_be32_idx(b []byte, idx int) uint32 {
	// amd64/v1,amd64/v2:`BSWAPL`,-`MOV[BW]`,-`OR`
	// amd64/v3: `MOVBEL\t\([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*1\), [A-Z]+[0-9]*`
	// s390x:`MOVWZ\s\(.*\)\(.*\*1\),`
	// arm64:`REVW`,`MOVWU\s\(R[0-9]+\)\(R[0-9]+\),`,-`MOV[HB]`,-`REV16W`
	// ppc64le:`MOVWBR`,-`MOV[BH]Z`
	// ppc64:`MOVWZ`,-MOV[BH]Z`
	return binary.BigEndian.Uint32(b[idx:])
}

func load_be16(b []byte) uint16 {
	// amd64:`ROLW\s\$8`,-`MOVB`,-`OR`
	// arm64:`REV16W`,`MOVHU\s\(R[0-9]+\),`,-`MOVB`
	// ppc64le:`MOVHBR`,-`MOVBZ`
	// ppc64:`MOVHZ`,-`MOVBZ`
	// s390x:`MOVHZ\s\(.*\),`,-`OR`,-`ORW`,-`SLD`,-`SLW`
	return binary.BigEndian.Uint16(b)
}

func load_be16_idx(b []byte, idx int) uint16 {
	// amd64:`ROLW\s\$8`,-`MOVB`,-`OR`
	// arm64:`REV16W`,`MOVHU\s\(R[0-9]+\)\(R[0-9]+\),`,-`MOVB`
	// ppc64le:`MOVHBR`,-`MOVBZ`
	// ppc64:`MOVHZ`,-`MOVBZ`
	// s390x:`MOVHZ\s\(.*\)\(.*\*1\),`,-`OR`,-`ORW`,-`SLD`,-`SLW`
	return binary.BigEndian.Uint16(b[idx:])
}

func load_le_byte2_uint16(s []byte) uint16 {
	// arm64:`MOVHU\t\(R[0-9]+\)`,-`ORR`,-`MOVB`
	// 386:`MOVWLZX\s\([A-Z]+\)`,-`MOVB`,-`OR`
	// amd64:`MOVWLZX\s\([A-Z]+\)`,-`MOVB`,-`OR`
	// ppc64le:`MOVHZ\t\(R[0-9]+\)`,-`MOVBZ`
	// ppc64:`MOVHBR`,-`MOVBZ`
	return uint16(s[0]) | uint16(s[1])<<8
}

func load_le_byte2_uint16_inv(s []byte) uint16 {
	// arm64:`MOVHU\t\(R[0-9]+\)`,-`ORR`,-`MOVB`
	// 386:`MOVWLZX\s\([A-Z]+\)`,-`MOVB`,-`OR`
	// amd64:`MOVWLZX\s\([A-Z]+\)`,-`MOVB`,-`OR`
	// ppc64le:`MOVHZ\t\(R[0-9]+\)`,-`MOVBZ`
	// ppc64:`MOVHBR`,-`MOVBZ`
	return uint16(s[1])<<8 | uint16(s[0])
}

func load_le_byte4_uint32(s []byte) uint32 {
	// arm64:`MOVWU\t\(R[0-9]+\)`,-`ORR`,-`MOV[BH]`
	// 386:`MOVL\s\([A-Z]+\)`,-`MOV[BW]`,-`OR`
	// amd64:`MOVL\s\([A-Z]+\)`,-`MOV[BW]`,-`OR`
	// ppc64le:`MOVWZ\t\(R[0-9]+\)`,-`MOV[BH]Z`
	// ppc64:`MOVWBR`,-MOV[BH]Z`
	return uint32(s[0]) | uint32(s[1])<<8 | uint32(s[2])<<16 | uint32(s[3])<<24
}

func load_le_byte4_uint32_inv(s []byte) uint32 {
	// arm64:`MOVWU\t\(R[0-9]+\)`,-`ORR`,-`MOV[BH]`
	// ppc64le:`MOVWZ`,-`MOV[BH]Z`
	// ppc64:`MOVWBR`,-`MOV[BH]Z`
	return uint32(s[3])<<24 | uint32(s[2])<<16 | uint32(s[1])<<8 | uint32(s[0])
}

func load_le_byte8_uint64(s []byte) uint64 {
	// arm64:`MOVD\t\(R[0-9]+\)`,-`ORR`,-`MOV[BHW]`
	// amd64:`MOVQ\s\([A-Z]+\),\s[A-Z]+`,-`MOV[BWL]\t[^$]`,-`OR`
	// ppc64le:`MOVD\t\(R[0-9]+\)`,-`MOV[BHW]Z`
	// ppc64:`MOVDBR`,-`MOVW[WHB]Z`
	return uint64(s[0]) | uint64(s[1])<<8 | uint64(s[2])<<16 | uint64(s[3])<<24 | uint64(s[4])<<32 | uint64(s[5])<<40 | uint64(s[6])<<48 | uint64(s[7])<<56
}

func load_le_byte8_uint64_inv(s []byte) uint64 {
	// arm64:`MOVD\t\(R[0-9]+\)`,-`ORR`,-`MOV[BHW]`
	// ppc64le:`MOVD`,-`MOV[WHB]Z`
	// ppc64:`MOVDBR`,-`MOV[WHB]Z`
	return uint64(s[7])<<56 | uint64(s[6])<<48 | uint64(s[5])<<40 | uint64(s[4])<<32 | uint64(s[3])<<24 | uint64(s[2])<<16 | uint64(s[1])<<8 | uint64(s[0])
}

func load_be_byte2_uint16(s []byte) uint16 {
	// arm64:`MOVHU\t\(R[0-9]+\)`,`REV16W`,-`ORR`,-`MOVB`
	// amd64:`MOVWLZX\s\([A-Z]+\)`,`ROLW`,-`MOVB`,-`OR`
	// ppc64le:`MOVHBR\t\(R[0-9]+\)`,-`MOVBZ`
	// ppc64:`MOVHZ`,-`MOVBZ`
	return uint16(s[0])<<8 | uint16(s[1])
}

func load_be_byte2_uint16_inv(s []byte) uint16 {
	// arm64:`MOVHU\t\(R[0-9]+\)`,`REV16W`,-`ORR`,-`MOVB`
	// amd64:`MOVWLZX\s\([A-Z]+\)`,`ROLW`,-`MOVB`,-`OR`
	// ppc64le:`MOVHBR\t\(R[0-9]+\)`,-`MOVBZ`
	// ppc64:`MOVHZ`,-`MOVBZ`
	return uint16(s[1]) | uint16(s[0])<<8
}

func load_be_byte4_uint32(s []byte) uint32 {
	// arm64:`MOVWU\t\(R[0-9]+\)`,`REVW`,-`ORR`,-`REV16W`,-`MOV[BH]`
	// ppc64le:`MOVWBR`,-`MOV[HB]Z`
	// ppc64:`MOVWZ`,-`MOV[HB]Z`
	return uint32(s[0])<<24 | uint32(s[1])<<16 | uint32(s[2])<<8 | uint32(s[3])
}

func load_be_byte4_uint32_inv(s []byte) uint32 {
	// arm64:`MOVWU\t\(R[0-9]+\)`,`REVW`,-`ORR`,-`REV16W`,-`MOV[BH]`
	// amd64/v1,amd64/v2:`MOVL\s\([A-Z]+\)`,`BSWAPL`,-`MOV[BW]`,-`OR`
	// amd64/v3: `MOVBEL`
	// ppc64le:`MOVWBR`,-`MOV[HB]Z`
	// ppc64:`MOVWZ`,-`MOV[HB]Z`
	return uint32(s[3]) | uint32(s[2])<<8 | uint32(s[1])<<16 | uint32(s[0])<<24
}

func load_be_byte8_uint64(s []byte) uint64 {
	// arm64:`MOVD\t\(R[0-9]+\)`,`REV`,-`ORR`,-`REVW`,-`REV16W`,-`MOV[BHW]`
	// ppc64le:`MOVDBR\t\(R[0-9]+\)`,-`MOV[BHW]Z`
	// ppc64:`MOVD`,-`MOV[WHB]Z`
	return uint64(s[0])<<56 | uint64(s[1])<<48 | uint64(s[2])<<40 | uint64(s[3])<<32 | uint64(s[4])<<24 | uint64(s[5])<<16 | uint64(s[6])<<8 | uint64(s[7])
}

func load_be_byte8_uint64_inv(s []byte) uint64 {
	// arm64:`MOVD\t\(R[0-9]+\)`,`REV`,-`ORR`,-`REVW`,-`REV16W`,-`MOV[BHW]`
	// amd64/v1,amd64/v2:`MOVQ\s\([A-Z]+\),\s[A-Z]+`,`BSWAPQ`,-`MOV[BWL]\t[^$]`,-`OR`
	// amd64/v3: `MOVBEQ`
	// ppc64le:`MOVDBR\t\(R[0-9]+\)`,-`MOV[BHW]Z`
	// ppc64:`MOVD`,-`MOV[BHW]Z`
	return uint64(s[7]) | uint64(s[6])<<8 | uint64(s[5])<<16 | uint64(s[4])<<24 | uint64(s[3])<<32 | uint64(s[2])<<40 | uint64(s[1])<<48 | uint64(s[0])<<56
}

func load_le_byte2_uint16_idx(s []byte, idx int) uint16 {
	// arm64:`MOVHU\s\(R[0-9]+\)\(R[0-9]+\)`,-`ORR`,-`MOVB`
	// 386:`MOVWLZX\s\([A-Z]+\)\([A-Z]+`,-`ORL`,-`MOVB`
	// amd64:`MOVWLZX\s\([A-Z]+\)\([A-Z]+`,-`MOVB`,-`OR`
	// ppc64le:`MOVHZ`,-`MOVBZ`
	// ppc64:`MOVHBR`,-`MOVBZ`
	return uint16(s[idx]) | uint16(s[idx+1])<<8
}

func load_le_byte2_uint16_idx_inv(s []byte, idx int) uint16 {
	// arm64:`MOVHU\s\(R[0-9]+\)\(R[0-9]+\)`,-`ORR`,-`MOVB`
	// 386:`MOVWLZX\s\([A-Z]+\)\([A-Z]+`,-`ORL`,-`MOVB`
	// amd64:`MOVWLZX\s\([A-Z]+\)\([A-Z]+`,-`MOVB`,-`OR`
	// ppc64le:`MOVHZ`,-`MOVBZ`
	// ppc64:`MOVHBR`,-`MOVBZ`
	return uint16(s[idx+1])<<8 | uint16(s[idx])
}

func load_le_byte4_uint32_idx(s []byte, idx int) uint32 {
	// arm64:`MOVWU\s\(R[0-9]+\)\(R[0-9]+\)`,-`ORR`,-`MOV[BH]`
	// amd64:`MOVL\s\([A-Z]+\)\([A-Z]+`,-`MOV[BW]`,-`OR`
	return uint32(s[idx]) | uint32(s[idx+1])<<8 | uint32(s[idx+2])<<16 | uint32(s[idx+3])<<24
}

func load_le_byte4_uint32_idx_inv(s []byte, idx int) uint32 {
	// arm64:`MOVWU\s\(R[0-9]+\)\(R[0-9]+\)`,-`ORR`,-`MOV[BH]`
	return uint32(s[idx+3])<<24 | uint32(s[idx+2])<<16 | uint32(s[idx+1])<<8 | uint32(s[idx])
}

func load_le_byte8_uint64_idx(s []byte, idx int) uint64 {
	// arm64:`MOVD\s\(R[0-9]+\)\(R[0-9]+\)`,-`ORR`,-`MOV[BHW]`
	// amd64:`MOVQ\s\([A-Z]+\)\([A-Z]+`,-`MOV[BWL]`,-`OR`
	return uint64(s[idx]) | uint64(s[idx+1])<<8 | uint64(s[idx+2])<<16 | uint64(s[idx+3])<<24 | uint64(s[idx+4])<<32 | uint64(s[idx+5])<<40 | uint64(s[idx+6])<<48 | uint64(s[idx+7])<<56
}

func load_le_byte8_uint64_idx_inv(s []byte, idx int) uint64 {
	// arm64:`MOVD\s\(R[0-9]+\)\(R[0-9]+\)`,-`ORR`,-`MOV[BHW]`
	return uint64(s[idx+7])<<56 | uint64(s[idx+6])<<48 | uint64(s[idx+5])<<40 | uint64(s[idx+4])<<32 | uint64(s[idx+3])<<24 | uint64(s[idx+2])<<16 | uint64(s[idx+1])<<8 | uint64(s[idx])
}

func load_be_byte2_uint16_idx(s []byte, idx int) uint16 {
	// arm64:`MOVHU\s\(R[0-9]+\)\(R[0-9]+\)`,`REV16W`,-`ORR`,-`MOVB`
	// amd64:`MOVWLZX\s\([A-Z]+\)\([A-Z]+`,-`MOVB`,-`OR`
	return uint16(s[idx])<<8 | uint16(s[idx+1])
}

func load_be_byte2_uint16_idx_inv(s []byte, idx int) uint16 {
	// arm64:`MOVHU\s\(R[0-9]+\)\(R[0-9]+\)`,`REV16W`,-`ORR`,-`MOVB`
	// amd64:`MOVWLZX\s\([A-Z]+\)\([A-Z]+`,-`MOVB`,-`OR`
	return uint16(s[idx+1]) | uint16(s[idx])<<8
}

func load_be_byte4_uint32_idx(s []byte, idx int) uint32 {
	// arm64:`MOVWU\s\(R[0-9]+\)\(R[0-9]+\)`,`REVW`,-`ORR`,-`MOV[BH]`,-`REV16W`
	return uint32(s[idx])<<24 | uint32(s[idx+1])<<16 | uint32(s[idx+2])<<8 | uint32(s[idx+3])
}

func load_be_byte8_uint64_idx(s []byte, idx int) uint64 {
	// arm64:`MOVD\s\(R[0-9]+\)\(R[0-9]+\)`,`REV`,-`ORR`,-`MOV[BHW]`,-`REVW`,-`REV16W`
	return uint64(s[idx])<<56 | uint64(s[idx+1])<<48 | uint64(s[idx+2])<<40 | uint64(s[idx+3])<<32 | uint64(s[idx+4])<<24 | uint64(s[idx+5])<<16 | uint64(s[idx+6])<<8 | uint64(s[idx+7])
}

func load_le_byte2_uint16_idx2(s []byte, idx int) uint16 {
	// arm64:`MOVHU\s\(R[0-9]+\)\(R[0-9]+<<1\)`,-`ORR`,-`MOVB`
	return uint16(s[idx<<1]) | uint16(s[(idx<<1)+1])<<8
}

func load_le_byte2_uint16_idx2_inv(s []byte, idx int) uint16 {
	// arm64:`MOVHU\s\(R[0-9]+\)\(R[0-9]+<<1\)`,-`ORR`,-`MOVB`
	return uint16(s[(idx<<1)+1])<<8 | uint16(s[idx<<1])
}

func load_le_byte4_uint32_idx4(s []byte, idx int) uint32 {
	// arm64:`MOVWU\s\(R[0-9]+\)\(R[0-9]+<<2\)`,-`ORR`,-`MOV[BH]`
	return uint32(s[idx<<2]) | uint32(s[(idx<<2)+1])<<8 | uint32(s[(idx<<2)+2])<<16 | uint32(s[(idx<<2)+3])<<24
}

func load_le_byte4_uint32_idx4_inv(s []byte, idx int) uint32 {
	// arm64:`MOVWU\s\(R[0-9]+\)\(R[0-9]+<<2\)`,-`ORR`,-`MOV[BH]`
	return uint32(s[(idx<<2)+3])<<24 | uint32(s[(idx<<2)+2])<<16 | uint32(s[(idx<<2)+1])<<8 | uint32(s[idx<<2])
}

func load_le_byte8_uint64_idx8(s []byte, idx int) uint64 {
	// arm64:`MOVD\s\(R[0-9]+\)\(R[0-9]+<<3\)`,-`ORR`,-`MOV[BHW]`
	return uint64(s[idx<<3]) | uint64(s[(idx<<3)+1])<<8 | uint64(s[(idx<<3)+2])<<16 | uint64(s[(idx<<3)+3])<<24 | uint64(s[(idx<<3)+4])<<32 | uint64(s[(idx<<3)+5])<<40 | uint64(s[(idx<<3)+6])<<48 | uint64(s[(idx<<3)+7])<<56
}

func load_le_byte8_uint64_idx8_inv(s []byte, idx int) uint64 {
	// arm64:`MOVD\s\(R[0-9]+\)\(R[0-9]+<<3\)`,-`ORR`,-`MOV[BHW]`
	return uint64(s[(idx<<3)+7])<<56 | uint64(s[(idx<<3)+6])<<48 | uint64(s[(idx<<3)+5])<<40 | uint64(s[(idx<<3)+4])<<32 | uint64(s[(idx<<3)+3])<<24 | uint64(s[(idx<<3)+2])<<16 | uint64(s[(idx<<3)+1])<<8 | uint64(s[idx<<3])
}

func load_be_byte2_uint16_idx2(s []byte, idx int) uint16 {
	// arm64:`MOVHU\s\(R[0-9]+\)\(R[0-9]+<<1\)`,`REV16W`,-`ORR`,-`MOVB`
	return uint16(s[idx<<1])<<8 | uint16(s[(idx<<1)+1])
}

func load_be_byte2_uint16_idx2_inv(s []byte, idx int) uint16 {
	// arm64:`MOVHU\s\(R[0-9]+\)\(R[0-9]+<<1\)`,`REV16W`,-`ORR`,-`MOVB`
	return uint16(s[(idx<<1)+1]) | uint16(s[idx<<1])<<8
}

func load_be_byte4_uint32_idx4(s []byte, idx int) uint32 {
	// arm64:`MOVWU\s\(R[0-9]+\)\(R[0-9]+<<2\)`,`REVW`,-`ORR`,-`MOV[BH]`,-`REV16W`
	return uint32(s[idx<<2])<<24 | uint32(s[(idx<<2)+1])<<16 | uint32(s[(idx<<2)+2])<<8 | uint32(s[(idx<<2)+3])
}

func load_be_byte8_uint64_idx8(s []byte, idx int) uint64 {
	// arm64:`MOVD\s\(R[0-9]+\)\(R[0-9]+<<3\)`,`REV`,-`ORR`,-`MOV[BHW]`,-`REVW`,-`REV16W`
	return uint64(s[idx<<3])<<56 | uint64(s[(idx<<3)+1])<<48 | uint64(s[(idx<<3)+2])<<40 | uint64(s[(idx<<3)+3])<<32 | uint64(s[(idx<<3)+4])<<24 | uint64(s[(idx<<3)+5])<<16 | uint64(s[(idx<<3)+6])<<8 | uint64(s[(idx<<3)+7])
}

// Some tougher cases for the memcombine pass.

func reassoc_load_uint32(b []byte) uint32 {
	// amd64:`MOVL\s\([A-Z]+\)`,-`MOV[BW]`,-`OR`
	return (uint32(b[0]) | uint32(b[1])<<8) | (uint32(b[2])<<16 | uint32(b[3])<<24)
}

func extrashift_load_uint32(b []byte) uint32 {
	// amd64:`MOVL\s\([A-Z]+\)`,`SHLL\s[$]2`,-`MOV[BW]`,-`OR`
	return uint32(b[0])<<2 | uint32(b[1])<<10 | uint32(b[2])<<18 | uint32(b[3])<<26
}

func outoforder_load_uint32(b []byte) uint32 {
	// amd64:`MOVL\s\([A-Z]+\)`,-`MOV[BW]`,-`OR`
	return uint32(b[0]) | uint32(b[2])<<16 | uint32(b[1])<<8 | uint32(b[3])<<24
}

func extraOr_load_uint32(b []byte, x, y uint32) uint32 {
	// amd64:`ORL\s\([A-Z]+\)`,-`MOV[BW]`
	return x | binary.LittleEndian.Uint32(b) | y
	// TODO: Note that
	//   x | uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24 | y
	// doesn't work because it associates in a way that memcombine can't detect it.
}

// Check load combining across function calls.

func fcall_byte(a [2]byte) [2]byte {
	return fcall_byte(fcall_byte(a)) // amd64:`MOVW`
}

func fcall_uint16(a [2]uint16) [2]uint16 {
	return fcall_uint16(fcall_uint16(a)) // amd64:`MOVL`
}

func fcall_uint32(a [2]uint32) [2]uint32 {
	return fcall_uint32(fcall_uint32(a)) // amd64:`MOVQ`
}

// We want to merge load+op in the first function, but not in the
// second. See Issue 19595.
func load_op_merge(p, q *int) {
	x := *p // amd64:`ADDQ\t\(`
	*q += x // The combined nilcheck and load would normally have this line number, but we want that combined operation to have the line number of the nil check instead (see #33724).
}
func load_op_no_merge(p, q *int) {
	x := *p
	for i := 0; i < 10; i++ {
		*q += x // amd64:`ADDQ\t[A-Z]`
	}
}

// Make sure offsets are folded into loads and stores.
func offsets_fold(_, a [20]byte) (b [20]byte) {
	// arm64:`MOVD\tcommand-line-arguments\.a\+[0-9]+\(FP\), R[0-9]+`,`MOVD\tR[0-9]+, command-line-arguments\.b\+[0-9]+\(FP\)`
	b = a
	return
}

// Make sure we don't put pointers in SSE registers across safe
// points.

func safe_point(p, q *[2]*int) {
	a, b := p[0], p[1] // amd64:-`MOVUPS`
	runtime.GC()
	q[0], q[1] = a, b // amd64:-`MOVUPS`
}

// ------------- //
//    Storing    //
// ------------- //

func store_le64(b []byte, x uint64) {
	// amd64:`MOVQ\s.*\(.*\)$`,-`SHR.`
	// arm64:`MOVD`,-`MOV[WBH]`
	// ppc64le:`MOVD\s`,-`MOV[BHW]\s`
	// ppc64:`MOVDBR`,-MOVB\s`
	// s390x:`MOVDBR\s.*\(.*\)$`
	binary.LittleEndian.PutUint64(b, x)
}

func store_le64_idx(b []byte, x uint64, idx int) {
	// amd64:`MOVQ\s.*\(.*\)\(.*\*1\)$`,-`SHR.`
	// arm64:`MOVD\sR[0-9]+,\s\(R[0-9]+\)\(R[0-9]+\)`,-`MOV[BHW]`
	// ppc64le:`MOVD\s`,-`MOV[BHW]\s`
	// ppc64:`MOVDBR`,-`MOVBZ`
	// s390x:`MOVDBR\s.*\(.*\)\(.*\*1\)$`
	binary.LittleEndian.PutUint64(b[idx:], x)
}

func store_le64_idx2(dst []byte, d, length, offset int) []byte {
	a := dst[d : d+length]
	b := dst[d-offset:]
	// amd64:`MOVQ\s.*\(.*\)\(.*\*1\)$`,-`SHR.`
	binary.LittleEndian.PutUint64(a, binary.LittleEndian.Uint64(b))
	return dst
}

func store_le64_idx_const(b []byte, idx int) {
	// amd64:`MOVQ\s\$123, \(.*\)\(.*\*1\)$`
	binary.LittleEndian.PutUint64(b[idx:], 123)
}

func store_le64_load(b []byte, x *[8]byte) {
	_ = b[8]
	// amd64:-`MOV[BWL]`
	// arm64:-`MOV[BWH]`
	// ppc64le:`MOVD\s`,-`MOV[BWH]Z`
	// ppc64:`MOVDBR`
	// s390x:-`MOVB`,-`MOV[WH]BR`
	binary.LittleEndian.PutUint64(b, binary.LittleEndian.Uint64(x[:]))
}

func store_le32(b []byte, x uint32) {
	// amd64:`MOVL\s`
	// arm64:`MOVW`,-`MOV[BH]`
	// ppc64le:`MOVW\s`
	// ppc64:`MOVWBR`
	// s390x:`MOVWBR\s.*\(.*\)$`
	binary.LittleEndian.PutUint32(b, x)
}

func store_le32_idx(b []byte, x uint32, idx int) {
	// amd64:`MOVL\s`
	// arm64:`MOVW\sR[0-9]+,\s\(R[0-9]+\)\(R[0-9]+\)`,-`MOV[BH]`
	// ppc64le:`MOVW\s`
	// ppc64:`MOVWBR`
	// s390x:`MOVWBR\s.*\(.*\)\(.*\*1\)$`
	binary.LittleEndian.PutUint32(b[idx:], x)
}

func store_le32_idx_const(b []byte, idx int) {
	// amd64:`MOVL\s\$123, \(.*\)\(.*\*1\)$`
	// ppc64x:`MOVW\s`,-MOV[HB]`
	binary.LittleEndian.PutUint32(b[idx:], 123)
}

func store_le16(b []byte, x uint16) {
	// amd64:`MOVW\s`
	// arm64:`MOVH`,-`MOVB`
	// ppc64le:`MOVH\s`
	// ppc64:`MOVHBR`
	// s390x:`MOVHBR\s.*\(.*\)$`
	binary.LittleEndian.PutUint16(b, x)
}

func store_le16_idx(b []byte, x uint16, idx int) {
	// amd64:`MOVW\s`
	// arm64:`MOVH\sR[0-9]+,\s\(R[0-9]+\)\(R[0-9]+\)`,-`MOVB`
	// ppc64le:`MOVH\s`
	// ppc64:`MOVHBR\s`
	// s390x:`MOVHBR\s.*\(.*\)\(.*\*1\)$`
	binary.LittleEndian.PutUint16(b[idx:], x)
}

func store_le16_idx_const(b []byte, idx int) {
	// amd64:`MOVW\s\$123, \(.*\)\(.*\*1\)$`
	// ppc64x:`MOVH\s`
	binary.LittleEndian.PutUint16(b[idx:], 123)
}

func store_be64(b []byte, x uint64) {
	// amd64/v1,amd64/v2:`BSWAPQ`,-`SHR.`
	// amd64/v3: `MOVBEQ`
	// arm64:`MOVD`,`REV`,-`MOV[WBH]`,-`REVW`,-`REV16W`
	// ppc64le:`MOVDBR`
	// ppc64:`MOVD\s`
	// s390x:`MOVD\s.*\(.*\)$`,-`SRW\s`,-`SRD\s`
	binary.BigEndian.PutUint64(b, x)
}

func store_be64_idx(b []byte, x uint64, idx int) {
	// amd64/v1,amd64/v2:`BSWAPQ`,-`SHR.`
	// amd64/v3:`MOVBEQ\t[A-Z]+[0-9]*, \([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*1\)`
	// arm64:`REV`,`MOVD\sR[0-9]+,\s\(R[0-9]+\)\(R[0-9]+\)`,-`MOV[BHW]`,-`REV16W`,-`REVW`
	// ppc64le:`MOVDBR`
	// ppc64:`MOVD\s`
	// s390x:`MOVD\s.*\(.*\)\(.*\*1\)$`,-`SRW\s`,-`SRD\s`
	binary.BigEndian.PutUint64(b[idx:], x)
}

func store_be32(b []byte, x uint32) {
	// amd64/v1,amd64/v2:`BSWAPL`,-`SHR.`
	// amd64/v3:`MOVBEL`
	// arm64:`MOVW`,`REVW`,-`MOV[BH]`,-`REV16W`
	// ppc64le:`MOVWBR`
	// ppc64:`MOVW\s`
	// s390x:`MOVW\s.*\(.*\)$`,-`SRW\s`,-`SRD\s`
	binary.BigEndian.PutUint32(b, x)
}

func store_be64_load(b, x *[8]byte) {
	// arm64:-`REV`
	// amd64:-`BSWAPQ`
	binary.BigEndian.PutUint64(b[:], binary.BigEndian.Uint64(x[:]))
}

func store_be32_load(b, x *[8]byte) {
	// arm64:-`REVW`
	// amd64:-`BSWAPL`
	binary.BigEndian.PutUint32(b[:], binary.BigEndian.Uint32(x[:]))
}

func store_be32_idx(b []byte, x uint32, idx int) {
	// amd64/v1,amd64/v2:`BSWAPL`,-`SHR.`
	// amd64/v3:`MOVBEL\t[A-Z]+[0-9]*, \([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*1\)`
	// arm64:`REVW`,`MOVW\sR[0-9]+,\s\(R[0-9]+\)\(R[0-9]+\)`,-`MOV[BH]`,-`REV16W`
	// ppc64le:`MOVWBR`
	// ppc64:`MOVW\s`
	// s390x:`MOVW\s.*\(.*\)\(.*\*1\)$`,-`SRW\s`,-`SRD\s`
	binary.BigEndian.PutUint32(b[idx:], x)
}

func store_be16(b []byte, x uint16) {
	// amd64/v1,amd64/v2:`ROLW\s\$8`,-`SHR.`
	// amd64/v3:`MOVBEW`,-`ROLW`
	// arm64:`MOVH`,`REV16W`,-`MOVB`
	// ppc64le:`MOVHBR`
	// ppc64:`MOVH\s`
	// s390x:`MOVH\s.*\(.*\)$`,-`SRW\s`,-`SRD\s`
	binary.BigEndian.PutUint16(b, x)
}

func store_be16_idx(b []byte, x uint16, idx int) {
	// amd64/v1,amd64/v2:`ROLW\s\$8`,-`SHR.`
	// amd64/v3:`MOVBEW\t[A-Z]+[0-9]*, \([A-Z]+[0-9]*\)\([A-Z]+[0-9]*\*1\)`
	// arm64:`MOVH\sR[0-9]+,\s\(R[0-9]+\)\(R[0-9]+\)`,`REV16W`,-`MOVB`
	// ppc64le:`MOVHBR`
	// ppc64:`MOVH\s`
	// s390x:`MOVH\s.*\(.*\)\(.*\*1\)$`,-`SRW\s`,-`SRD\s`
	binary.BigEndian.PutUint16(b[idx:], x)
}

func store_le_byte_2(b []byte, val uint16) {
	_ = b[2]
	// arm64:`MOVH\sR[0-9]+,\s1\(R[0-9]+\)`,-`MOVB`
	// 386:`MOVW\s[A-Z]+,\s1\([A-Z]+\)`,-`MOVB`
	// amd64:`MOVW\s[A-Z]+,\s1\([A-Z]+\)`,-`MOVB`
	// ppc64le:`MOVH\s`,-`MOVB`
	// ppc64:`MOVHBR`,-`MOVB`
	b[1], b[2] = byte(val), byte(val>>8)
}

func store_le_byte_2_inv(b []byte, val uint16) {
	_ = b[2]
	// 386:`MOVW\s[A-Z]+,\s1\([A-Z]+\)`,-`MOVB`
	// amd64:`MOVW\s[A-Z]+,\s1\([A-Z]+\)`,-`MOVB`
	// ppc64le:`MOVH\s`,-`MOVB`
	// ppc64:`MOVHBR`,-`MOVB`
	b[2], b[1] = byte(val>>8), byte(val)
}

func store_le_byte_4(b []byte, val uint32) {
	_ = b[4]
	// arm64:`MOVW\sR[0-9]+,\s1\(R[0-9]+\)`,-`MOVB`,-`MOVH`
	// 386:`MOVL\s[A-Z]+,\s1\([A-Z]+\)`,-`MOVB`,-`MOVW`
	// amd64:`MOVL\s[A-Z]+,\s1\([A-Z]+\)`,-`MOVB`,-`MOVW`
	// ppc64le:`MOVW\s`
	// ppc64:`MOVWBR\s`
	b[1], b[2], b[3], b[4] = byte(val), byte(val>>8), byte(val>>16), byte(val>>24)
}

func store_le_byte_8(b []byte, val uint64) {
	_ = b[8]
	// arm64:`MOVD\sR[0-9]+,\s1\(R[0-9]+\)`,-`MOVB`,-`MOVH`,-`MOVW`
	// amd64:`MOVQ\s[A-Z]+,\s1\([A-Z]+\)`,-`MOVB`,-`MOVW`,-`MOVL`
	// ppc64le:`MOVD\s`,-`MOVW`
	// ppc64:`MOVDBR\s`
	b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8] = byte(val), byte(val>>8), byte(val>>16), byte(val>>24), byte(val>>32), byte(val>>40), byte(val>>48), byte(val>>56)
}

func store_be_byte_2(b []byte, val uint16) {
	_ = b[2]
	// arm64:`REV16W`,`MOVH\sR[0-9]+,\s1\(R[0-9]+\)`,-`MOVB`
	// amd64/v1,amd64/v2:`MOVW\s[A-Z]+,\s1\([A-Z]+\)`,-`MOVB`
	// amd64/v3: `MOVBEW`
	// ppc64le:`MOVHBR`
	// ppc64:`MOVH\s`
	b[1], b[2] = byte(val>>8), byte(val)
}

func store_be_byte_4(b []byte, val uint32) {
	_ = b[4]
	// arm64:`REVW`,`MOVW\sR[0-9]+,\s1\(R[0-9]+\)`,-`MOVB`,-`MOVH`,-`REV16W`
	// amd64/v1,amd64/v2:`MOVL\s[A-Z]+,\s1\([A-Z]+\)`,-`MOVB`,-`MOVW`
	// amd64/v3:`MOVBEL\s[A-Z]+,\s1\([A-Z]+\)`
	// ppc64le:`MOVWBR`
	// ppc64:`MOVW\s`
	b[1], b[2], b[3], b[4] = byte(val>>24), byte(val>>16), byte(val>>8), byte(val)
}

func store_be_byte_8(b []byte, val uint64) {
	_ = b[8]
	// arm64:`REV`,`MOVD\sR[0-9]+,\s1\(R[0-9]+\)`,-`MOVB`,-`MOVH`,-`MOVW`,-`REV16W`,-`REVW`
	// amd64/v1,amd64/v2:`MOVQ\s[A-Z]+,\s1\([A-Z]+\)`,-`MOVB`,-`MOVW`,-`MOVL`
	// amd64/v3:`MOVBEQ\s[A-Z]+,\s1\([A-Z]+\)`, -`MOVBEL`
	// ppc64le:`MOVDBR`
	// ppc64:`MOVD`
	b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8] = byte(val>>56), byte(val>>48), byte(val>>40), byte(val>>32), byte(val>>24), byte(val>>16), byte(val>>8), byte(val)
}

func store_le_byte_2_idx(b []byte, idx int, val uint16) {
	_, _ = b[idx+0], b[idx+1]
	// arm64:`MOVH\sR[0-9]+,\s\(R[0-9]+\)\(R[0-9]+\)`,-`MOVB`
	// 386:`MOVW\s[A-Z]+,\s\([A-Z]+\)\([A-Z]+`,-`MOVB`
	// ppc64le:`MOVH\s`
	// ppc64:`MOVHBR`
	b[idx+1], b[idx+0] = byte(val>>8), byte(val)
}

func store_le_byte_2_idx_inv(b []byte, idx int, val uint16) {
	_, _ = b[idx+0], b[idx+1]
	// 386:`MOVW\s[A-Z]+,\s\([A-Z]+\)\([A-Z]+`,-`MOVB`
	// ppc64le:`MOVH\s`
	// ppc64:`MOVHBR`
	b[idx+0], b[idx+1] = byte(val), byte(val>>8)
}

func store_le_byte_4_idx(b []byte, idx int, val uint32) {
	_, _, _, _ = b[idx+0], b[idx+1], b[idx+2], b[idx+3]
	// arm64:`MOVW\sR[0-9]+,\s\(R[0-9]+\)\(R[0-9]+\)`,-`MOVB`,-`MOVH`
	// ppc64le:`MOVW\s`
	// ppc64:`MOVWBR`
	b[idx+3], b[idx+2], b[idx+1], b[idx+0] = byte(val>>24), byte(val>>16), byte(val>>8), byte(val)
}

func store_be_byte_2_idx(b []byte, idx int, val uint16) {
	_, _ = b[idx+0], b[idx+1]
	// arm64:`REV16W`,`MOVH\sR[0-9]+,\s\(R[0-9]+\)\(R[0-9]+\)`,-`MOVB`
	// ppc64le:`MOVHBR`
	// ppc64:`MOVH\s`
	b[idx+0], b[idx+1] = byte(val>>8), byte(val)
}

func store_be_byte_4_idx(b []byte, idx int, val uint32) {
	_, _, _, _ = b[idx+0], b[idx+1], b[idx+2], b[idx+3]
	// arm64:`REVW`,`MOVW\sR[0-9]+,\s\(R[0-9]+\)\(R[0-9]+\)`,-`MOVB`,-`MOVH`,-`REV16W`
	// ppc64le:`MOVWBR`
	// ppc64:`MOVW\s`
	b[idx+0], b[idx+1], b[idx+2], b[idx+3] = byte(val>>24), byte(val>>16), byte(val>>8), byte(val)
}

func store_be_byte_2_idx2(b []byte, idx int, val uint16) {
	_, _ = b[(idx<<1)+0], b[(idx<<1)+1]
	// arm64:`REV16W`,`MOVH\sR[0-9]+,\s\(R[0-9]+\)\(R[0-9]+<<1\)`,-`MOVB`
	// ppc64le:`MOVHBR`
	// ppc64:`MOVH\s`
	b[(idx<<1)+0], b[(idx<<1)+1] = byte(val>>8), byte(val)
}

func store_le_byte_2_idx2(b []byte, idx int, val uint16) {
	_, _ = b[(idx<<1)+0], b[(idx<<1)+1]
	// arm64:`MOVH\sR[0-9]+,\s\(R[0-9]+\)\(R[0-9]+<<1\)`,-`MOVB`
	// ppc64le:`MOVH\s`
	// ppc64:`MOVHBR`
	b[(idx<<1)+1], b[(idx<<1)+0] = byte(val>>8), byte(val)
}

func store_be_byte_4_idx4(b []byte, idx int, val uint32) {
	_, _, _, _ = b[(idx<<2)+0], b[(idx<<2)+1], b[(idx<<2)+2], b[(idx<<2)+3]
	// arm64:`REVW`,`MOVW\sR[0-9]+,\s\(R[0-9]+\)\(R[0-9]+<<2\)`,-`MOVB`,-`MOVH`,-`REV16W`
	// ppc64le:`MOVWBR`
	// ppc64:`MOVW\s`
	b[(idx<<2)+0], b[(idx<<2)+1], b[(idx<<2)+2], b[(idx<<2)+3] = byte(val>>24), byte(val>>16), byte(val>>8), byte(val)
}

func store_le_byte_4_idx4_inv(b []byte, idx int, val uint32) {
	_, _, _, _ = b[(idx<<2)+0], b[(idx<<2)+1], b[(idx<<2)+2], b[(idx<<2)+3]
	// arm64:`MOVW\sR[0-9]+,\s\(R[0-9]+\)\(R[0-9]+<<2\)`,-`MOVB`,-`MOVH`
	// ppc64le:`MOVW\s`
	// ppc64:`MOVWBR`
	b[(idx<<2)+3], b[(idx<<2)+2], b[(idx<<2)+1], b[(idx<<2)+0] = byte(val>>24), byte(val>>16), byte(val>>8), byte(val)
}

// ------------- //
//    Zeroing    //
// ------------- //

// Check that zero stores are combined into larger stores

func zero_byte_2(b1, b2 []byte) {
	// bounds checks to guarantee safety of writes below
	_, _ = b1[1], b2[1]
	// arm64:"MOVH\tZR",-"MOVB"
	// amd64:`MOVW\s[$]0,\s\([A-Z]+\)`
	// 386:`MOVW\s[$]0,\s\([A-Z]+\)`
	// ppc64x:`MOVH\s`
	b1[0], b1[1] = 0, 0
	// arm64:"MOVH\tZR",-"MOVB"
	// 386:`MOVW\s[$]0,\s\([A-Z]+\)`
	// amd64:`MOVW\s[$]0,\s\([A-Z]+\)`
	// ppc64x:`MOVH`
	b2[1], b2[0] = 0, 0
}

func zero_byte_4(b1, b2 []byte) {
	_, _ = b1[3], b2[3]
	// arm64:"MOVW\tZR",-"MOVB",-"MOVH"
	// amd64:`MOVL\s[$]0,\s\([A-Z]+\)`
	// 386:`MOVL\s[$]0,\s\([A-Z]+\)`
	// ppc64x:`MOVW\s`
	b1[0], b1[1], b1[2], b1[3] = 0, 0, 0, 0
	// arm64:"MOVW\tZR",-"MOVB",-"MOVH"
	// ppc64x:`MOVW\s`
	b2[2], b2[3], b2[1], b2[0] = 0, 0, 0, 0
}

func zero_byte_8(b []byte) {
	_ = b[7]
	b[0], b[1], b[2], b[3] = 0, 0, 0, 0 // arm64:"MOVD\tZR",-"MOVB",-"MOVH",-"MOVW"
	b[4], b[5], b[6], b[7] = 0, 0, 0, 0
}

func zero_byte_16(b []byte) {
	_ = b[15]
	b[0], b[1], b[2], b[3] = 0, 0, 0, 0 // arm64:"STP",-"MOVB",-"MOVH",-"MOVW"
	b[4], b[5], b[6], b[7] = 0, 0, 0, 0
	b[8], b[9], b[10], b[11] = 0, 0, 0, 0
	b[12], b[13], b[14], b[15] = 0, 0, 0, 0
}

func zero_byte_30(a *[30]byte) {
	*a = [30]byte{} // arm64:"STP",-"MOVB",-"MOVH",-"MOVW"
}

func zero_byte_39(a *[39]byte) {
	*a = [39]byte{} // arm64:"MOVD",-"MOVB",-"MOVH",-"MOVW"
}

func zero_byte_2_idx(b []byte, idx int) {
	_, _ = b[idx+0], b[idx+1]
	// arm64:`MOVH\sZR,\s\(R[0-9]+\)\(R[0-9]+\)`,-`MOVB`
	// ppc64x:`MOVH\s`
	b[idx+0], b[idx+1] = 0, 0
}

func zero_byte_2_idx2(b []byte, idx int) {
	_, _ = b[(idx<<1)+0], b[(idx<<1)+1]
	// arm64:`MOVH\sZR,\s\(R[0-9]+\)\(R[0-9]+<<1\)`,-`MOVB`
	// ppc64x:`MOVH\s`
	b[(idx<<1)+0], b[(idx<<1)+1] = 0, 0
}

func zero_uint16_2(h1, h2 []uint16) {
	_, _ = h1[1], h2[1]
	// arm64:"MOVW\tZR",-"MOVB",-"MOVH"
	// amd64:`MOVL\s[$]0,\s\([A-Z]+\)`
	// 386:`MOVL\s[$]0,\s\([A-Z]+\)`
	// ppc64x:`MOVW\s`
	h1[0], h1[1] = 0, 0
	// arm64:"MOVW\tZR",-"MOVB",-"MOVH"
	// amd64:`MOVL\s[$]0,\s\([A-Z]+\)`
	// 386:`MOVL\s[$]0,\s\([A-Z]+\)`
	// ppc64x:`MOVW`
	h2[1], h2[0] = 0, 0
}

func zero_uint16_4(h1, h2 []uint16) {
	_, _ = h1[3], h2[3]
	// arm64:"MOVD\tZR",-"MOVB",-"MOVH",-"MOVW"
	// amd64:`MOVQ\s[$]0,\s\([A-Z]+\)`
	// ppc64x:`MOVD\s`
	h1[0], h1[1], h1[2], h1[3] = 0, 0, 0, 0
	// arm64:"MOVD\tZR",-"MOVB",-"MOVH",-"MOVW"
	// ppc64x:`MOVD\s`
	h2[2], h2[3], h2[1], h2[0] = 0, 0, 0, 0
}

func zero_uint16_8(h []uint16) {
	_ = h[7]
	h[0], h[1], h[2], h[3] = 0, 0, 0, 0 // arm64:"STP",-"MOVB",-"MOVH"
	h[4], h[5], h[6], h[7] = 0, 0, 0, 0
}

func zero_uint32_2(w1, w2 []uint32) {
	_, _ = w1[1], w2[1]
	// arm64:"MOVD\tZR",-"MOVB",-"MOVH",-"MOVW"
	// amd64:`MOVQ\s[$]0,\s\([A-Z]+\)`
	// ppc64x:`MOVD\s`
	w1[0], w1[1] = 0, 0
	// arm64:"MOVD\tZR",-"MOVB",-"MOVH",-"MOVW"
	// amd64:`MOVQ\s[$]0,\s\([A-Z]+\)`
	// ppc64x:`MOVD\s`
	w2[1], w2[0] = 0, 0
}

func zero_uint32_4(w1, w2 []uint32) {
	_, _ = w1[3], w2[3]
	w1[0], w1[1], w1[2], w1[3] = 0, 0, 0, 0 // arm64:"STP",-"MOVB",-"MOVH"
	w2[2], w2[3], w2[1], w2[0] = 0, 0, 0, 0 // arm64:"STP",-"MOVB",-"MOVH"
}

func zero_uint64_2(d1, d2 []uint64) {
	_, _ = d1[1], d2[1]
	d1[0], d1[1] = 0, 0 // arm64:"STP",-"MOVB",-"MOVH"
	d2[1], d2[0] = 0, 0 // arm64:"STP",-"MOVB",-"MOVH"
}

func loadstore(p, q *[4]uint8) {
	// amd64:"MOVL",-"MOVB"
	// arm64:"MOVWU",-"MOVBU"
	x0, x1, x2, x3 := q[0], q[1], q[2], q[3]
	// amd64:"MOVL",-"MOVB"
	// arm64:"MOVW",-"MOVB"
	p[0], p[1], p[2], p[3] = x0, x1, x2, x3
}

type S1 struct {
	a, b int16
}

func loadstore2(p, q *S1) {
	// amd64:"MOVL",-"MOVWLZX"
	// arm64:"MOVWU",-"MOVH"
	a, b := p.a, p.b
	// amd64:"MOVL",-"MOVW"
	// arm64:"MOVW",-"MOVH"
	q.a, q.b = a, b
}

func wideStore(p *[8]uint64) {
	if p == nil {
		return
	}

	// amd64:"MOVUPS",-"MOVQ"
	// arm64:"STP",-"MOVD"
	p[0] = 0
	// amd64:-"MOVUPS",-"MOVQ"
	// arm64:-"STP",-"MOVD"
	p[1] = 0
}

func wideStore2(p *[8]uint64, x, y uint64) {
	if p == nil {
		return
	}

	// s390x:"STMG"
	p[0] = x
	// s390x:-"STMG",-"MOVD"
	p[1] = y
}

func store32le(p *struct{ a, b uint32 }, x uint64) {
	// amd64:"MOVQ",-"MOVL",-"SHRQ"
	// arm64:"MOVD",-"MOVW",-"LSR"
	// ppc64le:"MOVD",-"MOVW",-"SRD"
	p.a = uint32(x)
	// amd64:-"MOVL",-"SHRQ"
	// arm64:-"MOVW",-"LSR"
	// ppc64le:-"MOVW",-"SRD"
	p.b = uint32(x >> 32)
}
func store32be(p *struct{ a, b uint32 }, x uint64) {
	// ppc64:"MOVD",-"MOVW",-"SRD"
	// s390x:"MOVD",-"MOVW",-"SRD"
	p.a = uint32(x >> 32)
	// ppc64:-"MOVW",-"SRD"
	// s390x:-"MOVW",-"SRD"
	p.b = uint32(x)
}
func store16le(p *struct{ a, b uint16 }, x uint32) {
	// amd64:"MOVL",-"MOVW",-"SHRL"
	// arm64:"MOVW",-"MOVH",-"UBFX"
	// ppc64le:"MOVW",-"MOVH",-"SRW"
	p.a = uint16(x)
	// amd64:-"MOVW",-"SHRL"
	// arm64:-"MOVH",-"UBFX"
	// ppc64le:-"MOVH",-"SRW"
	p.b = uint16(x >> 16)
}
func store16be(p *struct{ a, b uint16 }, x uint32) {
	// ppc64:"MOVW",-"MOVH",-"SRW"
	// s390x:"MOVW",-"MOVH",-"SRW"
	p.a = uint16(x >> 16)
	// ppc64:-"MOVH",-"SRW"
	// s390x:-"MOVH",-"SRW"
	p.b = uint16(x)
}

func storeBoolConst(p *struct{ a, b bool }) {
	// amd64:"MOVW",-"MOVB"
	// arm64:"MOVH",-"MOVB"
	p.a = true
	p.b = true
}
func issue66413(p *struct {
	a byte
	b bool
	c bool
	d int8
}) {
	// amd64:"MOVL",-"MOVB"
	// arm64:"MOVW",-"MOVB"
	p.a = 31
	p.b = false
	p.c = true
	p.d = 12
}
