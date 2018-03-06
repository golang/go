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
	//amd64:"BSRQ"
	//s390x:"FLOGR"
	//arm:"CLZ" arm64:"CLZ"
	//mips:"CLZ"
	return bits.LeadingZeros(n)
}

func LeadingZeros64(n uint64) int {
	//amd64:"BSRQ"
	//s390x:"FLOGR"
	//arm:"CLZ" arm64:"CLZ"
	//mips:"CLZ"
	return bits.LeadingZeros64(n)
}

func LeadingZeros32(n uint32) int {
	//amd64:"BSRQ"
	//s390x:"FLOGR"
	//arm:"CLZ" arm64:"CLZ"
	//mips:"CLZ"
	return bits.LeadingZeros32(n)
}

func LeadingZeros16(n uint16) int {
	//amd64:"BSRQ"
	//s390x:"FLOGR"
	//arm:"CLZ" arm64:"CLZ"
	//mips:"CLZ"
	return bits.LeadingZeros16(n)
}

func LeadingZeros8(n uint8) int {
	//amd64 LeadingZeros8 not intrinsified (see ssa.go)
	//s390x:"FLOGR"
	//arm:"CLZ" arm64:"CLZ"
	//mips:"CLZ"
	return bits.LeadingZeros8(n)
}

// --------------- //
//    bits.Len*    //
// --------------- //

func Len(n uint) int {
	//amd64:"BSRQ"
	//s390x:"FLOGR"
	//arm:"CLZ" arm64:"CLZ"
	//mips:"CLZ"
	return bits.Len(n)
}

func Len64(n uint64) int {
	//amd64:"BSRQ"
	//s390x:"FLOGR"
	//arm:"CLZ" arm64:"CLZ"
	//mips:"CLZ"
	return bits.Len64(n)
}

func Len32(n uint32) int {
	//amd64:"BSRQ"
	//s390x:"FLOGR"
	//arm:"CLZ" arm64:"CLZ"
	//mips:"CLZ"
	return bits.Len32(n)
}

func Len16(n uint16) int {
	//amd64:"BSRQ"
	//s390x:"FLOGR"
	//arm:"CLZ" arm64:"CLZ"
	//mips:"CLZ"
	return bits.Len16(n)
}

func Len8(n uint8) int {
	//amd64 Len8 not intrisified (see ssa.go)
	//s390x:"FLOGR"
	//arm:"CLZ" arm64:"CLZ"
	//mips:"CLZ"
	return bits.Len8(n)
}

// -------------------- //
//    bits.OnesCount    //
// -------------------- //

func OnesCount(n uint) int {
	//amd64:"POPCNTQ",".*support_popcnt"
	//arm64:"VCNT","VUADDLV"
	return bits.OnesCount(n)
}

func OnesCount64(n uint64) int {
	//amd64:"POPCNTQ",".*support_popcnt"
	//arm64:"VCNT","VUADDLV"
	return bits.OnesCount64(n)
}

func OnesCount32(n uint32) int {
	//amd64:"POPCNTL",".*support_popcnt"
	//arm64:"VCNT","VUADDLV"
	return bits.OnesCount32(n)
}

func OnesCount16(n uint16) int {
	//amd64:"POPCNTL",".*support_popcnt"
	//arm64:"VCNT","VUADDLV"
	return bits.OnesCount16(n)
}

// ------------------------ //
//    bits.TrailingZeros    //
// ------------------------ //

func TrailingZeros(n uint) int {
	//amd64:"BSFQ","MOVL\t\\$64","CMOVQEQ"
	//s390x:"FLOGR"
	return bits.TrailingZeros(n)
}

func TrailingZeros64(n uint64) int {
	//amd64:"BSFQ","MOVL\t\\$64","CMOVQEQ"
	//s390x:"FLOGR"
	return bits.TrailingZeros64(n)
}

func TrailingZeros32(n uint32) int {
	//amd64:"MOVQ\t\\$4294967296","ORQ\t[^$]","BSFQ"
	//s390x:"FLOGR","MOVWZ"
	return bits.TrailingZeros32(n)
}

func TrailingZeros16(n uint16) int {
	//amd64:"BSFQ","ORQ\t\\$65536"
	//s390x:"FLOGR","OR\t\\$65536"
	return bits.TrailingZeros16(n)
}

func TrailingZeros8(n uint8) int {
	//amd64:"BSFQ","ORQ\t\\$256"
	//s390x:"FLOGR","OR\t\\$256"
	return bits.TrailingZeros8(n)
}
