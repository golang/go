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
