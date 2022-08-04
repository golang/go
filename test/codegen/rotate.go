// asmcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

import "math/bits"

// ------------------- //
//    const rotates    //
// ------------------- //

func rot64(x uint64) uint64 {
	var a uint64

	// amd64:"ROLQ\t[$]7"
	// ppc64:"ROTL\t[$]7"
	// ppc64le:"ROTL\t[$]7"
	a += x<<7 | x>>57

	// amd64:"ROLQ\t[$]8"
	// arm64:"ROR\t[$]56"
	// s390x:"RISBGZ\t[$]0, [$]63, [$]8, "
	// ppc64:"ROTL\t[$]8"
	// ppc64le:"ROTL\t[$]8"
	a += x<<8 + x>>56

	// amd64:"ROLQ\t[$]9"
	// arm64:"ROR\t[$]55"
	// s390x:"RISBGZ\t[$]0, [$]63, [$]9, "
	// ppc64:"ROTL\t[$]9"
	// ppc64le:"ROTL\t[$]9"
	a += x<<9 ^ x>>55

	// amd64:"ROLQ\t[$]10"
	// arm64:"ROR\t[$]54"
	// s390x:"RISBGZ\t[$]0, [$]63, [$]10, "
	// ppc64:"ROTL\t[$]10"
	// ppc64le:"ROTL\t[$]10"
	// arm64:"ROR\t[$]54"
	// s390x:"RISBGZ\t[$]0, [$]63, [$]10, "
	a += bits.RotateLeft64(x, 10)

	return a
}

func rot32(x uint32) uint32 {
	var a uint32

	// amd64:"ROLL\t[$]7"
	// arm:"MOVW\tR\\d+@>25"
	// ppc64:"ROTLW\t[$]7"
	// ppc64le:"ROTLW\t[$]7"
	a += x<<7 | x>>25

	// amd64:`ROLL\t[$]8`
	// arm:"MOVW\tR\\d+@>24"
	// arm64:"RORW\t[$]24"
	// s390x:"RLL\t[$]8"
	// ppc64:"ROTLW\t[$]8"
	// ppc64le:"ROTLW\t[$]8"
	a += x<<8 + x>>24

	// amd64:"ROLL\t[$]9"
	// arm:"MOVW\tR\\d+@>23"
	// arm64:"RORW\t[$]23"
	// s390x:"RLL\t[$]9"
	// ppc64:"ROTLW\t[$]9"
	// ppc64le:"ROTLW\t[$]9"
	a += x<<9 ^ x>>23

	// amd64:"ROLL\t[$]10"
	// arm:"MOVW\tR\\d+@>22"
	// arm64:"RORW\t[$]22"
	// s390x:"RLL\t[$]10"
	// ppc64:"ROTLW\t[$]10"
	// ppc64le:"ROTLW\t[$]10"
	// arm64:"RORW\t[$]22"
	// s390x:"RLL\t[$]10"
	a += bits.RotateLeft32(x, 10)

	return a
}

func rot16(x uint16) uint16 {
	var a uint16

	// amd64:"ROLW\t[$]7"
	a += x<<7 | x>>9

	// amd64:`ROLW\t[$]8`
	a += x<<8 + x>>8

	// amd64:"ROLW\t[$]9"
	a += x<<9 ^ x>>7

	return a
}

func rot8(x uint8) uint8 {
	var a uint8

	// amd64:"ROLB\t[$]5"
	a += x<<5 | x>>3

	// amd64:`ROLB\t[$]6`
	a += x<<6 + x>>2

	// amd64:"ROLB\t[$]7"
	a += x<<7 ^ x>>1

	return a
}

// ----------------------- //
//    non-const rotates    //
// ----------------------- //

func rot64nc(x uint64, z uint) uint64 {
	var a uint64

	z &= 63

	// amd64:"ROLQ",-"AND"
	// arm64:"ROR","NEG",-"AND"
	// ppc64:"ROTL",-"NEG",-"AND"
	// ppc64le:"ROTL",-"NEG",-"AND"
	a += x<<z | x>>(64-z)

	// amd64:"RORQ",-"AND"
	// arm64:"ROR",-"NEG",-"AND"
	// ppc64:"ROTL","NEG",-"AND"
	// ppc64le:"ROTL","NEG",-"AND"
	a += x>>z | x<<(64-z)

	return a
}

func rot32nc(x uint32, z uint) uint32 {
	var a uint32

	z &= 31

	// amd64:"ROLL",-"AND"
	// arm64:"ROR","NEG",-"AND"
	// ppc64:"ROTLW",-"NEG",-"AND"
	// ppc64le:"ROTLW",-"NEG",-"AND"
	a += x<<z | x>>(32-z)

	// amd64:"RORL",-"AND"
	// arm64:"ROR",-"NEG",-"AND"
	// ppc64:"ROTLW","NEG",-"AND"
	// ppc64le:"ROTLW","NEG",-"AND"
	a += x>>z | x<<(32-z)

	return a
}

func rot16nc(x uint16, z uint) uint16 {
	var a uint16

	z &= 15

	// amd64:"ROLW",-"ANDQ"
	a += x<<z | x>>(16-z)

	// amd64:"RORW",-"ANDQ"
	a += x>>z | x<<(16-z)

	return a
}

func rot8nc(x uint8, z uint) uint8 {
	var a uint8

	z &= 7

	// amd64:"ROLB",-"ANDQ"
	a += x<<z | x>>(8-z)

	// amd64:"RORB",-"ANDQ"
	a += x>>z | x<<(8-z)

	return a
}

// Issue 18254: rotate after inlining
func f32(x uint32) uint32 {
	// amd64:"ROLL\t[$]7"
	return rot32nc(x, 7)
}

// --------------------------------------- //
//    Combined Rotate + Masking operations //
// --------------------------------------- //

func checkMaskedRotate32(a []uint32, r int) {
	i := 0

	// ppc64le: "RLWNM\t[$]16, R[0-9]+, [$]8, [$]15, R[0-9]+"
	// ppc64: "RLWNM\t[$]16, R[0-9]+, [$]8, [$]15, R[0-9]+"
	a[i] = bits.RotateLeft32(a[i], 16) & 0xFF0000
	i++
	// ppc64le: "RLWNM\t[$]16, R[0-9]+, [$]8, [$]15, R[0-9]+"
	// ppc64: "RLWNM\t[$]16, R[0-9]+, [$]8, [$]15, R[0-9]+"
	a[i] = bits.RotateLeft32(a[i]&0xFF, 16)
	i++
	// ppc64le: "RLWNM\t[$]4, R[0-9]+, [$]20, [$]27, R[0-9]+"
	// ppc64: "RLWNM\t[$]4, R[0-9]+, [$]20, [$]27, R[0-9]+"
	a[i] = bits.RotateLeft32(a[i], 4) & 0xFF0
	i++
	// ppc64le: "RLWNM\t[$]16, R[0-9]+, [$]24, [$]31, R[0-9]+"
	// ppc64: "RLWNM\t[$]16, R[0-9]+, [$]24, [$]31, R[0-9]+"
	a[i] = bits.RotateLeft32(a[i]&0xFF0000, 16)
	i++

	// ppc64le: "RLWNM\tR[0-9]+, R[0-9]+, [$]8, [$]15, R[0-9]+"
	// ppc64: "RLWNM\tR[0-9]+, R[0-9]+, [$]8, [$]15, R[0-9]+"
	a[i] = bits.RotateLeft32(a[i], r) & 0xFF0000
	i++
	// ppc64le: "RLWNM\tR[0-9]+, R[0-9]+, [$]16, [$]23, R[0-9]+"
	// ppc64: "RLWNM\tR[0-9]+, R[0-9]+, [$]16, [$]23, R[0-9]+"
	a[i] = bits.RotateLeft32(a[3], r) & 0xFF00
	i++

	// ppc64le: "RLWNM\tR[0-9]+, R[0-9]+, [$]20, [$]11, R[0-9]+"
	// ppc64: "RLWNM\tR[0-9]+, R[0-9]+, [$]20, [$]11, R[0-9]+"
	a[i] = bits.RotateLeft32(a[3], r) & 0xFFF00FFF
	i++
	// ppc64le: "RLWNM\t[$]4, R[0-9]+, [$]20, [$]11, R[0-9]+"
	// ppc64: "RLWNM\t[$]4, R[0-9]+, [$]20, [$]11, R[0-9]+"
	a[i] = bits.RotateLeft32(a[3], 4) & 0xFFF00FFF
	i++
}

// combined arithmetic and rotate on arm64
func checkArithmeticWithRotate(a *[1000]uint64) {
	// arm64: "AND\tR[0-9]+@>51, R[0-9]+, R[0-9]+"
	a[2] = a[1] & bits.RotateLeft64(a[0], 13)
	// arm64: "ORR\tR[0-9]+@>51, R[0-9]+, R[0-9]+"
	a[5] = a[4] | bits.RotateLeft64(a[3], 13)
	// arm64: "EOR\tR[0-9]+@>51, R[0-9]+, R[0-9]+"
	a[8] = a[7] ^ bits.RotateLeft64(a[6], 13)
	// arm64: "MVN\tR[0-9]+@>51, R[0-9]+"
	a[10] = ^bits.RotateLeft64(a[9], 13)
	// arm64: "BIC\tR[0-9]+@>51, R[0-9]+, R[0-9]+"
	a[13] = a[12] &^ bits.RotateLeft64(a[11], 13)
	// arm64: "EON\tR[0-9]+@>51, R[0-9]+, R[0-9]+"
	a[16] = a[15] ^ ^bits.RotateLeft64(a[14], 13)
	// arm64: "ORN\tR[0-9]+@>51, R[0-9]+, R[0-9]+"
	a[19] = a[18] | ^bits.RotateLeft64(a[17], 13)
	// arm64: "TST\tR[0-9]+@>51, R[0-9]+"
	if a[18]&bits.RotateLeft64(a[19], 13) == 0 {
		a[20] = 1
	}

}
