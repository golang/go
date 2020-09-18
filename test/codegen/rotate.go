// asmcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

// ------------------- //
//    const rotates    //
// ------------------- //

func rot64(x uint64) uint64 {
	var a uint64

	// amd64:"ROLQ\t[$]7"
	// arm64:"ROR\t[$]57"
	// s390x:"RLLG\t[$]7"
	// ppc64:"ROTL\t[$]7"
	// ppc64le:"ROTL\t[$]7"
	a += x<<7 | x>>57

	// amd64:"ROLQ\t[$]8"
	// arm64:"ROR\t[$]56"
	// s390x:"RLLG\t[$]8"
	// ppc64:"ROTL\t[$]8"
	// ppc64le:"ROTL\t[$]8"
	a += x<<8 + x>>56

	// amd64:"ROLQ\t[$]9"
	// arm64:"ROR\t[$]55"
	// s390x:"RLLG\t[$]9"
	// ppc64:"ROTL\t[$]9"
	// ppc64le:"ROTL\t[$]9"
	a += x<<9 ^ x>>55

	return a
}

func rot32(x uint32) uint32 {
	var a uint32

	// amd64:"ROLL\t[$]7"
	// arm:"MOVW\tR\\d+@>25"
	// arm64:"RORW\t[$]25"
	// s390x:"RLL\t[$]7"
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

	// amd64:"ROLQ"
	// ppc64:"ROTL"
	// ppc64le:"ROTL"
	a += x<<z | x>>(64-z)

	// amd64:"RORQ"
	a += x>>z | x<<(64-z)

	return a
}

func rot32nc(x uint32, z uint) uint32 {
	var a uint32

	z &= 31

	// amd64:"ROLL"
	// ppc64:"ROTLW"
	// ppc64le:"ROTLW"
	a += x<<z | x>>(32-z)

	// amd64:"RORL"
	a += x>>z | x<<(32-z)

	return a
}

func rot16nc(x uint16, z uint) uint16 {
	var a uint16

	z &= 15

	// amd64:"ROLW"
	a += x<<z | x>>(16-z)

	// amd64:"RORW"
	a += x>>z | x<<(16-z)

	return a
}

func rot8nc(x uint8, z uint) uint8 {
	var a uint8

	z &= 7

	// amd64:"ROLB"
	a += x<<z | x>>(8-z)

	// amd64:"RORB"
	a += x>>z | x<<(8-z)

	return a
}

// Issue 18254: rotate after inlining
func f32(x uint32) uint32 {
	// amd64:"ROLL\t[$]7"
	return rot32nc(x, 7)
}
