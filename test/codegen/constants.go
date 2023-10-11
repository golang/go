// asmcheck

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

// A uint16 or sint16 constant shifted left.
func shifted16BitConstants(out [64]uint64) {
	// ppc64x: "MOVD\t[$]8193,", "SLD\t[$]27,"
	out[0] = 0x0000010008000000
	// ppc64x: "MOVD\t[$]-32767", "SLD\t[$]26,"
	out[1] = 0xFFFFFE0004000000
	// ppc64x: "MOVD\t[$]-1", "SLD\t[$]48,"
	out[2] = 0xFFFF000000000000
	// ppc64x: "MOVD\t[$]65535", "SLD\t[$]44,"
	out[3] = 0x0FFFF00000000000
}

// A contiguous set of 1 bits, potentially wrapping.
func contiguousMaskConstants(out [64]uint64) {
	// ppc64x: "MOVD\t[$]-1", "RLDC\tR[0-9]+, [$]44, [$]63,"
	out[0] = 0xFFFFF00000000001
	// ppc64x: "MOVD\t[$]-1", "RLDC\tR[0-9]+, [$]43, [$]63,"
	out[1] = 0xFFFFF80000000001
	// ppc64x: "MOVD\t[$]-1", "RLDC\tR[0-9]+, [$]43, [$]4,"
	out[2] = 0x0FFFF80000000000
	// ppc64x/power8: "MOVD\t[$]-1", "RLDC\tR[0-9]+, [$]33, [$]63,"
	// ppc64x/power9: "MOVD\t[$]-1", "RLDC\tR[0-9]+, [$]33, [$]63,"
	// ppc64x/power10: "MOVD\t[$]-8589934591,"
	out[3] = 0xFFFFFFFE00000001
}
