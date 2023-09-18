// asmcheck

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

func shifted16BitConstants(out [64]uint64) {
	// ppc64x: "MOVD\t[$]8193,", "SLD\t[$]27,"
	out[0] = 0x0000010008000000
	// ppc64x: "MOVD\t[$]-32767", "SLD\t[$]26,"
	out[1] = 0xFFFFFE0004000000
	// ppc64x: "MOVD\t[$]-1", "SLD\t[$]48,"
	out[2] = 0xFFFF000000000000
	// ppc64x: "MOVD\t[$]65535", "SLD\t[$]44,"
	out[3] = 0x0FFFF00000000000

	// ppc64x: "MOVD\t[$]i64.fffff00000000001[(]SB[)]"
	out[4] = 0xFFFFF00000000001
	// ppc64x: "MOVD\t[$]i64.fffff80000000001[(]SB[)]"
	out[5] = 0xFFFFF80000000001
	// ppc64x: "MOVD\t[$]i64.0ffff80000000000[(]SB[)]"
	out[6] = 0x0FFFF80000000000
}
