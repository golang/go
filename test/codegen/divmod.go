// asmcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

// Div and mod rewrites, testing cmd/compile/internal/ssa/_gen/divmod.rules.
// See comments there for "Case 1" etc.

// Convert multiplication by a power of two to a shift.

func mul32_uint8(i uint8) uint8 {
	// 386: "SHLL [$]5,"
	// arm64: "LSL [$]5,"
	return i * 32
}

func mul32_uint16(i uint16) uint16 {
	// 386: "SHLL [$]5,"
	// arm64: "LSL [$]5,"
	return i * 32
}

func mul32_uint32(i uint32) uint32 {
	// 386: "SHLL [$]5,"
	// arm64: "LSL [$]5,"
	return i * 32
}

func mul32_uint64(i uint64) uint64 {
	// 386: "SHLL [$]5,"
	// 386: "SHRL [$]27,"
	// arm64: "LSL [$]5,"
	return i * 32
}

func mulNeg32_int8(i int8) int8 {
	// 386: "SHLL [$]5,"
	// 386: "NEGL"
	// arm64: "NEG R[0-9]+<<5,"
	return i * -32
}

func mulNeg32_int16(i int16) int16 {
	// 386: "SHLL [$]5,"
	// 386: "NEGL"
	// arm64: "NEG R[0-9]+<<5,"
	return i * -32
}

func mulNeg32_int32(i int32) int32 {
	// 386: "SHLL [$]5,"
	// 386: "NEGL"
	// arm64: "NEG R[0-9]+<<5,"
	return i * -32
}

func mulNeg32_int64(i int64) int64 {
	// 386: "SHLL [$]5,"
	// 386: "SHRL [$]27,"
	// 386: "SBBL"
	// arm64: "NEG R[0-9]+<<5,"
	return i * -32
}

// Signed divide by power of 2.

func div32_int8(i int8) int8 {
	// 386: "SARB [$]7,"
	// 386: "SHRB [$]3,"
	// 386: "ADDL"
	// 386: "SARB [$]5,"
	// arm64: "SBFX [$]7, R[0-9]+, [$]1,"
	// arm64: "ADD R[0-9]+>>3,"
	// arm64: "SBFX [$]5, R[0-9]+, [$]3,"
	return i / 32
}

func div32_int16(i int16) int16 {
	// 386: "SARW [$]15,"
	// 386: "SHRW [$]11,"
	// 386: "ADDL"
	// 386: "SARW [$]5,"
	// arm64: "SBFX [$]15, R[0-9]+, [$]1,"
	// arm64: "ADD R[0-9]+>>11,"
	// arm64: "SBFX [$]5, R[0-9]+, [$]11,"
	return i / 32
}

func div32_int32(i int32) int32 {
	// 386: "SARL [$]31,"
	// 386: "SHRL [$]27,"
	// 386: "ADDL"
	// 386: "SARL [$]5,"
	// arm64: "SBFX [$]31, R[0-9]+, [$]1,"
	// arm64: "ADD R[0-9]+>>27,"
	// arm64: "SBFX [$]5, R[0-9]+, [$]27,"
	return i / 32
}

func div32_int64(i int64) int64 {
	// 386: "SARL [$]31,"
	// 386: "SHRL [$]27,"
	// 386: "ADDL"
	// 386: "SARL [$]5,"
	// 386: "SHRL [$]5,"
	// 386: "SHLL [$]27,"
	// arm64: "ASR [$]63,"
	// arm64: "ADD R[0-9]+>>59,"
	// arm64: "ASR [$]5,"
	return i / 32
}

// Case 1. Signed divides where 2N ≤ register size.

func div7_int8(i int8) int8 {
	// 386: "SARL [$]31,"
	// 386: "IMUL3L [$]147,"
	// 386: "SARL [$]10,"
	// 386: "SUBL"
	// arm64: "MOVD [$]147,"
	// arm64: "MULW"
	// arm64: "SBFX [$]10, R[0-9]+, [$]22,"
	// arm64: "SUB R[0-9]+->31,"
	// wasm: "I64Const [$]147"
	return i / 7
}

func div7_int16(i int16) int16 {
	// 386: "SARL [$]31,"
	// 386: "IMUL3L [$]37450,"
	// 386: "SARL [$]18,"
	// 386: "SUBL"
	// arm64: "MOVD [$]37450,"
	// arm64: "MULW"
	// arm64: "SBFX [$]18, R[0-9]+, [$]14,"
	// arm64: "SUB R[0-9]+->31,"
	// wasm: "I64Const [$]37450"
	return i / 7
}

func div7_int32(i int32) int32 {
	// 64-bit only
	// arm64: "MOVD [$]2454267027,"
	// arm64: "MUL "
	// arm64: "ASR [$]34,"
	// arm64: "SUB R[0-9]+->63,"
	// wasm: "I64Const [$]2454267027"
	return i / 7
}

// Case 2. Signed divides where m is even.

func div9_int32(i int32) int32 {
	// 386: "SARL [$]31,"
	// 386: "MOVL [$]1908874354,"
	// 386: "IMULL"
	// 386: "SARL [$]2,"
	// 386: "SUBL"
	// arm64: "MOVD [$]3817748708,"
	// arm64: "MUL "
	// arm64: "ASR [$]35,"
	// arm64: "SUB R[0-9]+->63,"
	// wasm: "I64Const [$]3817748708"
	return i / 9
}

func div7_int64(i int64) int64 {
	// 64-bit only
	// arm64 MOVD $5270498306774157605, SMULH, ASR $1, SUB ->63
	// arm64: "MOVD [$]5270498306774157605,"
	// arm64: "SMULH"
	// arm64: "ASR [$]1,"
	// arm64: "SUB R[0-9]+->63,"
	// wasm: "I64Const [$]613566757"
	// wasm: "I64Const [$]1227133513"
	return i / 7
}

// Case 3. Signed divides where m is odd.

func div3_int32(i int32) int32 {
	// 386: "SARL [$]31,"
	// 386: "MOVL [$]-1431655765,"
	// 386: "IMULL"
	// 386: "SARL [$]1,"
	// 386: "SUBL"
	// arm64: "MOVD [$]2863311531,"
	// arm64: "MUL"
	// arm64: "ASR [$]33,"
	// arm64: "SUB R[0-9]+->63,"
	// wasm: "I64Const [$]2863311531"
	return i / 3
}

func div3_int64(i int64) int64 {
	// 64-bit only
	// arm64: "MOVD [$]-6148914691236517205,"
	// arm64: "SMULH"
	// arm64: "ADD"
	// arm64: "ASR [$]1,"
	// arm64: "SUB R[0-9]+->63,"
	// wasm: "I64Const [$]-1431655766"
	// wasm: "I64Const [$]2863311531"
	return i / 3
}

// Case 4. Unsigned divide where x < 1<<(N-1).

func div7_int16u(i int16) int16 {
	if i < 0 {
		return 0
	}
	// 386: "IMUL3L [$]37450,"
	// 386: "SHRL [$]18,"
	// 386: -"SUBL"
	// arm64: "MOVD [$]37450,"
	// arm64: "MULW"
	// arm64: "UBFX [$]18, R[0-9]+, [$]14,"
	// arm64: -"SUB"
	// wasm: "I64Const [$]37450"
	// wasm -"I64Sub"
	return i / 7
}

func div7_int32u(i int32) int32 {
	if i < 0 {
		return 0
	}
	// 386: "MOVL [$]-1840700269,"
	// 386: "MULL"
	// 386: "SHRL [$]2"
	// 386: -"SUBL"
	// arm64: "MOVD [$]2454267027,"
	// arm64: "MUL"
	// arm64: "LSR [$]34,"
	// arm64: -"SUB"
	// wasm: "I64Const [$]2454267027"
	// wasm -"I64Sub"
	return i / 7
}

func div7_int64u(i int64) int64 {
	// 64-bit only
	if i < 0 {
		return 0
	}
	// arm64: "MOVD [$]-7905747460161236406,"
	// arm64: "UMULH"
	// arm64: "LSR [$]2,"
	// arm64: -"SUB"
	// wasm: "I64Const [$]1227133514"
	// wasm: "I64Const [$]2454267026"
	// wasm -"I64Sub"
	return i / 7
}

// Case 5. Unsigned divide where 2N+1 ≤ register size.

func div7_uint8(i uint8) uint8 {
	// 386: "IMUL3L [$]293,"
	// 386: "SHRL [$]11,"
	// arm64: "MOVD [$]293,"
	// arm64: "MULW"
	// arm64: "UBFX [$]11, R[0-9]+, [$]21,"
	// wasm: "I64Const [$]293"
	return i / 7
}

func div7_uint16(i uint16) uint16 {
	// only 64-bit
	// arm64: "MOVD [$]74899,"
	// arm64: "MUL"
	// arm64: "LSR [$]19,"
	// wasm: "I64Const [$]74899"
	return i / 7
}

// Case 6. Unsigned divide where m is even.

func div3_uint16(i uint16) uint16 {
	// 386: "IMUL3L [$]43691," "SHRL [$]17,"
	// arm64: "MOVD [$]87382,"
	// arm64: "MUL"
	// arm64: "LSR [$]18,"
	// wasm: "I64Const [$]87382"
	return i / 3
}

func div3_uint32(i uint32) uint32 {
	// 386: "MOVL [$]-1431655765," "MULL", "SHRL [$]1,"
	// arm64: "MOVD [$]2863311531,"
	// arm64: "MUL"
	// arm64: "LSR [$]33,"
	// wasm: "I64Const [$]2863311531"
	return i / 3
}

func div3_uint64(i uint64) uint64 {
	// 386: "MOVL [$]-1431655766"
	// 386: "MULL"
	// 386: "SHRL [$]1"
	// 386 -".*CALL"
	// arm64: "MOVD [$]-6148914691236517205,"
	// arm64: "UMULH"
	// arm64: "LSR [$]1,"
	// wasm: "I64Const [$]2863311530"
	// wasm: "I64Const [$]2863311531"
	return i / 3
}

// Case 7. Unsigned divide where c is even.

func div14_uint16(i uint16) uint16 {
	// 32-bit only
	// 386: "SHRL [$]1,"
	// 386: "IMUL3L [$]37450,"
	// 386: "SHRL [$]18,"
	return i / 14
}

func div14_uint32(i uint32) uint32 {
	// 386: "SHRL [$]1,"
	// 386: "MOVL [$]-1840700269,"
	// 386: "SHRL [$]2,"
	// arm64: "UBFX [$]1, R[0-9]+, [$]31,"
	// arm64: "MOVD [$]2454267027,"
	// arm64: "MUL"
	// arm64: "LSR [$]34,"
	// wasm: "I64Const [$]2454267027"
	return i / 14
}

func div14_uint64(i uint64) uint64 {
	// 386: "MOVL [$]-1840700270,"
	// 386: "MULL"
	// 386: "SHRL [$]2,"
	// 386: -".*CALL"
	// arm64: "MOVD [$]-7905747460161236406,"
	// arm64: "UMULH"
	// arm64: "LSR [$]2,"
	// wasm: "I64Const [$]1227133514"
	// wasm: "I64Const [$]2454267026"
	return i / 14
}

// Case 8. Unsigned divide on systems with avg.

func div7_uint16a(i uint16) uint16 {
	// only 32-bit
	// 386: "SHLL [$]16,"
	// 386: "IMUL3L [$]9363,"
	// 386: "ADDL"
	// 386: "RCRL [$]1,"
	// 386: "SHRL [$]18,"
	return i / 7
}

func div7_uint32(i uint32) uint32 {
	// 386: "MOVL [$]613566757,"
	// 386: "MULL"
	// 386: "ADDL"
	// 386: "RCRL [$]1,"
	// 386: "SHRL [$]2,"
	// arm64: "UBFIZ [$]32, R[0-9]+, [$]32,"
	// arm64: "MOVD [$]613566757,"
	// arm64: "MUL"
	// arm64: "SUB"
	// arm64: "ADD R[0-9]+>>1,"
	// arm64: "LSR [$]34,"
	// wasm: "I64Const [$]613566757"
	return i / 7
}

func div7_uint64(i uint64) uint64 {
	// 386: "MOVL [$]-1840700269,"
	// 386: "MULL"
	// 386: "SHRL [$]2,"
	// 386: -".*CALL"
	// arm64: "MOVD [$]2635249153387078803,"
	// arm64: "UMULH"
	// arm64: "SUB",
	// arm64: "ADD R[0-9]+>>1,"
	// arm64: "LSR [$]2,"
	// wasm: "I64Const [$]613566756"
	// wasm: "I64Const [$]2454267027"
	return i / 7
}

func div12345_uint64(i uint64) uint64 {
	// 386: "MOVL [$]-1444876402,"
	// 386: "MOVL [$]835683390,"
	// 386: "MULL"
	// 386: "SHRL [$]13,"
	// 386: "SHLL [$]19,"
	// arm64: "MOVD [$]-6205696892516465602,"
	// arm64: "UMULH"
	// arm64: "LSR [$]13,"
	// wasm: "I64Const [$]835683390"
	// wasm: "I64Const [$]2850090894"
	return i / 12345
}

// Divisibility and non-divisibility by power of two.

func divis32_uint8(i uint8) bool {
	// 386: "TESTB [$]31,"
	// arm64: "TSTW [$]31,"
	return i%32 == 0
}

func ndivis32_uint8(i uint8) bool {
	// 386: "TESTB [$]31,"
	// arm64: "TSTW [$]31,"
	return i%32 != 0
}

func divis32_uint16(i uint16) bool {
	// 386: "TESTW [$]31,"
	// arm64: "TSTW [$]31,"
	return i%32 == 0
}

func ndivis32_uint16(i uint16) bool {
	// 386: "TESTW [$]31,"
	// arm64: "TSTW [$]31,"
	return i%32 != 0
}

func divis32_uint32(i uint32) bool {
	// 386: "TESTL [$]31,"
	// arm64: "TSTW [$]31,"
	return i%32 == 0
}

func ndivis32_uint32(i uint32) bool {
	// 386: "TESTL [$]31,"
	// arm64: "TSTW [$]31,"
	return i%32 != 0
}

func divis32_uint64(i uint64) bool {
	// 386: "TESTL [$]31,"
	// arm64: "TST [$]31,"
	return i%32 == 0
}

func ndivis32_uint64(i uint64) bool {
	// 386: "TESTL [$]31,"
	// arm64: "TST [$]31,"
	return i%32 != 0
}

func divis32_int8(i int8) bool {
	// 386: "TESTB [$]31,"
	// arm64: "TSTW [$]31,"
	return i%32 == 0
}

func ndivis32_int8(i int8) bool {
	// 386: "TESTB [$]31,"
	// arm64: "TSTW [$]31,"
	return i%32 != 0
}

func divis32_int16(i int16) bool {
	// 386: "TESTW [$]31,"
	// arm64: "TSTW [$]31,"
	return i%32 == 0
}

func ndivis32_int16(i int16) bool {
	// 386: "TESTW [$]31,"
	// arm64: "TSTW [$]31,"
	return i%32 != 0
}

func divis32_int32(i int32) bool {
	// 386: "TESTL [$]31,"
	// arm64: "TSTW [$]31,"
	return i%32 == 0
}

func ndivis32_int32(i int32) bool {
	// 386: "TESTL [$]31,"
	// arm64: "TSTW [$]31,"
	return i%32 != 0
}

func divis32_int64(i int64) bool {
	// 386: "TESTL [$]31,"
	// arm64: "TST [$]31,"
	return i%32 == 0
}

func ndivis32_int64(i int64) bool {
	// 386: "TESTL [$]31,"
	// arm64: "TST [$]31,"
	return i%32 != 0
}

// Divide with divisibility check; reuse divide intermediate mod.

func div_divis32_uint8(i uint8) (uint8, bool) {
	// 386: "SHRB [$]5,"
	// 386: "TESTB [$]31,",
	// 386: "SETEQ"
	// arm64: "UBFX [$]5, R[0-9]+, [$]3"
	// arm64: "TSTW [$]31,"
	// arm64: "CSET EQ"
	return i / 32, i%32 == 0
}

func div_ndivis32_uint8(i uint8) (uint8, bool) {
	// 386: "SHRB [$]5,"
	// 386: "TESTB [$]31,",
	// 386: "SETNE"
	// arm64: "UBFX [$]5, R[0-9]+, [$]3"
	// arm64: "TSTW [$]31,"
	// arm64: "CSET NE"
	return i / 32, i%32 != 0
}

func div_divis32_uint16(i uint16) (uint16, bool) {
	// 386: "SHRW [$]5,"
	// 386: "TESTW [$]31,",
	// 386: "SETEQ"
	// arm64: "UBFX [$]5, R[0-9]+, [$]11"
	// arm64: "TSTW [$]31,"
	// arm64: "CSET EQ"
	return i / 32, i%32 == 0
}

func div_ndivis32_uint16(i uint16) (uint16, bool) {
	// 386: "SHRW [$]5,"
	// 386: "TESTW [$]31,",
	// 386: "SETNE"
	// arm64: "UBFX [$]5, R[0-9]+, [$]11,"
	// arm64: "TSTW [$]31,"
	// arm64: "CSET NE"
	return i / 32, i%32 != 0
}

func div_divis32_uint32(i uint32) (uint32, bool) {
	// 386: "SHRL [$]5,"
	// 386: "TESTL [$]31,",
	// 386: "SETEQ"
	// arm64: "UBFX [$]5, R[0-9]+, [$]27,"
	// arm64: "TSTW [$]31,"
	// arm64: "CSET EQ"
	return i / 32, i%32 == 0
}

func div_ndivis32_uint32(i uint32) (uint32, bool) {
	// 386: "SHRL [$]5,"
	// 386: "TESTL [$]31,",
	// 386: "SETNE"
	// arm64: "UBFX [$]5, R[0-9]+, [$]27,"
	// arm64: "TSTW [$]31,"
	// arm64: "CSET NE"
	return i / 32, i%32 != 0
}

func div_divis32_uint64(i uint64) (uint64, bool) {
	// 386: "SHRL [$]5,"
	// 386: "SHLL [$]27,"
	// 386: "TESTL [$]31,",
	// 386: "SETEQ"
	// arm64: "LSR [$]5,"
	// arm64: "TST [$]31,"
	// arm64: "CSET EQ"
	return i / 32, i%32 == 0
}

func div_ndivis32_uint64(i uint64) (uint64, bool) {
	// 386: "SHRL [$]5,"
	// 386: "SHLL [$]27,"
	// 386: "TESTL [$]31,",
	// 386: "SETNE"
	// arm64: "LSR [$]5,"
	// arm64: "TST [$]31,"
	// arm64: "CSET NE"
	return i / 32, i%32 != 0
}

func div_divis32_int8(i int8) (int8, bool) {
	// 386: "SARB [$]7,"
	// 386: "SHRB [$]3,"
	// 386: "SARB [$]5,"
	// 386: "TESTB [$]31,",
	// 386: "SETEQ"
	// arm64: "SBFX [$]7, R[0-9]+, [$]1,"
	// arm64: "ADD R[0-9]+>>3,"
	// arm64: "SBFX [$]5, R[0-9]+, [$]3,"
	// arm64: "TSTW [$]31,"
	// arm64: "CSET EQ"
	return i / 32, i%32 == 0
}

func div_ndivis32_int8(i int8) (int8, bool) {
	// 386: "SARB [$]7,"
	// 386: "SHRB [$]3,"
	// 386: "SARB [$]5,"
	// 386: "TESTB [$]31,",
	// 386: "SETNE"
	// arm64: "SBFX [$]7, R[0-9]+, [$]1,"
	// arm64: "ADD R[0-9]+>>3,"
	// arm64: "SBFX [$]5, R[0-9]+, [$]3,"
	// arm64: "TSTW [$]31,"
	// arm64: "CSET NE"
	return i / 32, i%32 != 0
}

func div_divis32_int16(i int16) (int16, bool) {
	// 386: "SARW [$]15,"
	// 386: "SHRW [$]11,"
	// 386: "SARW [$]5,"
	// 386: "TESTW [$]31,",
	// 386: "SETEQ"
	// arm64: "SBFX [$]15, R[0-9]+, [$]1,"
	// arm64: "ADD R[0-9]+>>11,"
	// arm64: "SBFX [$]5, R[0-9]+, [$]11,"
	// arm64: "TSTW [$]31,"
	// arm64: "CSET EQ"
	return i / 32, i%32 == 0
}

func div_ndivis32_int16(i int16) (int16, bool) {
	// 386: "SARW [$]15,"
	// 386: "SHRW [$]11,"
	// 386: "SARW [$]5,"
	// 386: "TESTW [$]31,",
	// 386: "SETNE"
	// arm64: "SBFX [$]15, R[0-9]+, [$]1,"
	// arm64: "ADD R[0-9]+>>11,"
	// arm64: "SBFX [$]5, R[0-9]+, [$]11,"
	// arm64: "TSTW [$]31,"
	// arm64: "CSET NE"
	return i / 32, i%32 != 0
}

func div_divis32_int32(i int32) (int32, bool) {
	// 386: "SARL [$]31,"
	// 386: "SHRL [$]27,"
	// 386: "SARL [$]5,"
	// 386: "TESTL [$]31,",
	// 386: "SETEQ"
	// arm64: "SBFX [$]31, R[0-9]+, [$]1,"
	// arm64: "ADD R[0-9]+>>27,"
	// arm64: "SBFX [$]5, R[0-9]+, [$]27,"
	// arm64: "TSTW [$]31,"
	// arm64: "CSET EQ"
	return i / 32, i%32 == 0
}

func div_ndivis32_int32(i int32) (int32, bool) {
	// 386: "SARL [$]31,"
	// 386: "SHRL [$]27,"
	// 386: "SARL [$]5,"
	// 386: "TESTL [$]31,",
	// 386: "SETNE"
	// arm64: "SBFX [$]31, R[0-9]+, [$]1,"
	// arm64: "ADD R[0-9]+>>27,"
	// arm64: "SBFX [$]5, R[0-9]+, [$]27,"
	// arm64: "TSTW [$]31,"
	// arm64: "CSET NE"
	return i / 32, i%32 != 0
}

func div_divis32_int64(i int64) (int64, bool) {
	// 386: "SARL [$]31,"
	// 386: "SHRL [$]27,"
	// 386: "SARL [$]5,"
	// 386: "SHLL [$]27,"
	// 386: "TESTL [$]31,",
	// 386: "SETEQ"
	// arm64: "ASR [$]63,"
	// arm64: "ADD R[0-9]+>>59,"
	// arm64: "ASR [$]5,"
	// arm64: "TST [$]31,"
	// arm64: "CSET EQ"
	return i / 32, i%32 == 0
}

func div_ndivis32_int64(i int64) (int64, bool) {
	// 386: "SARL [$]31,"
	// 386: "SHRL [$]27,"
	// 386: "SARL [$]5,"
	// 386: "SHLL [$]27,"
	// 386: "TESTL [$]31,",
	// 386: "SETNE"
	// arm64: "ASR [$]63,"
	// arm64: "ADD R[0-9]+>>59,"
	// arm64: "ASR [$]5,"
	// arm64: "TST [$]31,"
	// arm64: "CSET NE"
	return i / 32, i%32 != 0
}

// Divisibility and non-divisibility by non-power-of-two.

func divis6_uint8(i uint8) bool {
	// 386: "IMUL3L [$]-85,"
	// 386: "ROLB [$]7,"
	// 386: "CMPB .*, [$]42"
	// 386: "SETLS"
	// arm64: "MOVD [$]-85,"
	// arm64: "MULW"
	// arm64: "UBFX [$]1, R[0-9]+, [$]7,"
	// arm64: "ORR R[0-9]+<<7"
	// arm64: "CMPW [$]42,"
	// arm64: "CSET LS"
	return i%6 == 0
}

func ndivis6_uint8(i uint8) bool {
	// 386: "IMUL3L [$]-85,"
	// 386: "ROLB [$]7,"
	// 386: "CMPB .*, [$]42"
	// 386: "SETHI"
	// arm64: "MOVD [$]-85,"
	// arm64: "MULW"
	// arm64: "UBFX [$]1, R[0-9]+, [$]7,"
	// arm64: "ORR R[0-9]+<<7"
	// arm64: "CMPW [$]42,"
	// arm64: "CSET HI"
	return i%6 != 0
}

func divis6_uint16(i uint16) bool {
	// 386: "IMUL3L [$]-21845,"
	// 386: "ROLW [$]15,"
	// 386: "CMPW .*, [$]10922"
	// 386: "SETLS"
	// arm64: "MOVD [$]-21845,"
	// arm64: "MULW"
	// arm64: "ORR R[0-9]+<<16"
	// arm64: "RORW [$]17,"
	// arm64: "MOVD [$]10922,"
	// arm64: "CSET LS"
	return i%6 == 0
}

func ndivis6_uint16(i uint16) bool {
	// 386: "IMUL3L [$]-21845,"
	// 386: "ROLW [$]15,"
	// 386: "CMPW .*, [$]10922"
	// 386: "SETHI"
	// arm64: "MOVD [$]-21845,"
	// arm64: "MULW"
	// arm64: "ORR R[0-9]+<<16"
	// arm64: "RORW [$]17,"
	// arm64: "MOVD [$]10922,"
	// arm64: "CSET HI"
	return i%6 != 0
}

func divis6_uint32(i uint32) bool {
	// 386: "IMUL3L [$]-1431655765,"
	// 386: "ROLL [$]31,"
	// 386: "CMPL .*, [$]715827882"
	// 386: "SETLS"
	// arm64: "MOVD [$]-1431655765,"
	// arm64: "MULW"
	// arm64: "RORW [$]1,"
	// arm64: "MOVD [$]715827882,"
	// arm64: "CSET LS"
	return i%6 == 0
}

func ndivis6_uint32(i uint32) bool {
	// 386: "IMUL3L [$]-1431655765,"
	// 386: "ROLL [$]31,"
	// 386: "CMPL .*, [$]715827882"
	// 386: "SETHI"
	// arm64: "MOVD [$]-1431655765,"
	// arm64: "MULW"
	// arm64: "RORW [$]1,"
	// arm64: "MOVD [$]715827882,"
	// arm64: "CSET HI"
	return i%6 != 0
}

func divis6_uint64(i uint64) bool {
	// 386: "IMUL3L [$]-1431655766,"
	// 386: "IMUL3L [$]-1431655765,"
	// 386: "MULL"
	// 386: "SHRL [$]1,"
	// 386: "SHLL [$]31,"
	// 386: "CMPL .*, [$]715827882"
	// 386: "SETLS"
	// arm64: "MOVD [$]-6148914691236517205,"
	// arm64: "MUL "
	// arm64: "ROR [$]1,"
	// arm64: "MOVD [$]3074457345618258602,"
	// arm64: "CSET LS"
	return i%6 == 0
}

func ndivis6_uint64(i uint64) bool {
	// 386: "IMUL3L [$]-1431655766,"
	// 386: "IMUL3L [$]-1431655765,"
	// 386: "MULL"
	// 386: "SHRL [$]1,"
	// 386: "SHLL [$]31,"
	// 386: "CMPL .*, [$]715827882"
	// 386: "SETHI"
	// arm64: "MOVD [$]-6148914691236517205,"
	// arm64: "MUL "
	// arm64: "ROR [$]1,"
	// arm64: "MOVD [$]3074457345618258602,"
	// arm64: "CSET HI"
	return i%6 != 0
}

func divis6_int8(i int8) bool {
	// 386: "IMUL3L [$]-85,"
	// 386: "ADDL [$]42,"
	// 386: "ROLB [$]7,"
	// 386: "CMPB .*, [$]42"
	// 386: "SETLS"
	// arm64: "MOVD [$]-85,"
	// arm64: "MULW"
	// arm64: "ADD [$]42,"
	// arm64: "UBFX [$]1, R[0-9]+, [$]7,"
	// arm64: "ORR R[0-9]+<<7"
	// arm64: "CMPW [$]42,"
	// arm64: "CSET LS"
	return i%6 == 0
}

func ndivis6_int8(i int8) bool {
	// 386: "IMUL3L [$]-85,"
	// 386: "ADDL [$]42,"
	// 386: "ROLB [$]7,"
	// 386: "CMPB .*, [$]42"
	// 386: "SETHI"
	// arm64: "MOVD [$]-85,"
	// arm64: "MULW"
	// arm64: "ADD [$]42,"
	// arm64: "UBFX [$]1, R[0-9]+, [$]7,"
	// arm64: "ORR R[0-9]+<<7"
	// arm64: "CMPW [$]42,"
	// arm64: "CSET HI"
	return i%6 != 0
}

func divis6_int16(i int16) bool {
	// 386: "IMUL3L [$]-21845,"
	// 386: "ADDL [$]10922,"
	// 386: "ROLW [$]15,"
	// 386: "CMPW .*, [$]10922"
	// 386: "SETLS"
	// arm64: "MOVD [$]-21845,"
	// arm64: "MULW"
	// arm64: "MOVD [$]10922,"
	// arm64: "ADD "
	// arm64: "ORR R[0-9]+<<16"
	// arm64: "RORW [$]17,"
	// arm64: "MOVD [$]10922,"
	// arm64: "CSET LS"
	return i%6 == 0
}

func ndivis6_int16(i int16) bool {
	// 386: "IMUL3L [$]-21845,"
	// 386: "ADDL [$]10922,"
	// 386: "ROLW [$]15,"
	// 386: "CMPW .*, [$]10922"
	// 386: "SETHI"
	// arm64: "MOVD [$]-21845,"
	// arm64: "MULW"
	// arm64: "MOVD [$]10922,"
	// arm64: "ADD "
	// arm64: "ORR R[0-9]+<<16"
	// arm64: "RORW [$]17,"
	// arm64: "MOVD [$]10922,"
	// arm64: "CSET HI"
	return i%6 != 0
}

func divis6_int32(i int32) bool {
	// 386: "IMUL3L [$]-1431655765,"
	// 386: "ADDL [$]715827882,"
	// 386: "ROLL [$]31,"
	// 386: "CMPL .*, [$]715827882"
	// 386: "SETLS"
	// arm64: "MOVD [$]-1431655765,"
	// arm64: "MULW"
	// arm64: "MOVD [$]715827882,"
	// arm64: "ADD "
	// arm64: "RORW [$]1,"
	// arm64: "CSET LS"
	return i%6 == 0
}

func ndivis6_int32(i int32) bool {
	// 386: "IMUL3L [$]-1431655765,"
	// 386: "ADDL [$]715827882,"
	// 386: "ROLL [$]31,"
	// 386: "CMPL .*, [$]715827882"
	// 386: "SETHI"
	// arm64: "MOVD [$]-1431655765,"
	// arm64: "MULW"
	// arm64: "MOVD [$]715827882,"
	// arm64: "ADD "
	// arm64: "RORW [$]1,"
	// arm64: "CSET HI"
	return i%6 != 0
}

func divis6_int64(i int64) bool {
	// 386: "IMUL3L [$]-1431655766,"
	// 386: "IMUL3L [$]-1431655765,"
	// 386: "ADCL [$]715827882,"
	// 386: "CMPL .*, [$]715827882"
	// 386: "CMPL .*, [$]-1431655766"
	// 386: "SETLS"
	// arm64: "MOVD [$]-6148914691236517205,"
	// arm64: "MUL "
	// arm64: "MOVD [$]3074457345618258602,"
	// arm64: "ADD "
	// arm64: "ROR [$]1,"
	// arm64: "CSET LS"
	return i%6 == 0
}

func ndivis6_int64(i int64) bool {
	// 386: "IMUL3L [$]-1431655766,"
	// 386: "IMUL3L [$]-1431655765,"
	// 386: "ADCL [$]715827882,"
	// 386: "CMPL .*, [$]715827882"
	// 386: "CMPL .*, [$]-1431655766"
	// 386: "SETHI"
	// arm64: "MOVD [$]-6148914691236517205,"
	// arm64: "MUL "
	// arm64: "MOVD [$]3074457345618258602,"
	// arm64: "ADD "
	// arm64: "ROR [$]1,"
	// arm64: "CSET HI"
	return i%6 != 0
}

func div_divis6_uint8(i uint8) (uint8, bool) {
	// 386: "IMUL3L [$]342,"
	// 386: "SHRL [$]11,"
	// 386: "SETEQ"
	// 386: -"RO[RL]"
	// arm64: "MOVD [$]342,"
	// arm64: "MULW"
	// arm64: "UBFX [$]11, R[0-9]+, [$]21,"
	// arm64: "CSET EQ"
	// arm64: -"RO[RL]"
	return i / 6, i%6 == 0
}

func div_ndivis6_uint8(i uint8) (uint8, bool) {
	// 386: "IMUL3L [$]342,"
	// 386: "SHRL [$]11,"
	// 386: "SETNE"
	// 386: -"RO[RL]"
	// arm64: "MOVD [$]342,"
	// arm64: "MULW"
	// arm64: "UBFX [$]11, R[0-9]+, [$]21,"
	// arm64: "CSET NE"
	// arm64: -"RO[RL]"
	return i / 6, i%6 != 0
}

func div_divis6_uint16(i uint16) (uint16, bool) {
	// 386: "IMUL3L [$]43691,"
	// 386: "SHRL [$]18,"
	// 386: "SHLL [$]1,"
	// 386: "SETEQ"
	// 386: -"RO[RL]"
	// arm64: "MOVD [$]87382,"
	// arm64: "MUL "
	// arm64: "LSR [$]19,"
	// arm64: "CSET EQ"
	// arm64: -"RO[RL]"
	return i / 6, i%6 == 0
}

func div_ndivis6_uint16(i uint16) (uint16, bool) {
	// 386: "IMUL3L [$]43691,"
	// 386: "SHRL [$]18,"
	// 386: "SHLL [$]1,"
	// 386: "SETNE"
	// 386: -"RO[RL]"
	// arm64: "MOVD [$]87382,"
	// arm64: "MUL "
	// arm64: "LSR [$]19,"
	// arm64: "CSET NE"
	// arm64: -"RO[RL]"
	return i / 6, i%6 != 0
}

func div_divis6_uint32(i uint32) (uint32, bool) {
	// 386: "MOVL [$]-1431655765,"
	// 386: "SHRL [$]2,"
	// 386: "SHLL [$]1,"
	// 386: "SETEQ"
	// 386: -"RO[RL]"
	// arm64: "MOVD [$]2863311531,"
	// arm64: "MUL "
	// arm64: "LSR [$]34,"
	// arm64: "CSET EQ"
	// arm64: -"RO[RL]"
	return i / 6, i%6 == 0
}

func div_ndivis6_uint32(i uint32) (uint32, bool) {
	// 386: "MOVL [$]-1431655765,"
	// 386: "SHRL [$]2,"
	// 386: "SHLL [$]1,"
	// 386: "SETNE"
	// 386: -"RO[RL]"
	// arm64: "MOVD [$]2863311531,"
	// arm64: "MUL "
	// arm64: "LSR [$]34,"
	// arm64: "CSET NE"
	// arm64: -"RO[RL]"
	return i / 6, i%6 != 0
}

func div_divis6_uint64(i uint64) (uint64, bool) {
	// 386: "MOVL [$]-1431655766,"
	// 386: "MOVL [$]-1431655765,"
	// 386: "MULL"
	// 386: "SHRL [$]2,"
	// 386: "SHLL [$]30,"
	// 386: "SETEQ"
	// 386: -".*CALL"
	// 386: -"RO[RL]"
	// arm64: "MOVD [$]-6148914691236517205,"
	// arm64: "UMULH"
	// arm64: "LSR [$]2,"
	// arm64: "CSET EQ"
	// arm64: -"RO[RL]"
	return i / 6, i%6 == 0
}

func div_ndivis6_uint64(i uint64) (uint64, bool) {
	// 386: "MOVL [$]-1431655766,"
	// 386: "MOVL [$]-1431655765,"
	// 386: "MULL"
	// 386: "SHRL [$]2,"
	// 386: "SHLL [$]30,"
	// 386: "SETNE"
	// 386: -".*CALL"
	// 386: -"RO[RL]"
	// arm64: "MOVD [$]-6148914691236517205,"
	// arm64: "UMULH"
	// arm64: "LSR [$]2,"
	// arm64: "CSET NE"
	// arm64: -"RO[RL]"
	return i / 6, i%6 != 0
}

func div_divis6_int8(i int8) (int8, bool) {
	// 386: "SARL [$]31,"
	// 386: "IMUL3L [$]171,"
	// 386: "SARL [$]10,"
	// 386: "SHLL [$]1,"
	// 386: "SETEQ"
	// 386: -"RO[RL]"
	// arm64: "MOVD [$]171,"
	// arm64: "MULW"
	// arm64: "SBFX [$]10, R[0-9]+, [$]22,"
	// arm64: "SUB R[0-9]+->31,"
	// arm64: "CSET EQ"
	// arm64: -"RO[RL]"
	return i / 6, i%6 == 0
}

func div_ndivis6_int8(i int8) (int8, bool) {
	// 386: "SARL [$]31,"
	// 386: "IMUL3L [$]171,"
	// 386: "SARL [$]10,"
	// 386: "SHLL [$]1,"
	// 386: "SETNE"
	// 386: -"RO[RL]"
	// arm64: "MOVD [$]171,"
	// arm64: "MULW"
	// arm64: "SBFX [$]10, R[0-9]+, [$]22,"
	// arm64: "SUB R[0-9]+->31,"
	// arm64: "CSET NE"
	// arm64: -"RO[RL]"
	return i / 6, i%6 != 0
}

func div_divis6_int16(i int16) (int16, bool) {
	// 386: "SARL [$]31,"
	// 386: "IMUL3L [$]43691,"
	// 386: "SARL [$]18,"
	// 386: "SHLL [$]1,"
	// 386: "SETEQ"
	// 386: -"RO[RL]"
	// arm64: "MOVD [$]43691,"
	// arm64: "MULW"
	// arm64: "SBFX [$]18, R[0-9]+, [$]14,"
	// arm64: "SUB R[0-9]+->31,"
	// arm64: "CSET EQ"
	// arm64: -"RO[RL]"
	return i / 6, i%6 == 0
}

func div_ndivis6_int16(i int16) (int16, bool) {
	// 386: "SARL [$]31,"
	// 386: "IMUL3L [$]43691,"
	// 386: "SARL [$]18,"
	// 386: "SHLL [$]1,"
	// 386: "SETNE"
	// 386: -"RO[RL]"
	// arm64: "MOVD [$]43691,"
	// arm64: "MULW"
	// arm64: "SBFX [$]18, R[0-9]+, [$]14,"
	// arm64: "SUB R[0-9]+->31,"
	// arm64: "CSET NE"
	// arm64: -"RO[RL]"
	return i / 6, i%6 != 0
}

func div_divis6_int32(i int32) (int32, bool) {
	// 386: "SARL [$]31,"
	// 386: "MOVL [$]-1431655765,"
	// 386: "IMULL"
	// 386: "SARL [$]2,"
	// 386: "SHLL [$]1,"
	// 386: "SETEQ"
	// 386: -"RO[RL]"
	// arm64: "MOVD [$]2863311531,"
	// arm64: "MUL "
	// arm64: "ASR [$]34,"
	// arm64: "SUB R[0-9]+->63,"
	// arm64: "CSET EQ"
	// arm64: -"RO[RL]"
	return i / 6, i%6 == 0
}

func div_ndivis6_int32(i int32) (int32, bool) {
	// 386: "SARL [$]31,"
	// 386: "MOVL [$]-1431655765,"
	// 386: "IMULL"
	// 386: "SARL [$]2,"
	// 386: "SHLL [$]1,"
	// 386: "SETNE"
	// 386: -"RO[RL]"
	// arm64: "MOVD [$]2863311531,"
	// arm64: "MUL "
	// arm64: "ASR [$]34,"
	// arm64: "SUB R[0-9]+->63,"
	// arm64: "CSET NE"
	// arm64: -"RO[RL]"
	return i / 6, i%6 != 0
}

func div_divis6_int64(i int64) (int64, bool) {
	// 386: "ANDL [$]-1431655766,"
	// 386: "ANDL [$]-1431655765,"
	// 386: "MOVL [$]-1431655766,"
	// 386: "MOVL [$]-1431655765,"
	// 386: "SUBL" "SBBL"
	// 386: "MULL"
	// 386: "SETEQ"
	// 386: -"SET(LS|HI)"
	// 386: -".*CALL"
	// 386: -"RO[RL]"
	// arm64: "MOVD [$]-6148914691236517205,"
	// arm64: "SMULH"
	// arm64: "ADD"
	// arm64: "ASR [$]2,"
	// arm64: "SUB R[0-9]+->63,"
	// arm64: "CSET EQ"
	// arm64: -"RO[RL]"
	return i / 6, i%6 == 0
}

func div_ndivis6_int64(i int64) (int64, bool) {
	// 386: "ANDL [$]-1431655766,"
	// 386: "ANDL [$]-1431655765,"
	// 386: "MOVL [$]-1431655766,"
	// 386: "MOVL [$]-1431655765,"
	// 386: "SUBL" "SBBL"
	// 386: "MULL"
	// 386: "SETNE"
	// 386: -"SET(LS|HI)"
	// 386: -".*CALL"
	// 386: -"RO[RL]"
	// arm64: "MOVD [$]-6148914691236517205,"
	// arm64: "SMULH"
	// arm64: "ADD"
	// arm64: "ASR [$]2,"
	// arm64: "SUB R[0-9]+->63,"
	// arm64: "CSET NE"
	// arm64: -"RO[RL]"
	return i / 6, i%6 != 0
}
