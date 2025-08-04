// asmcheck

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

// This file contains codegen tests related to strength
// reduction of integer multiply.

func m0(x int64) int64 {
	// amd64: "XORL"
	// arm64: "MOVD\tZR"
	// loong64: "MOVV\t[$]0"
	return x * 0
}
func m2(x int64) int64 {
	// amd64: "ADDQ"
	// arm64: "ADD"
	// loong64: "ADDVU"
	return x * 2
}
func m3(x int64) int64 {
	// amd64: "LEAQ\t.*[*]2"
	// arm64: "ADD\tR[0-9]+<<1,"
	// loong64: "ADDVU","ADDVU"
	return x * 3
}
func m4(x int64) int64 {
	// amd64: "SHLQ\t[$]2,"
	// arm64: "LSL\t[$]2,"
	// loong64: "SLLV\t[$]2,"
	return x * 4
}
func m5(x int64) int64 {
	// amd64: "LEAQ\t.*[*]4"
	// arm64: "ADD\tR[0-9]+<<2,"
	// loong64: "SLLV\t[$]2,","ADDVU"
	return x * 5
}
func m6(x int64) int64 {
	// amd64: "LEAQ\t.*[*]1", "LEAQ\t.*[*]2"
	// arm64: "ADD\tR[0-9]+,", "ADD\tR[0-9]+<<1,"
	// loong64: "ADDVU","ADDVU","ADDVU"
	return x * 6
}
func m7(x int64) int64 {
	// amd64: "LEAQ\t.*[*]2"
	// arm64: "LSL\t[$]3,", "SUB\tR[0-9]+,"
	// loong64: "SLLV\t[$]3,","SUBVU"
	return x * 7
}
func m8(x int64) int64 {
	// amd64: "SHLQ\t[$]3,"
	// arm64: "LSL\t[$]3,"
	// loong64: "SLLV\t[$]3,"
	return x * 8
}
func m9(x int64) int64 {
	// amd64: "LEAQ\t.*[*]8"
	// arm64: "ADD\tR[0-9]+<<3,"
	// loong64: "SLLV\t[$]3,","ADDVU"
	return x * 9
}
func m10(x int64) int64 {
	// amd64: "LEAQ\t.*[*]1", "LEAQ\t.*[*]4"
	// arm64: "ADD\tR[0-9]+,", "ADD\tR[0-9]+<<2,"
	// loong64: "ADDVU","SLLV\t[$]3,","ADDVU"
	return x * 10
}
func m11(x int64) int64 {
	// amd64: "LEAQ\t.*[*]4", "LEAQ\t.*[*]2"
	// arm64: "MOVD\t[$]11,", "MUL"
	// loong64: "MOVV\t[$]11,", "MULV"
	return x * 11
}
func m12(x int64) int64 {
	// amd64: "LEAQ\t.*[*]2", "SHLQ\t[$]2,"
	// arm64: "LSL\t[$]2,", "ADD\tR[0-9]+<<1,"
	// loong64: "ADDVU","ADDVU","SLLV\t[$]2,"
	return x * 12
}
func m13(x int64) int64 {
	// amd64: "LEAQ\t.*[*]2", "LEAQ\t.*[*]4"
	// arm64: "MOVD\t[$]13,", "MUL"
	// loong64: "MOVV\t[$]13,","MULV"
	return x * 13
}
func m14(x int64) int64 {
	// amd64: "IMUL3Q\t[$]14,"
	// arm64: "LSL\t[$]4,", "SUB\tR[0-9]+<<1,"
	// loong64: "ADDVU","SLLV\t[$]4,","SUBVU"
	return x * 14
}
func m15(x int64) int64 {
	// amd64: "LEAQ\t.*[*]2", "LEAQ\t.*[*]4"
	// arm64: "LSL\t[$]4,", "SUB\tR[0-9]+,"
	// loong64: "SLLV\t[$]4,","SUBVU"
	return x * 15
}
func m16(x int64) int64 {
	// amd64: "SHLQ\t[$]4,"
	// arm64: "LSL\t[$]4,"
	// loong64: "SLLV\t[$]4,"
	return x * 16
}
func m17(x int64) int64 {
	// amd64: "LEAQ\t.*[*]1", "LEAQ\t.*[*]8"
	// arm64: "ADD\tR[0-9]+<<4,"
	// loong64: "SLLV\t[$]4,","ADDVU"
	return x * 17
}
func m18(x int64) int64 {
	// amd64: "LEAQ\t.*[*]1", "LEAQ\t.*[*]8"
	// arm64: "ADD\tR[0-9]+,", "ADD\tR[0-9]+<<3,"
	// loong64: "ADDVU","SLLV\t[$]4,","ADDVU"
	return x * 18
}
func m19(x int64) int64 {
	// amd64: "LEAQ\t.*[*]8", "LEAQ\t.*[*]2"
	// arm64: "MOVD\t[$]19,", "MUL"
	// loong64: "MOVV\t[$]19,","MULV"
	return x * 19
}
func m20(x int64) int64 {
	// amd64: "LEAQ\t.*[*]4", "SHLQ\t[$]2,"
	// arm64: "LSL\t[$]2,", "ADD\tR[0-9]+<<2,"
	// loong64: "SLLV\t[$]2,","SLLV\t[$]4,","ADDVU"
	return x * 20
}
func m21(x int64) int64 {
	// amd64: "LEAQ\t.*[*]4", "LEAQ\t.*[*]4"
	// arm64: "MOVD\t[$]21,", "MUL"
	// loong64: "MOVV\t[$]21,","MULV"
	return x * 21
}
func m22(x int64) int64 {
	// amd64: "IMUL3Q\t[$]22,"
	// arm64: "MOVD\t[$]22,", "MUL"
	// loong64: "MOVV\t[$]22,","MULV"
	return x * 22
}
func m23(x int64) int64 {
	// amd64: "IMUL3Q\t[$]23,"
	// arm64: "MOVD\t[$]23,", "MUL"
	// loong64: "MOVV\t[$]23,","MULV"
	return x * 23
}
func m24(x int64) int64 {
	// amd64: "LEAQ\t.*[*]2", "SHLQ\t[$]3,"
	// arm64: "LSL\t[$]3,", "ADD\tR[0-9]+<<1,"
	// loong64: "ADDVU","ADDVU","SLLV\t[$]3,"
	return x * 24
}
func m25(x int64) int64 {
	// amd64: "LEAQ\t.*[*]4", "LEAQ\t.*[*]4"
	// arm64: "MOVD\t[$]25,", "MUL"
	// loong64: "MOVV\t[$]25,","MULV"
	return x * 25
}
func m26(x int64) int64 {
	// amd64: "IMUL3Q\t[$]26,"
	// arm64: "MOVD\t[$]26,", "MUL"
	// loong64: "MOVV\t[$]26,","MULV"
	return x * 26
}
func m27(x int64) int64 {
	// amd64: "LEAQ\t.*[*]2", "LEAQ\t.*[*]8"
	// arm64: "MOVD\t[$]27,", "MUL"
	// loong64: "MOVV\t[$]27,","MULV"
	return x * 27
}
func m28(x int64) int64 {
	// amd64: "IMUL3Q\t[$]28,"
	// arm64: "LSL\t[$]5, "SUB\tR[0-9]+<<2,"
	// loong64: "SLLV\t[$]5,","SLLV\t[$]2,","SUBVU"
	return x * 28
}
func m29(x int64) int64 {
	// amd64: "IMUL3Q\t[$]29,"
	// arm64: "MOVD\t[$]29,", "MUL"
	// loong64: "MOVV\t[$]29,","MULV"
	return x * 29
}
func m30(x int64) int64 {
	// amd64: "IMUL3Q\t[$]30,"
	// arm64: "LSL\t[$]5,", "SUB\tR[0-9]+<<1,"
	// loong64: "ADDVU","SLLV\t[$]5,","SUBVU"
	return x * 30
}
func m31(x int64) int64 {
	// amd64: "SHLQ\t[$]5,", "SUBQ"
	// arm64: "LSL\t[$]5,", "SUB\tR[0-9]+,"
	// loong64: "SLLV\t[$]5,","SUBVU"
	return x * 31
}
func m32(x int64) int64 {
	// amd64: "SHLQ\t[$]5,"
	// arm64: "LSL\t[$]5,"
	// loong64: "SLLV\t[$]5,"
	return x * 32
}
func m33(x int64) int64 {
	// amd64: "SHLQ\t[$]2,", "LEAQ\t.*[*]8"
	// arm64: "ADD\tR[0-9]+<<5,"
	// loong64: "SLLV\t[$]5,","ADDVU"
	return x * 33
}
func m34(x int64) int64 {
	// amd64: "SHLQ\t[$]5,", "LEAQ\t.*[*]2"
	// arm64: "ADD\tR[0-9]+,", "ADD\tR[0-9]+<<4,"
	// loong64: "ADDVU","SLLV\t[$]5,","ADDVU"
	return x * 34
}
func m35(x int64) int64 {
	// amd64: "IMUL3Q\t[$]35,"
	// arm64: "MOVD\t[$]35,", "MUL"
	// loong64: "MOVV\t[$]35,","MULV"
	return x * 35
}
func m36(x int64) int64 {
	// amd64: "LEAQ\t.*[*]8", "SHLQ\t[$]2,"
	// arm64: "LSL\t[$]2,", "ADD\tR[0-9]+<<3,"
	// loong64: "SLLV\t[$]2,","SLLV\t[$]5,","ADDVU"
	return x * 36
}
func m37(x int64) int64 {
	// amd64: "LEAQ\t.*[*]8", "LEAQ\t.*[*]4"
	// arm64: "MOVD\t[$]37,", "MUL"
	// loong64: "MOVV\t[$]37,","MULV"
	return x * 37
}
func m38(x int64) int64 {
	// amd64: "IMUL3Q\t[$]38,"
	// arm64: "MOVD\t[$]38,", "MUL"
	// loong64: "MOVV\t[$]38,","MULV"
	return x * 38
}
func m39(x int64) int64 {
	// amd64: "IMUL3Q\t[$]39,"
	// arm64: "MOVD\t[$]39,", "MUL"
	// loong64: "MOVV\t[$]39,", "MULV"
	return x * 39
}
func m40(x int64) int64 {
	// amd64: "LEAQ\t.*[*]4", "SHLQ\t[$]3,"
	// arm64: "LSL\t[$]3,", "ADD\tR[0-9]+<<2,"
	// loong64: "SLLV\t[$]3,","SLLV\t[$]5,","ADDVU"
	return x * 40
}

func mn1(x int64) int64 {
	// amd64: "NEGQ\t"
	// arm64: "NEG\tR[0-9]+,"
	// loong64: "SUBVU\tR[0-9], R0,"
	return x * -1
}
func mn2(x int64) int64 {
	// amd64: "NEGQ", "ADDQ"
	// arm64: "NEG\tR[0-9]+<<1,"
	// loong64: "ADDVU","SUBVU\tR[0-9], R0,"
	return x * -2
}
func mn3(x int64) int64 {
	// amd64: "NEGQ", "LEAQ\t.*[*]2"
	// arm64: "SUB\tR[0-9]+<<2,"
	// loong64: "SLLV\t[$]2,","SUBVU"
	return x * -3
}
func mn4(x int64) int64 {
	// amd64: "NEGQ", "SHLQ\t[$]2,"
	// arm64: "NEG\tR[0-9]+<<2,"
	// loong64: "SLLV\t[$]2,","SUBVU\tR[0-9], R0,"
	return x * -4
}
func mn5(x int64) int64 {
	// amd64: "NEGQ", "LEAQ\t.*[*]4"
	// arm64: "NEG\tR[0-9]+,", "ADD\tR[0-9]+<<2,"
	// loong64: "SUBVU\tR[0-9], R0,","SLLV\t[$]2,","SUBVU"
	return x * -5
}
func mn6(x int64) int64 {
	// amd64: "IMUL3Q\t[$]-6,"
	// arm64: "ADD\tR[0-9]+,", "SUB\tR[0-9]+<<2,"
	// loong64: "ADDVU","SLLV\t[$]3,","SUBVU"
	return x * -6
}
func mn7(x int64) int64 {
	// amd64: "NEGQ", "LEAQ\t.*[*]8"
	// arm64: "SUB\tR[0-9]+<<3,"
	// loong64: "SLLV\t[$]3","SUBVU"
	return x * -7
}
func mn8(x int64) int64 {
	// amd64: "NEGQ", "SHLQ\t[$]3,"
	// arm64: "NEG\tR[0-9]+<<3,"
	// loong64: "SLLV\t[$]3","SUBVU\tR[0-9], R0,"
	return x * -8
}
func mn9(x int64) int64 {
	// amd64: "NEGQ", "LEAQ\t.*[*]8"
	// arm64: "NEG\tR[0-9]+,", "ADD\tR[0-9]+<<3,"
	// loong64: "SUBVU\tR[0-9], R0,","SLLV\t[$]3","SUBVU"
	return x * -9
}
func mn10(x int64) int64 {
	// amd64: "IMUL3Q\t[$]-10,"
	// arm64: "MOVD\t[$]-10,", "MUL"
	// loong64: "MOVV\t[$]-10","MULV"
	return x * -10
}
func mn11(x int64) int64 {
	// amd64: "IMUL3Q\t[$]-11,"
	// arm64: "MOVD\t[$]-11,", "MUL"
	// loong64: "MOVV\t[$]-11","MULV"
	return x * -11
}
func mn12(x int64) int64 {
	// amd64: "IMUL3Q\t[$]-12,"
	// arm64: "LSL\t[$]2,", "SUB\tR[0-9]+<<2,"
	// loong64: "SLLV\t[$]2,","SLLV\t[$]4,","SUBVU"
	return x * -12
}
func mn13(x int64) int64 {
	// amd64: "IMUL3Q\t[$]-13,"
	// arm64: "MOVD\t[$]-13,", "MUL"
	// loong64: "MOVV\t[$]-13","MULV"
	return x * -13
}
func mn14(x int64) int64 {
	// amd64: "IMUL3Q\t[$]-14,"
	// arm64: "ADD\tR[0-9]+,", "SUB\tR[0-9]+<<3,"
	// loong64: "ADDVU","SLLV\t[$]4,","SUBVU"
	return x * -14
}
func mn15(x int64) int64 {
	// amd64: "SHLQ\t[$]4,", "SUBQ"
	// arm64: "SUB\tR[0-9]+<<4,"
	// loong64: "SLLV\t[$]4,","SUBVU"
	return x * -15
}
func mn16(x int64) int64 {
	// amd64: "NEGQ", "SHLQ\t[$]4,"
	// arm64: "NEG\tR[0-9]+<<4,"
	// loong64: "SLLV\t[$]4,","SUBVU\tR[0-9], R0,"
	return x * -16
}
func mn17(x int64) int64 {
	// amd64: "IMUL3Q\t[$]-17,"
	// arm64: "NEG\tR[0-9]+,", "ADD\tR[0-9]+<<4,"
	// loong64: "SUBVU\tR[0-9], R0,","SLLV\t[$]4,","SUBVU"
	return x * -17
}
func mn18(x int64) int64 {
	// amd64: "IMUL3Q\t[$]-18,"
	// arm64: "MOVD\t[$]-18,", "MUL"
	// loong64: "MOVV\t[$]-18","MULV"
	return x * -18
}
func mn19(x int64) int64 {
	// amd64: "IMUL3Q\t[$]-19,"
	// arm64: "MOVD\t[$]-19,", "MUL"
	// loong64: "MOVV\t[$]-19","MULV"
	return x * -19
}
func mn20(x int64) int64 {
	// amd64: "IMUL3Q\t[$]-20,"
	// arm64: "MOVD\t[$]-20,", "MUL"
	// loong64: "MOVV\t[$]-20","MULV"
	return x * -20
}
