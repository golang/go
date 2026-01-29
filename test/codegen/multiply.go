// asmcheck

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

// This file contains codegen tests related to strength
// reduction of integer multiply.

func m0(x int64) int64 {
	// amd64: "XORL"
	// arm64: "MOVD ZR"
	// loong64: "MOVV R0"
	return x * 0
}
func m2(x int64) int64 {
	// amd64: "ADDQ"
	// arm64: "ADD"
	// loong64: "ADDVU"
	return x * 2
}
func m3(x int64) int64 {
	// amd64: "LEAQ .*[*]2"
	// arm64: "ADD R[0-9]+<<1,"
	// loong64: "ALSLV [$]1,"
	return x * 3
}
func m4(x int64) int64 {
	// amd64: "SHLQ [$]2,"
	// arm64: "LSL [$]2,"
	// loong64: "SLLV [$]2,"
	return x * 4
}
func m5(x int64) int64 {
	// amd64: "LEAQ .*[*]4"
	// arm64: "ADD R[0-9]+<<2,"
	// loong64: "ALSLV [$]2,"
	return x * 5
}
func m6(x int64) int64 {
	// amd64: "LEAQ .*[*]1", "LEAQ .*[*]2"
	// arm64: "ADD R[0-9]+,", "ADD R[0-9]+<<1,"
	// loong64: "ADDVU", "ADDVU", "ADDVU"
	return x * 6
}
func m7(x int64) int64 {
	// amd64: "LEAQ .*[*]2"
	// arm64: "LSL [$]3,", "SUB R[0-9]+,"
	// loong64: "ALSLV [$]1,", "ALSLV [$]1,"
	return x * 7
}
func m8(x int64) int64 {
	// amd64: "SHLQ [$]3,"
	// arm64: "LSL [$]3,"
	// loong64: "SLLV [$]3,"
	return x * 8
}
func m9(x int64) int64 {
	// amd64: "LEAQ .*[*]8"
	// arm64: "ADD R[0-9]+<<3,"
	// loong64: "ALSLV [$]3,"
	return x * 9
}
func m10(x int64) int64 {
	// amd64: "LEAQ .*[*]1", "LEAQ .*[*]4"
	// arm64: "ADD R[0-9]+,", "ADD R[0-9]+<<2,"
	// loong64: "ADDVU", "ALSLV [$]2,"
	return x * 10
}
func m11(x int64) int64 {
	// amd64: "LEAQ .*[*]4", "LEAQ .*[*]2"
	// arm64: "MOVD [$]11,", "MUL"
	// loong64: "ALSLV [$]2,", "ALSLV [$]1,"
	return x * 11
}
func m12(x int64) int64 {
	// amd64: "LEAQ .*[*]2", "SHLQ [$]2,"
	// arm64: "LSL [$]2,", "ADD R[0-9]+<<1,"
	// loong64: "SLLV", "ALSLV [$]1,"
	return x * 12
}
func m13(x int64) int64 {
	// amd64: "LEAQ .*[*]2", "LEAQ .*[*]4"
	// arm64: "MOVD [$]13,", "MUL"
	// loong64: "ALSLV [$]1,", "ALSLV [$]2,"
	return x * 13
}
func m14(x int64) int64 {
	// amd64: "IMUL3Q [$]14,"
	// arm64: "LSL [$]4,", "SUB R[0-9]+<<1,"
	// loong64: "ADDVU", "ALSLV [$]1", "ALSLV [$]2"
	return x * 14
}
func m15(x int64) int64 {
	// amd64: "LEAQ .*[*]2", "LEAQ .*[*]4"
	// arm64: "LSL [$]4,", "SUB R[0-9]+,"
	// loong64: "ALSLV [$]1,", "ALSLV [$]2,"
	return x * 15
}
func m16(x int64) int64 {
	// amd64: "SHLQ [$]4,"
	// arm64: "LSL [$]4,"
	// loong64: "SLLV [$]4,"
	return x * 16
}
func m17(x int64) int64 {
	// amd64: "LEAQ .*[*]1", "LEAQ .*[*]8"
	// arm64: "ADD R[0-9]+<<4,"
	// loong64: "ALSLV [$]"
	return x * 17
}
func m18(x int64) int64 {
	// amd64: "LEAQ .*[*]1", "LEAQ .*[*]8"
	// arm64: "ADD R[0-9]+,", "ADD R[0-9]+<<3,"
	// loong64: "ADDVU", "ALSLV [$]3,"
	return x * 18
}
func m19(x int64) int64 {
	// amd64: "LEAQ .*[*]8", "LEAQ .*[*]2"
	// arm64: "MOVD [$]19,", "MUL"
	// loong64: "ALSLV [$]3,", "ALSLV [$]1,"
	return x * 19
}
func m20(x int64) int64 {
	// amd64: "LEAQ .*[*]4", "SHLQ [$]2,"
	// arm64: "LSL [$]2,", "ADD R[0-9]+<<2,"
	// loong64: "SLLV [$]2,", "ALSLV [$]2," 
	return x * 20
}
func m21(x int64) int64 {
	// amd64: "LEAQ .*[*]4", "LEAQ .*[*]4"
	// arm64: "MOVD [$]21,", "MUL"
	// loong64: "ALSLV [$]2,", "ALSLV [$]2,"
	return x * 21
}
func m22(x int64) int64 {
	// amd64: "IMUL3Q [$]22,"
	// arm64: "MOVD [$]22,", "MUL"
	// loong64: "ADDVU", "ALSLV [$]2,", "ALSLV [$]2,"
	return x * 22
}
func m23(x int64) int64 {
	// amd64: "IMUL3Q [$]23,"
	// arm64: "MOVD [$]23,", "MUL"
	// loong64: "ALSLV [$]1,", "SUBVU", "ALSLV [$]3,"
	return x * 23
}
func m24(x int64) int64 {
	// amd64: "LEAQ .*[*]2", "SHLQ [$]3,"
	// arm64: "LSL [$]3,", "ADD R[0-9]+<<1,"
	// loong64: "SLLV [$]3", "ALSLV [$]1,"
	return x * 24
}
func m25(x int64) int64 {
	// amd64: "LEAQ .*[*]4", "LEAQ .*[*]4"
	// arm64: "MOVD [$]25,", "MUL"
	// loong64: "ALSLV [$]2,", "ALSLV [$]2,"
	return x * 25
}
func m26(x int64) int64 {
	// amd64: "IMUL3Q [$]26,"
	// arm64: "MOVD [$]26,", "MUL"
	// loong64: "ADDVU", "ALSLV [$]1,", "ALSLV [$]3,"
	return x * 26
}
func m27(x int64) int64 {
	// amd64: "LEAQ .*[*]2", "LEAQ .*[*]8"
	// arm64: "MOVD [$]27,", "MUL"
	// loong64: "ALSLV [$]1,", "ALSLV [$]3,"
	return x * 27
}
func m28(x int64) int64 {
	// amd64: "IMUL3Q [$]28,"
	// arm64: "LSL [$]5, "SUB R[0-9]+<<2,"
	// loong64: "ALSLV [$]1," "SLLV [$]2," "ALSLV [$]3,"
	return x * 28
}
func m29(x int64) int64 {
	// amd64: "IMUL3Q [$]29,"
	// arm64: "MOVD [$]29,", "MUL"
	// loong64: "ALSLV [$]1," "SLLV [$]5," "SUBVU"
	return x * 29
}
func m30(x int64) int64 {
	// amd64: "IMUL3Q [$]30,"
	// arm64: "LSL [$]5,", "SUB R[0-9]+<<1,"
	// loong64: "ADDVU" "SLLV [$]5," "SUBVU"
	return x * 30
}
func m31(x int64) int64 {
	// amd64: "SHLQ [$]5,", "SUBQ"
	// arm64: "LSL [$]5,", "SUB R[0-9]+,"
	// loong64: "SLLV [$]5," "SUBVU"
	return x * 31
}
func m32(x int64) int64 {
	// amd64: "SHLQ [$]5,"
	// arm64: "LSL [$]5,"
	// loong64: "SLLV [$]5,"
	return x * 32
}
func m33(x int64) int64 {
	// amd64: "SHLQ [$]2,", "LEAQ .*[*]8"
	// arm64: "ADD R[0-9]+<<5,"
	// loong64: "ADDVU", "ALSLV [$]4,"
	return x * 33
}
func m34(x int64) int64 {
	// amd64: "SHLQ [$]5,", "LEAQ .*[*]2"
	// arm64: "ADD R[0-9]+,", "ADD R[0-9]+<<4,"
	// loong64: "ADDVU", "ALSLV [$]4,"
	return x * 34
}
func m35(x int64) int64 {
	// amd64: "IMUL3Q [$]35,"
	// arm64: "MOVD [$]35,", "MUL"
	// loong64: "ALSLV [$]4,", "ALSLV [$]1,"
	return x * 35
}
func m36(x int64) int64 {
	// amd64: "LEAQ .*[*]8", "SHLQ [$]2,"
	// arm64: "LSL [$]2,", "ADD R[0-9]+<<3,"
	// loong64: "SLLV [$]2,", "ALSLV [$]3,"
	return x * 36
}
func m37(x int64) int64 {
	// amd64: "LEAQ .*[*]8", "LEAQ .*[*]4"
	// arm64: "MOVD [$]37,", "MUL"
	// loong64: "ALSLV [$]3,", "ALSLV [$]2,"
	return x * 37
}
func m38(x int64) int64 {
	// amd64: "IMUL3Q [$]38,"
	// arm64: "MOVD [$]38,", "MUL"
	// loong64: "ALSLV [$]3,", "ALSLV [$]2,"
	return x * 38
}
func m39(x int64) int64 {
	// amd64: "IMUL3Q [$]39,"
	// arm64: "MOVD [$]39,", "MUL"
	// loong64: "ALSLV [$]2,", "SUBVU", "ALSLV [$]3,"
	return x * 39
}
func m40(x int64) int64 {
	// amd64: "LEAQ .*[*]4", "SHLQ [$]3,"
	// arm64: "LSL [$]3,", "ADD R[0-9]+<<2,"
	// loong64: "SLLV [$]3,", "ALSLV [$]2,"
	return x * 40
}

func mn1(x int64) int64 {
	// amd64: "NEGQ "
	// arm64: "NEG R[0-9]+,"
	// loong64: "SUBVU R[0-9], R0,"
	return x * -1
}
func mn2(x int64) int64 {
	// amd64: "NEGQ", "ADDQ"
	// arm64: "NEG R[0-9]+<<1,"
	// loong64: "ADDVU" "SUBVU R[0-9], R0,"
	return x * -2
}
func mn3(x int64) int64 {
	// amd64: "NEGQ", "LEAQ .*[*]2"
	// arm64: "SUB R[0-9]+<<2,"
	// loong64: "SUBVU", "ALSLV [$]1,"
	return x * -3
}
func mn4(x int64) int64 {
	// amd64: "NEGQ", "SHLQ [$]2,"
	// arm64: "NEG R[0-9]+<<2,"
	// loong64: "SLLV [$]2," "SUBVU R[0-9], R0,"
	return x * -4
}
func mn5(x int64) int64 {
	// amd64: "NEGQ", "LEAQ .*[*]4"
	// arm64: "NEG R[0-9]+,", "ADD R[0-9]+<<2,"
	// loong64: "SUBVU", "ALSLV [$]2,"
	return x * -5
}
func mn6(x int64) int64 {
	// amd64: "IMUL3Q [$]-6,"
	// arm64: "ADD R[0-9]+,", "SUB R[0-9]+<<2,"
	// loong64: "ADDVU", "SUBVU", "ALSLV [$]3,"
	return x * -6
}
func mn7(x int64) int64 {
	// amd64: "NEGQ", "LEAQ .*[*]8"
	// arm64: "SUB R[0-9]+<<3,"
	// loong64: "SUBVU", "ALSLV [$]3,"
	return x * -7
}
func mn8(x int64) int64 {
	// amd64: "NEGQ", "SHLQ [$]3,"
	// arm64: "NEG R[0-9]+<<3,"
	// loong64: "SLLV [$]3" "SUBVU R[0-9], R0,"
	return x * -8
}
func mn9(x int64) int64 {
	// amd64: "NEGQ", "LEAQ .*[*]8"
	// arm64: "NEG R[0-9]+,", "ADD R[0-9]+<<3,"
	// loong64: "SUBVU", "ALSLV [$]3,"
	return x * -9
}
func mn10(x int64) int64 {
	// amd64: "IMUL3Q [$]-10,"
	// arm64: "MOVD [$]-10,", "MUL"
	// loong64: "ADDVU", "ALSLV [$]3", "SUBVU"
	return x * -10
}
func mn11(x int64) int64 {
	// amd64: "IMUL3Q [$]-11,"
	// arm64: "MOVD [$]-11,", "MUL"
	// loong64: "ALSLV [$]2,", "SUBVU", "ALSLV [$]4,"
	return x * -11
}
func mn12(x int64) int64 {
	// amd64: "IMUL3Q [$]-12,"
	// arm64: "LSL [$]2,", "SUB R[0-9]+<<2,"
	// loong64: "SUBVU", "SLLV [$]2,", "ALSLV [$]4,"
	return x * -12
}
func mn13(x int64) int64 {
	// amd64: "IMUL3Q [$]-13,"
	// arm64: "MOVD [$]-13,", "MUL"
	// loong64: "ALSLV [$]4,", "SLLV [$]2, ", "SUBVU"
	return x * -13
}
func mn14(x int64) int64 {
	// amd64: "IMUL3Q [$]-14,"
	// arm64: "ADD R[0-9]+,", "SUB R[0-9]+<<3,"
	// loong64: "ADDVU", "SUBVU", "ALSLV [$]4,"
	return x * -14
}
func mn15(x int64) int64 {
	// amd64: "SHLQ [$]4,", "SUBQ"
	// arm64: "SUB R[0-9]+<<4,"
	// loong64: "SUBVU", "ALSLV [$]4,"
	return x * -15
}
func mn16(x int64) int64 {
	// amd64: "NEGQ", "SHLQ [$]4,"
	// arm64: "NEG R[0-9]+<<4,"
	// loong64: "SLLV [$]4," "SUBVU R[0-9], R0,"
	return x * -16
}
func mn17(x int64) int64 {
	// amd64: "IMUL3Q [$]-17,"
	// arm64: "NEG R[0-9]+,", "ADD R[0-9]+<<4,"
	// loong64: "SUBVU", "ALSLV [$]4,"
	return x * -17
}
func mn18(x int64) int64 {
	// amd64: "IMUL3Q [$]-18,"
	// arm64: "MOVD [$]-18,", "MUL"
	// loong64: "ADDVU", "ALSLV [$]4,", "SUBVU"
	return x * -18
}
func mn19(x int64) int64 {
	// amd64: "IMUL3Q [$]-19,"
	// arm64: "MOVD [$]-19,", "MUL"
	// loong64: "ALSLV [$]1,", "ALSLV [$]4,", "SUBVU"
	return x * -19
}
func mn20(x int64) int64 {
	// amd64: "IMUL3Q [$]-20,"
	// arm64: "MOVD [$]-20,", "MUL"
	// loong64: "SLLV [$]2,", "ALSLV [$]4,", "SUBVU"
	return x * -20
}
