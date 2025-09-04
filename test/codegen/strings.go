// asmcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

// This file contains code generation tests related to the handling of
// string types.

func CountRunes(s string) int { // Issue #24923
	// amd64:`.*countrunes`
	return len([]rune(s))
}

func CountBytes(s []byte) int {
	// amd64:-`.*runtime.slicebytetostring`
	return len(string(s))
}

func ToByteSlice() []byte { // Issue #24698
	// amd64:`LEAQ\ttype:\[3\]uint8`
	// amd64:`CALL\truntime\.newobject`
	// amd64:-`.*runtime.stringtoslicebyte`
	return []byte("foo")
}

func ConvertToByteSlice(a, b, c string) []byte {
	// amd64:`.*runtime.concatbyte3`
	return []byte(a + b + c)
}

// Loading from read-only symbols should get transformed into constants.
func ConstantLoad() {
	// 12592 = 0x3130
	//    50 = 0x32
	// amd64:`MOVW\t\$12592, \(`,`MOVB\t\$50, 2\(`
	//   386:`MOVW\t\$12592, \(`,`MOVB\t\$50, 2\(`
	//   arm:`MOVW\t\$48`,`MOVW\t\$49`,`MOVW\t\$50`
	// arm64:`MOVD\t\$12592`,`MOVD\t\$50`
	// loong64:`MOVV\t\$12592`,`MOVV\t\$50`
	//  wasm:`I64Const\t\$12592`,`I64Store16\t\$0`,`I64Const\t\$50`,`I64Store8\t\$2`
	// mips64:`MOVV\t\$48`,`MOVV\t\$49`,`MOVV\t\$50`
	bsink = []byte("012")

	// 858927408 = 0x33323130
	//     13620 = 0x3534
	// amd64:`MOVL\t\$858927408`,`MOVW\t\$13620, 4\(`
	//   386:`MOVL\t\$858927408`,`MOVW\t\$13620, 4\(`
	// arm64:`MOVD\t\$858927408`,`MOVD\t\$13620`
	// loong64:`MOVV\t\$858927408`,`MOVV\t\$13620`
	//  wasm:`I64Const\t\$858927408`,`I64Store32\t\$0`,`I64Const\t\$13620`,`I64Store16\t\$4`
	bsink = []byte("012345")

	// 3978425819141910832 = 0x3736353433323130
	// 7306073769690871863 = 0x6564636261393837
	// amd64:`MOVQ\t\$3978425819141910832`,`MOVQ\t\$7306073769690871863`
	//   386:`MOVL\t\$858927408, \(`,`DUFFCOPY`
	// arm64:`MOVD\t\$3978425819141910832`,`MOVD\t\$7306073769690871863`,`MOVD\t\$15`
	// loong64:`MOVV\t\$3978425819141910832`,`MOVV\t\$7306073769690871863`,`MOVV\t\$15`
	//  wasm:`I64Const\t\$3978425819141910832`,`I64Store\t\$0`,`I64Const\t\$7306073769690871863`,`I64Store\t\$7`
	bsink = []byte("0123456789abcde")

	// 56 = 0x38
	// amd64:`MOVQ\t\$3978425819141910832`,`MOVB\t\$56`
	// loong64:`MOVV\t\$3978425819141910832`,`MOVV\t\$56`
	bsink = []byte("012345678")

	// 14648 = 0x3938
	// amd64:`MOVQ\t\$3978425819141910832`,`MOVW\t\$14648`
	// loong64:`MOVV\t\$3978425819141910832`,`MOVV\t\$14648`
	bsink = []byte("0123456789")

	// 1650538808 = 0x62613938
	// amd64:`MOVQ\t\$3978425819141910832`,`MOVL\t\$1650538808`
	// loong64:`MOVV\t\$3978425819141910832`,`MOVV\t\$1650538808`
	bsink = []byte("0123456789ab")
}

// self-equality is always true. See issue 60777.
func EqualSelf(s string) bool {
	// amd64:`MOVL\t\$1, AX`,-`.*memequal.*`
	return s == s
}
func NotEqualSelf(s string) bool {
	// amd64:`XORL\tAX, AX`,-`.*memequal.*`
	return s != s
}

var bsink []byte
