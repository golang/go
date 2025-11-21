// asmcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

import "strings"

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
	// amd64:`LEAQ type:\[3\]uint8`
	// amd64:`CALL runtime\.(newobject|mallocTiny3)`
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
	// amd64:`MOVW \$12592, \(`,`MOVB \$50, 2\(`
	//   386:`MOVW \$12592, \(`,`MOVB \$50, 2\(`
	//   arm:`MOVW \$48`,`MOVW \$49`,`MOVW \$50`
	// arm64:`MOVD \$12592`,`MOVD \$50`
	// loong64:`MOVV \$12592`,`MOVV \$50`
	//  wasm:`I64Const \$12592`,`I64Store16 \$0`,`I64Const \$50`,`I64Store8 \$2`
	// mips64:`MOVV \$48`,`MOVV \$49`,`MOVV \$50`
	bsink = []byte("012")

	// 858927408 = 0x33323130
	//     13620 = 0x3534
	// amd64:`MOVL \$858927408`,`MOVW \$13620, 4\(`
	//   386:`MOVL \$858927408`,`MOVW \$13620, 4\(`
	// arm64:`MOVD \$858927408`,`MOVD \$13620`
	// loong64:`MOVV \$858927408`,`MOVV \$13620`
	//  wasm:`I64Const \$858927408`,`I64Store32 \$0`,`I64Const \$13620`,`I64Store16 \$4`
	bsink = []byte("012345")

	// 3978425819141910832 = 0x3736353433323130
	// 7306073769690871863 = 0x6564636261393837
	// amd64:`MOVQ \$3978425819141910832`,`MOVQ \$7306073769690871863`
	//   386:`MOVL \$858927408, \(`,`DUFFCOPY`
	// arm64:`MOVD \$3978425819141910832`,`MOVD \$7306073769690871863`,`MOVD \$15`
	// loong64:`MOVV \$3978425819141910832`,`MOVV \$7306073769690871863`,`MOVV \$15`
	//  wasm:`I64Const \$3978425819141910832`,`I64Store \$0`,`I64Const \$7306073769690871863`,`I64Store \$7`
	bsink = []byte("0123456789abcde")

	// 56 = 0x38
	// amd64:`MOVQ \$3978425819141910832`,`MOVB \$56`
	// loong64:`MOVV \$3978425819141910832`,`MOVV \$56`
	bsink = []byte("012345678")

	// 14648 = 0x3938
	// amd64:`MOVQ \$3978425819141910832`,`MOVW \$14648`
	// loong64:`MOVV \$3978425819141910832`,`MOVV \$14648`
	bsink = []byte("0123456789")

	// 1650538808 = 0x62613938
	// amd64:`MOVQ \$3978425819141910832`,`MOVL \$1650538808`
	// loong64:`MOVV \$3978425819141910832`,`MOVV \$1650538808`
	bsink = []byte("0123456789ab")
}

// self-equality is always true. See issue 60777.
func EqualSelf(s string) bool {
	// amd64:`MOVL \$1, AX`,-`.*memequal.*`
	return s == s
}
func NotEqualSelf(s string) bool {
	// amd64:`XORL AX, AX`,-`.*memequal.*`
	return s != s
}

var bsink []byte

func HasPrefix3(s string) bool {
	// amd64:-`.*memequal.*`
	return strings.HasPrefix(s, "str")
}

func HasPrefix5(s string) bool {
	// amd64:-`.*memequal.*`
	return strings.HasPrefix(s, "strin")
}

func HasPrefix6(s string) bool {
	// amd64:-`.*memequal.*`
	return strings.HasPrefix(s, "string")
}

func HasPrefix7(s string) bool {
	// amd64:-`.*memequal.*`
	return strings.HasPrefix(s, "strings")
}
