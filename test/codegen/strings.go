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

func ToByteSlice() []byte { // Issue #24698
	// amd64:`LEAQ\ttype\.\[3\]uint8`
	// amd64:`CALL\truntime\.newobject`
	// amd64:-`.*runtime.stringtoslicebyte`
	return []byte("foo")
}

// Loading from read-only symbols should get transformed into constants.
func ConstantLoad() {
	// 12592 = 0x3130
	//    50 = 0x32
	// amd64:`MOVW\t\$12592, \(`,`MOVB\t\$50, 2\(`
	//   386:`MOVW\t\$12592, \(`,`MOVB\t\$50, 2\(`
	//   arm:`MOVW\t\$48`,`MOVW\t\$49`,`MOVW\t\$50`
	// arm64:`MOVD\t\$12592`,`MOVD\t\$50`
	bsink = []byte("012")

	// 858927408 = 0x33323130
	//     13620 = 0x3534
	// amd64:`MOVL\t\$858927408`,`MOVW\t\$13620, 4\(`
	//   386:`MOVL\t\$858927408`,`MOVW\t\$13620, 4\(`
	// arm64:`MOVD\t\$858927408`,`MOVD\t\$13620`
	bsink = []byte("012345")

	// 3978425819141910832 = 0x3736353433323130
	// 7306073769690871863 = 0x6564636261393837
	// amd64:`MOVQ\t\$3978425819141910832`,`MOVQ\t\$7306073769690871863`
	//   386:`MOVL\t\$858927408, \(`,`DUFFCOPY`
	// arm64:`MOVD\t\$3978425819141910832`,`MOVD\t\$1650538808`,`MOVD\t\$25699`,`MOVD\t\$101`
	bsink = []byte("0123456789abcde")

	// 56 = 0x38
	// amd64:`MOVQ\t\$3978425819141910832`,`MOVB\t\$56`
	bsink = []byte("012345678")

	// 14648 = 0x3938
	// amd64:`MOVQ\t\$3978425819141910832`,`MOVW\t\$14648`
	bsink = []byte("0123456789")

	// 1650538808 = 0x62613938
	// amd64:`MOVQ\t\$3978425819141910832`,`MOVL\t\$1650538808`
	bsink = []byte("0123456789ab")
}

var bsink []byte
