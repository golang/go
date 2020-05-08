// asmcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

var wsp = [256]bool{
	' ':  true,
	'\t': true,
	'\n': true,
	'\r': true,
}

func zeroExtArgByte(ch byte) bool {
	return wsp[ch] // amd64:-"MOVBLZX\t..,.."
}

func zeroExtArgUint16(ch uint16) bool {
	return wsp[ch] // amd64:-"MOVWLZX\t..,.."
}
