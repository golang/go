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

func zeroExtArgByte(ch [2]byte) bool {
	return wsp[ch[0]] // amd64:-"MOVBLZX\t..,.."
}

func zeroExtArgUint16(ch [2]uint16) bool {
	return wsp[ch[0]] // amd64:-"MOVWLZX\t..,.."
}
