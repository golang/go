// asmcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

import "runtime"

// Check small copies are replaced with moves.

func movesmall4() {
	x := [...]byte{1, 2, 3, 4}
	// 386:-".*memmove"
	// amd64:-".*memmove"
	// arm:-".*memmove"
	// arm64:-".*memmove"
	// ppc64x:-".*memmove"
	copy(x[1:], x[:])
}

func movesmall7() {
	x := [...]byte{1, 2, 3, 4, 5, 6, 7}
	// 386:-".*memmove"
	// amd64:-".*memmove"
	// arm64:-".*memmove"
	// ppc64x:-".*memmove"
	copy(x[1:], x[:])
}

func movesmall16() {
	x := [...]byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	// amd64:-".*memmove"
	// ppc64x:".*memmove"
	copy(x[1:], x[:])
}

var x [256]byte

// Check that large disjoint copies are replaced with moves.

func moveDisjointStack32() {
	var s [32]byte
	// ppc64x:-".*memmove"
	// ppc64x/power8:"LXVD2X",-"ADD",-"BC"
	// ppc64x/power9:"LXV",-"LXVD2X",-"ADD",-"BC"
	copy(s[:], x[:32])
	runtime.KeepAlive(&s)
}

func moveDisjointStack64() {
	var s [96]byte
	// ppc64x:-".*memmove"
	// ppc64x/power8:"LXVD2X","ADD","BC"
	// ppc64x/power9:"LXV",-"LXVD2X",-"ADD",-"BC"
	copy(s[:], x[:96])
	runtime.KeepAlive(&s)
}

func moveDisjointStack() {
	var s [256]byte
	// s390x:-".*memmove"
	// amd64:-".*memmove"
	// ppc64x:-".*memmove"
	// ppc64x/power8:"LXVD2X"
	// ppc64x/power9:"LXV",-"LXVD2X"
	copy(s[:], x[:])
	runtime.KeepAlive(&s)
}

func moveDisjointArg(b *[256]byte) {
	var s [256]byte
	// s390x:-".*memmove"
	// amd64:-".*memmove"
	// ppc64x:-".*memmove"
	// ppc64x/power8:"LXVD2X"
	// ppc64x/power9:"LXV",-"LXVD2X"
	copy(s[:], b[:])
	runtime.KeepAlive(&s)
}

func moveDisjointNoOverlap(a *[256]byte) {
	// s390x:-".*memmove"
	// amd64:-".*memmove"
	// ppc64x:-".*memmove"
	// ppc64x/power8:"LXVD2X"
	// ppc64x/power9:"LXV",-"LXVD2X"
	copy(a[:], a[128:])
}

// Check arch-specific memmove lowering. See issue 41662 fot details

func moveArchLowering1(b []byte, x *[1]byte) {
	_ = b[1]
	// amd64:-".*memmove"
	// arm64:-".*memmove"
	// loong64:-".*memmove"
	// ppc64x:-".*memmove"
	copy(b, x[:])
}

func moveArchLowering2(b []byte, x *[2]byte) {
	_ = b[2]
	// amd64:-".*memmove"
	// arm64:-".*memmove"
	// loong64:-".*memmove"
	// ppc64x:-".*memmove"
	copy(b, x[:])
}

func moveArchLowering4(b []byte, x *[4]byte) {
	_ = b[4]
	// amd64:-".*memmove"
	// arm64:-".*memmove"
	// loong64:-".*memmove"
	// ppc64x:-".*memmove"
	copy(b, x[:])
}

func moveArchLowering8(b []byte, x *[8]byte) {
	_ = b[8]
	// amd64:-".*memmove"
	// arm64:-".*memmove"
	// ppc64x:-".*memmove"
	copy(b, x[:])
}

func moveArchLowering16(b []byte, x *[16]byte) {
	_ = b[16]
	// amd64:-".*memmove"
	copy(b, x[:])
}

// Check that no branches are generated when the pointers are [not] equal.

func ptrEqual() {
	// amd64:-"JEQ",-"JNE"
	// ppc64x:-"BEQ",-"BNE"
	// s390x:-"BEQ",-"BNE"
	copy(x[:], x[:])
}

func ptrOneOffset() {
	// amd64:-"JEQ",-"JNE"
	// ppc64x:-"BEQ",-"BNE"
	// s390x:-"BEQ",-"BNE"
	copy(x[1:], x[:])
}

func ptrBothOffset() {
	// amd64:-"JEQ",-"JNE"
	// ppc64x:-"BEQ",-"BNE"
	// s390x:-"BEQ",-"BNE"
	copy(x[1:], x[2:])
}

// Verify #62698 on PPC64.
func noMaskOnCopy(a []int, s string, x int) int {
	// ppc64x:-"MOVD\t$-1", -"AND"
	return a[x&^copy([]byte{}, s)]
}
