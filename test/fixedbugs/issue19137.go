// compile

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 19137: folding address into load/store causes
// odd offset on ARM64.

package p

type T struct {
	p *int
	a [2]byte
	b [6]byte // not 4-byte aligned
}

func f(b [6]byte) T {
	var x [1000]int // a large stack frame
	_ = x
	return T{b: b}
}

// Arg symbol's base address may be not at an aligned offset to
// SP. Folding arg's address into load/store may cause odd offset.
func move(a, b [20]byte) [20]byte {
	var x [1000]int // a large stack frame
	_ = x
	return b // b is not 8-byte aligned to SP
}
func zero() ([20]byte, [20]byte) {
	var x [1000]int // a large stack frame
	_ = x
	return [20]byte{}, [20]byte{} // the second return value is not 8-byte aligned to SP
}

// Issue 21992: unaligned offset between 256 and 504 and handled
// incorrectly.
type T2 struct {
	a [257]byte
	// fields below are not 8-, 4-, 2-byte aligned
	b [8]byte
	c [4]byte
	d [2]byte
}

func f2(x *T2) {
	x.b = [8]byte{}
	x.c = [4]byte{}
	x.d = [2]byte{}
}
