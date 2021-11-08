// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "unsafe"

const (
	smallInt = 42

	// For bigInt, we use a value that's too big for an int64, but still
	// fits in uint64. go/constant uses a different representation for
	// values larger than int64, but the cmd/asm parser can't parse
	// anything bigger than a uint64.
	bigInt = 0xffffffffffffffff

	stringVal = "test"
)

var (
	smallIntAsm int64
	bigIntAsm   uint64
	stringAsm   [len(stringVal)]byte
)

type typ struct {
	a uint64
	b [100]uint8
	c uint8
}

var (
	typSize uint64

	typA, typB, typC uint64
)

func main() {
	if smallInt != smallIntAsm {
		println("smallInt", smallInt, "!=", smallIntAsm)
	}
	if bigInt != bigIntAsm {
		println("bigInt", uint64(bigInt), "!=", bigIntAsm)
	}
	if stringVal != string(stringAsm[:]) {
		println("stringVal", stringVal, "!=", string(stringAsm[:]))
	}

	// We also include boolean consts in go_asm.h, but they're
	// defined to be "true" or "false", and it's not clear how to
	// use that in assembly.

	if want := unsafe.Sizeof(typ{}); want != uintptr(typSize) {
		println("typSize", want, "!=", typSize)
	}
	if want := unsafe.Offsetof(typ{}.a); want != uintptr(typA) {
		println("typA", want, "!=", typA)
	}
	if want := unsafe.Offsetof(typ{}.b); want != uintptr(typB) {
		println("typB", want, "!=", typB)
	}
	if want := unsafe.Offsetof(typ{}.c); want != uintptr(typC) {
		println("typC", want, "!=", typC)
	}
}
