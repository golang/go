// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"encoding/binary"
	"testing"
)

var gv = [16]byte{0, 1, 2, 3, 4, 5, 6, 7, 8}

//go:noinline
func readGlobalUnaligned() uint64 {
	return binary.LittleEndian.Uint64(gv[1:])
}

func TestUnalignedGlobal(t *testing.T) {
	// Note: this is a test not so much of the result of the read, but of
	// the correct compilation of that read. On s390x unaligned global
	// accesses fail to compile.
	if got, want := readGlobalUnaligned(), uint64(0x0807060504030201); got != want {
		t.Errorf("read global %x, want %x", got, want)
	}
}

func TestSpillOfExtendedEndianLoads(t *testing.T) {
	b := []byte{0xaa, 0xbb, 0xcc, 0xdd}

	var testCases = []struct {
		fn   func([]byte) uint64
		want uint64
	}{
		{readUint16le, 0xbbaa},
		{readUint16be, 0xaabb},
		{readUint32le, 0xddccbbaa},
		{readUint32be, 0xaabbccdd},
	}
	for _, test := range testCases {
		if got := test.fn(b); got != test.want {
			t.Errorf("got %x, want %x", got, test.want)
		}
	}
}

func readUint16le(b []byte) uint64 {
	y := uint64(binary.LittleEndian.Uint16(b))
	nop() // force spill
	return y
}

func readUint16be(b []byte) uint64 {
	y := uint64(binary.BigEndian.Uint16(b))
	nop() // force spill
	return y
}

func readUint32le(b []byte) uint64 {
	y := uint64(binary.LittleEndian.Uint32(b))
	nop() // force spill
	return y
}

func readUint32be(b []byte) uint64 {
	y := uint64(binary.BigEndian.Uint32(b))
	nop() // force spill
	return y
}

//go:noinline
func nop() {
}
