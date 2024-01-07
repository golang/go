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

type T32 struct {
	a, b uint32
}

//go:noinline
func (t *T32) bigEndianLoad() uint64 {
	return uint64(t.a)<<32 | uint64(t.b)
}

//go:noinline
func (t *T32) littleEndianLoad() uint64 {
	return uint64(t.a) | (uint64(t.b) << 32)
}

//go:noinline
func (t *T32) bigEndianStore(x uint64) {
	t.a = uint32(x >> 32)
	t.b = uint32(x)
}

//go:noinline
func (t *T32) littleEndianStore(x uint64) {
	t.a = uint32(x)
	t.b = uint32(x >> 32)
}

type T16 struct {
	a, b uint16
}

//go:noinline
func (t *T16) bigEndianLoad() uint32 {
	return uint32(t.a)<<16 | uint32(t.b)
}

//go:noinline
func (t *T16) littleEndianLoad() uint32 {
	return uint32(t.a) | (uint32(t.b) << 16)
}

//go:noinline
func (t *T16) bigEndianStore(x uint32) {
	t.a = uint16(x >> 16)
	t.b = uint16(x)
}

//go:noinline
func (t *T16) littleEndianStore(x uint32) {
	t.a = uint16(x)
	t.b = uint16(x >> 16)
}

type T8 struct {
	a, b uint8
}

//go:noinline
func (t *T8) bigEndianLoad() uint16 {
	return uint16(t.a)<<8 | uint16(t.b)
}

//go:noinline
func (t *T8) littleEndianLoad() uint16 {
	return uint16(t.a) | (uint16(t.b) << 8)
}

//go:noinline
func (t *T8) bigEndianStore(x uint16) {
	t.a = uint8(x >> 8)
	t.b = uint8(x)
}

//go:noinline
func (t *T8) littleEndianStore(x uint16) {
	t.a = uint8(x)
	t.b = uint8(x >> 8)
}

func TestIssue64468(t *testing.T) {
	t32 := T32{1, 2}
	if got, want := t32.bigEndianLoad(), uint64(1<<32+2); got != want {
		t.Errorf("T32.bigEndianLoad got %x want %x\n", got, want)
	}
	if got, want := t32.littleEndianLoad(), uint64(1+2<<32); got != want {
		t.Errorf("T32.littleEndianLoad got %x want %x\n", got, want)
	}
	t16 := T16{1, 2}
	if got, want := t16.bigEndianLoad(), uint32(1<<16+2); got != want {
		t.Errorf("T16.bigEndianLoad got %x want %x\n", got, want)
	}
	if got, want := t16.littleEndianLoad(), uint32(1+2<<16); got != want {
		t.Errorf("T16.littleEndianLoad got %x want %x\n", got, want)
	}
	t8 := T8{1, 2}
	if got, want := t8.bigEndianLoad(), uint16(1<<8+2); got != want {
		t.Errorf("T8.bigEndianLoad got %x want %x\n", got, want)
	}
	if got, want := t8.littleEndianLoad(), uint16(1+2<<8); got != want {
		t.Errorf("T8.littleEndianLoad got %x want %x\n", got, want)
	}
	t32.bigEndianStore(1<<32 + 2)
	if got, want := t32, (T32{1, 2}); got != want {
		t.Errorf("T32.bigEndianStore got %x want %x\n", got, want)
	}
	t32.littleEndianStore(1<<32 + 2)
	if got, want := t32, (T32{2, 1}); got != want {
		t.Errorf("T32.littleEndianStore got %x want %x\n", got, want)
	}
	t16.bigEndianStore(1<<16 + 2)
	if got, want := t16, (T16{1, 2}); got != want {
		t.Errorf("T16.bigEndianStore got %x want %x\n", got, want)
	}
	t16.littleEndianStore(1<<16 + 2)
	if got, want := t16, (T16{2, 1}); got != want {
		t.Errorf("T16.littleEndianStore got %x want %x\n", got, want)
	}
	t8.bigEndianStore(1<<8 + 2)
	if got, want := t8, (T8{1, 2}); got != want {
		t.Errorf("T8.bigEndianStore got %x want %x\n", got, want)
	}
	t8.littleEndianStore(1<<8 + 2)
	if got, want := t8, (T8{2, 1}); got != want {
		t.Errorf("T8.littleEndianStore got %x want %x\n", got, want)
	}
}
