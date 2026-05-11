// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.


package main

import (
	"testing"
)

type AlignedBuffer8 struct {
	data [8]byte
}

type AlignedBuffer4 struct {
	data [4]byte
}

type AlignedBuffer2 struct {
	data [2]byte
}

var (
	globalBuffer8 = AlignedBuffer8{data: [8]byte{0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef}}
	globalBuffer4 = AlignedBuffer4{data: [4]byte{0x01, 0x23, 0x45, 0x67}}
	globalBuffer2 = AlignedBuffer2{data: [2]byte{0x01, 0x23}}
)

//go:noinline
func readUint64Global() uint64 {
	buf := globalBuffer8.data[:]
	_ = buf[7]
	return uint64(buf[0]) | uint64(buf[1])<<8 | uint64(buf[2])<<16 | uint64(buf[3])<<24 |
		uint64(buf[4])<<32 | uint64(buf[5])<<40 | uint64(buf[6])<<48 | uint64(buf[7])<<56
}

//go:noinline
func readUint64Local() uint64 {
	local := AlignedBuffer8{data: [8]byte{0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef}}
	buf := local.data[:]
	_ = buf[7]
	return uint64(buf[0]) | uint64(buf[1])<<8 | uint64(buf[2])<<16 | uint64(buf[3])<<24 |
		uint64(buf[4])<<32 | uint64(buf[5])<<40 | uint64(buf[6])<<48 | uint64(buf[7])<<56
}

//go:noinline
func readUint32Global() uint32 {
	buf := globalBuffer4.data[:]
	_ = buf[3]
	return uint32(buf[0]) | uint32(buf[1])<<8 | uint32(buf[2])<<16 | uint32(buf[3])<<24
}

//go:noinline
func readUint32Local() uint32 {
	local := AlignedBuffer4{data: [4]byte{0x01, 0x23, 0x45, 0x67}}
	buf := local.data[:]
	_ = buf[3]
	return uint32(buf[0]) | uint32(buf[1])<<8 | uint32(buf[2])<<16 | uint32(buf[3])<<24
}

//go:noinline
func readUint16Global() uint16 {
	buf := globalBuffer2.data[:]
	_ = buf[1]
	return uint16(buf[0]) | uint16(buf[1])<<8
}

//go:noinline
func readUint16Local() uint16 {
	local := AlignedBuffer2{data: [2]byte{0x01, 0x23}}
	buf := local.data[:]
	_ = buf[1]
	return uint16(buf[0]) | uint16(buf[1])<<8
}

//go:noinline
func readUint64Dynamic(data []byte, offset int) uint64 {
	buf := data[offset:]
	_ = buf[7]
	return uint64(buf[0]) | uint64(buf[1])<<8 | uint64(buf[2])<<16 | uint64(buf[3])<<24 |
		uint64(buf[4])<<32 | uint64(buf[5])<<40 | uint64(buf[6])<<48 | uint64(buf[7])<<56
}

//go:noinline
func readUint64Slice(buf []byte) uint64 {
	_ = buf[7]
	return uint64(buf[0]) | uint64(buf[1])<<8 | uint64(buf[2])<<16 | uint64(buf[3])<<24 |
		uint64(buf[4])<<32 | uint64(buf[5])<<40 | uint64(buf[6])<<48 | uint64(buf[7])<<56
}

//go:noinline
func writeUint64Global(v uint64) {
	buf := globalBuffer8.data[:]
	_ = buf[7]
	buf[0] = byte(v)
	buf[1] = byte(v >> 8)
	buf[2] = byte(v >> 16)
	buf[3] = byte(v >> 24)
	buf[4] = byte(v >> 32)
	buf[5] = byte(v >> 40)
	buf[6] = byte(v >> 48)
	buf[7] = byte(v >> 56)
}

//go:noinline
func writeUint32Global(v uint32) {
	buf := globalBuffer4.data[:]
	_ = buf[3]
	buf[0] = byte(v)
	buf[1] = byte(v >> 8)
	buf[2] = byte(v >> 16)
	buf[3] = byte(v >> 24)
}

//go:noinline
func writeUint16Global(v uint16) {
	buf := globalBuffer2.data[:]
	_ = buf[1]
	buf[0] = byte(v)
	buf[1] = byte(v >> 8)
}

func BenchmarkReadUint64Global(b *testing.B) {
	var result uint64
	for i := 0; i < b.N; i++ {
		result = readUint64Global()
	}
	_ = result
}

func BenchmarkReadUint64Local(b *testing.B) {
	var result uint64
	for i := 0; i < b.N; i++ {
		result = readUint64Local()
	}
	_ = result
}

func BenchmarkReadUint32Global(b *testing.B) {
	var result uint32
	for i := 0; i < b.N; i++ {
		result = readUint32Global()
	}
	_ = result
}

func BenchmarkReadUint32Local(b *testing.B) {
	var result uint32
	for i := 0; i < b.N; i++ {
		result = readUint32Local()
	}
	_ = result
}

func BenchmarkReadUint16Global(b *testing.B) {
	var result uint16
	for i := 0; i < b.N; i++ {
		result = readUint16Global()
	}
	_ = result
}

func BenchmarkReadUint16Local(b *testing.B) {
	var result uint16
	for i := 0; i < b.N; i++ {
		result = readUint16Local()
	}
	_ = result
}

func BenchmarkReadUint64Dynamic(b *testing.B) {
	data := []byte{0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0xff}
	var result uint64
	for i := 0; i < b.N; i++ {
		result = readUint64Dynamic(data, 0)
	}
	_ = result
}

func BenchmarkReadUint64Slice(b *testing.B) {
	data := []byte{0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef}
	var result uint64
	for i := 0; i < b.N; i++ {
		result = readUint64Slice(data)
	}
	_ = result
}

func BenchmarkWriteUint64Global(b *testing.B) {
	for i := 0; i < b.N; i++ {
		writeUint64Global(0xefcdab8967452301)
	}
}

func BenchmarkWriteUint32Global(b *testing.B) {
	for i := 0; i < b.N; i++ {
		writeUint32Global(0x67452301)
	}
}

func BenchmarkWriteUint16Global(b *testing.B) {
	for i := 0; i < b.N; i++ {
		writeUint16Global(0x2301)
	}
}

func TestMemcombineCorrectness(t *testing.T) {

	const (
		want64 = uint64(0xefcdab8967452301)
		want32 = uint32(0x67452301)
		want16 = uint16(0x2301)
	)

	if got := readUint64Global(); got != want64 {
		t.Errorf("readUint64Global() = %#x, want %#x", got, want64)
	}
	if got := readUint64Local(); got != want64 {
		t.Errorf("readUint64Local() = %#x, want %#x", got, want64)
	}
	if got := readUint32Global(); got != want32 {
		t.Errorf("readUint32Global() = %#x, want %#x", got, want32)
	}
	if got := readUint32Local(); got != want32 {
		t.Errorf("readUint32Local() = %#x, want %#x", got, want32)
	}
	if got := readUint16Global(); got != want16 {
		t.Errorf("readUint16Global() = %#x, want %#x", got, want16)
	}
	if got := readUint16Local(); got != want16 {
		t.Errorf("readUint16Local() = %#x, want %#x", got, want16)
	}
	slicedata := []byte{0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef}
	if got := readUint64Slice(slicedata); got != want64 {
		t.Errorf("readUint64Slice() = %#x, want %#x", got, want64)
	}

	// Dynamic offset test (should not be optimized, but still correct)
	dynamicdata := []byte{0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0xff}
	if got := readUint64Dynamic(dynamicdata, 0); got != want64 {
		t.Errorf("readUint64Dynamic(offset=0) = %#x, want %#x", got, want64)
	}
	if got := readUint64Dynamic(dynamicdata, 1); got != 0xefcdab89674523<<8|0x01 {
		expected := uint64(dynamicdata[1]) | uint64(dynamicdata[2])<<8 | uint64(dynamicdata[3])<<16 |
			uint64(dynamicdata[4])<<24 | uint64(dynamicdata[5])<<32 | uint64(dynamicdata[6])<<40 |
			uint64(dynamicdata[7])<<48 | uint64(dynamicdata[8])<<56
		if got != expected {
			t.Errorf("readUint64Dynamic(offset=1) = %#x, want %#x", got, expected)
		}
	}

	writeUint64Global(0xdeadbeefcafebabe)
	if got := readUint64Global(); got != 0xdeadbeefcafebabe {
		t.Errorf("after writeUint64Global, read = %#x, want 0xdeadbeefcafebabe", got)
	}

	writeUint64Global(want64)

	writeUint32Global(0x12345678)
	if got := readUint32Global(); got != 0x12345678 {
		t.Errorf("after writeUint32Global, read = %#x, want 0x12345678", got)
	}
	writeUint32Global(want32)

	writeUint16Global(0xabcd)
	if got := readUint16Global(); got != 0xabcd {
		t.Errorf("after writeUint16Global, read = %#x, want 0xabcd", got)
	}
	writeUint16Global(want16)
}

func main() {
	println("Uint64 Global:", readUint64Global())
	println("Uint32 Global:", readUint32Global())
	println("Uint16 Global:", readUint16Global())
	println("Uint64 Local:", readUint64Local())
	println("Uint64 Dynamic:", readUint64Dynamic([]byte{0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef}, 0))
}
