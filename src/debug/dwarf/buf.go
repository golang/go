// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Buffered reading and decoding of DWARF data streams.

package dwarf

import (
	"bytes"
	"encoding/binary"
	"strconv"
)

// Data buffer being decoded.
type buf struct {
	dwarf  *Data
	order  binary.ByteOrder
	format dataFormat
	name   string
	off    Offset
	data   []byte
	err    error
}

// Data format, other than byte order. This affects the handling of
// certain field formats.
type dataFormat interface {
	// DWARF version number. Zero means unknown.
	version() int

	// 64-bit DWARF format?
	dwarf64() (dwarf64 bool, isKnown bool)

	// Size of an address, in bytes. Zero means unknown.
	addrsize() int
}

// Some parts of DWARF have no data format, e.g., abbrevs.
type unknownFormat struct{}

func (u unknownFormat) version() int {
	return 0
}

func (u unknownFormat) dwarf64() (bool, bool) {
	return false, false
}

func (u unknownFormat) addrsize() int {
	return 0
}

func makeBuf(d *Data, format dataFormat, name string, off Offset, data []byte) buf {
	return buf{d, d.order, format, name, off, data, nil}
}

func (b *buf) uint8() uint8 {
	if len(b.data) < 1 {
		b.error("underflow")
		return 0
	}
	val := b.data[0]
	b.data = b.data[1:]
	b.off++
	return val
}

func (b *buf) bytes(n int) []byte {
	if n < 0 || len(b.data) < n {
		b.error("underflow")
		return nil
	}
	data := b.data[0:n]
	b.data = b.data[n:]
	b.off += Offset(n)
	return data
}

func (b *buf) skip(n int) { b.bytes(n) }

func (b *buf) string() string {
	i := bytes.IndexByte(b.data, 0)
	if i < 0 {
		b.error("underflow")
		return ""
	}

	s := string(b.data[0:i])
	b.data = b.data[i+1:]
	b.off += Offset(i + 1)
	return s
}

func (b *buf) uint16() uint16 {
	a := b.bytes(2)
	if a == nil {
		return 0
	}
	return b.order.Uint16(a)
}

func (b *buf) uint24() uint32 {
	a := b.bytes(3)
	if a == nil {
		return 0
	}
	if b.dwarf.bigEndian {
		return uint32(a[2]) | uint32(a[1])<<8 | uint32(a[0])<<16
	} else {
		return uint32(a[0]) | uint32(a[1])<<8 | uint32(a[2])<<16
	}
}

func (b *buf) uint32() uint32 {
	a := b.bytes(4)
	if a == nil {
		return 0
	}
	return b.order.Uint32(a)
}

func (b *buf) uint64() uint64 {
	a := b.bytes(8)
	if a == nil {
		return 0
	}
	return b.order.Uint64(a)
}

// Read a varint, which is 7 bits per byte, little endian.
// the 0x80 bit means read another byte.
func (b *buf) varint() (c uint64, bits uint) {
	for i := 0; i < len(b.data); i++ {
		byte := b.data[i]
		c |= uint64(byte&0x7F) << bits
		bits += 7
		if byte&0x80 == 0 {
			b.off += Offset(i + 1)
			b.data = b.data[i+1:]
			return c, bits
		}
	}
	return 0, 0
}

// Unsigned int is just a varint.
func (b *buf) uint() uint64 {
	x, _ := b.varint()
	return x
}

// Signed int is a sign-extended varint.
func (b *buf) int() int64 {
	ux, bits := b.varint()
	x := int64(ux)
	if x&(1<<(bits-1)) != 0 {
		x |= -1 << bits
	}
	return x
}

// Address-sized uint.
func (b *buf) addr() uint64 {
	switch b.format.addrsize() {
	case 1:
		return uint64(b.uint8())
	case 2:
		return uint64(b.uint16())
	case 4:
		return uint64(b.uint32())
	case 8:
		return b.uint64()
	}
	b.error("unknown address size")
	return 0
}

func (b *buf) unitLength() (length Offset, dwarf64 bool) {
	length = Offset(b.uint32())
	if length == 0xffffffff {
		dwarf64 = true
		length = Offset(b.uint64())
	} else if length >= 0xfffffff0 {
		b.error("unit length has reserved value")
	}
	return
}

func (b *buf) error(s string) {
	if b.err == nil {
		b.data = nil
		b.err = DecodeError{b.name, b.off, s}
	}
}

type DecodeError struct {
	Name   string
	Offset Offset
	Err    string
}

func (e DecodeError) Error() string {
	return "decoding dwarf section " + e.Name + " at offset 0x" + strconv.FormatInt(int64(e.Offset), 16) + ": " + e.Err
}
