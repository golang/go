// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// This file is a simple protocol buffer encoder and decoder.
//
// A protocol message must implement the message interface:
//   decoder() []decoder
//   encode(*buffer)
//
// The decode method returns a slice indexed by field number that gives the
// function to decode that field.
// The encode method encodes its receiver into the given buffer.
//
// The two methods are simple enough to be implemented by hand rather than
// by using a protocol compiler.
//
// See profile.go for examples of messages implementing this interface.
//
// There is no support for groups, message sets, or "has" bits.

package profile

type buffer struct {
	field int
	typ   int
	u64   uint64
	data  []byte
	tmp   [16]byte
}

type message interface {
	encode(*buffer)
}

func encodeVarint(b *buffer, x uint64) {
	for x >= 128 {
		b.data = append(b.data, byte(x)|0x80)
		x >>= 7
	}
	b.data = append(b.data, byte(x))
}

func encodeLength(b *buffer, tag int, len int) {
	encodeVarint(b, uint64(tag)<<3|2)
	encodeVarint(b, uint64(len))
}

func encodeUint64(b *buffer, tag int, x uint64) {
	// append varint to b.data
	encodeVarint(b, uint64(tag)<<3|0)
	encodeVarint(b, x)
}

func encodeUint64s(b *buffer, tag int, x []uint64) {
	if len(x) > 2 {
		// Use packed encoding
		n1 := len(b.data)
		for _, u := range x {
			encodeVarint(b, u)
		}
		n2 := len(b.data)
		encodeLength(b, tag, n2-n1)
		n3 := len(b.data)
		copy(b.tmp[:], b.data[n2:n3])
		copy(b.data[n1+(n3-n2):], b.data[n1:n2])
		copy(b.data[n1:], b.tmp[:n3-n2])
		return
	}
	for _, u := range x {
		encodeUint64(b, tag, u)
	}
}

func encodeUint64Opt(b *buffer, tag int, x uint64) {
	if x == 0 {
		return
	}
	encodeUint64(b, tag, x)
}

func encodeInt64(b *buffer, tag int, x int64) {
	u := uint64(x)
	encodeUint64(b, tag, u)
}

func encodeInt64Opt(b *buffer, tag int, x int64) {
	if x == 0 {
		return
	}
	encodeInt64(b, tag, x)
}

func encodeInt64s(b *buffer, tag int, x []int64) {
	if len(x) > 2 {
		// Use packed encoding
		n1 := len(b.data)
		for _, u := range x {
			encodeVarint(b, uint64(u))
		}
		n2 := len(b.data)
		encodeLength(b, tag, n2-n1)
		n3 := len(b.data)
		copy(b.tmp[:], b.data[n2:n3])
		copy(b.data[n1+(n3-n2):], b.data[n1:n2])
		copy(b.data[n1:], b.tmp[:n3-n2])
		return
	}
	for _, u := range x {
		encodeInt64(b, tag, u)
	}
}

func encodeString(b *buffer, tag int, x string) {
	encodeLength(b, tag, len(x))
	b.data = append(b.data, x...)
}

func encodeStrings(b *buffer, tag int, x []string) {
	for _, s := range x {
		encodeString(b, tag, s)
	}
}

func encodeStringOpt(b *buffer, tag int, x string) {
	if x == "" {
		return
	}
	encodeString(b, tag, x)
}

func encodeBool(b *buffer, tag int, x bool) {
	if x {
		encodeUint64(b, tag, 1)
	} else {
		encodeUint64(b, tag, 0)
	}
}

func encodeBoolOpt(b *buffer, tag int, x bool) {
	if x == false {
		return
	}
	encodeBool(b, tag, x)
}

func encodeMessage(b *buffer, tag int, m message) {
	n1 := len(b.data)
	m.encode(b)
	n2 := len(b.data)
	encodeLength(b, tag, n2-n1)
	n3 := len(b.data)
	copy(b.tmp[:], b.data[n2:n3])
	copy(b.data[n1+(n3-n2):], b.data[n1:n2])
	copy(b.data[n1:], b.tmp[:n3-n2])
}
