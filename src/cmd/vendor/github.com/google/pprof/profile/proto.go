// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This file is a simple protocol buffer encoder and decoder.
// The format is described at
// https://developers.google.com/protocol-buffers/docs/encoding
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

import "errors"

type buffer struct {
	field int // field tag
	typ   int // proto wire type code for field
	u64   uint64
	data  []byte
	tmp   [16]byte
}

type decoder func(*buffer, message) error

type message interface {
	decoder() []decoder
	encode(*buffer)
}

func marshal(m message) []byte {
	var b buffer
	m.encode(&b)
	return b.data
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
	encodeVarint(b, uint64(tag)<<3)
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

func encodeInt64Opt(b *buffer, tag int, x int64) {
	if x == 0 {
		return
	}
	encodeInt64(b, tag, x)
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

func encodeBool(b *buffer, tag int, x bool) {
	if x {
		encodeUint64(b, tag, 1)
	} else {
		encodeUint64(b, tag, 0)
	}
}

func encodeBoolOpt(b *buffer, tag int, x bool) {
	if x {
		encodeBool(b, tag, x)
	}
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

func unmarshal(data []byte, m message) (err error) {
	b := buffer{data: data, typ: 2}
	return decodeMessage(&b, m)
}

func le64(p []byte) uint64 {
	return uint64(p[0]) | uint64(p[1])<<8 | uint64(p[2])<<16 | uint64(p[3])<<24 | uint64(p[4])<<32 | uint64(p[5])<<40 | uint64(p[6])<<48 | uint64(p[7])<<56
}

func le32(p []byte) uint32 {
	return uint32(p[0]) | uint32(p[1])<<8 | uint32(p[2])<<16 | uint32(p[3])<<24
}

func decodeVarint(data []byte) (uint64, []byte, error) {
	var u uint64
	for i := 0; ; i++ {
		if i >= 10 || i >= len(data) {
			return 0, nil, errors.New("bad varint")
		}
		u |= uint64(data[i]&0x7F) << uint(7*i)
		if data[i]&0x80 == 0 {
			return u, data[i+1:], nil
		}
	}
}

func decodeField(b *buffer, data []byte) ([]byte, error) {
	x, data, err := decodeVarint(data)
	if err != nil {
		return nil, err
	}
	b.field = int(x >> 3)
	b.typ = int(x & 7)
	b.data = nil
	b.u64 = 0
	switch b.typ {
	case 0:
		b.u64, data, err = decodeVarint(data)
		if err != nil {
			return nil, err
		}
	case 1:
		if len(data) < 8 {
			return nil, errors.New("not enough data")
		}
		b.u64 = le64(data[:8])
		data = data[8:]
	case 2:
		var n uint64
		n, data, err = decodeVarint(data)
		if err != nil {
			return nil, err
		}
		if n > uint64(len(data)) {
			return nil, errors.New("too much data")
		}
		b.data = data[:n]
		data = data[n:]
	case 5:
		if len(data) < 4 {
			return nil, errors.New("not enough data")
		}
		b.u64 = uint64(le32(data[:4]))
		data = data[4:]
	default:
		return nil, errors.New("unknown wire type: " + string(b.typ))
	}

	return data, nil
}

func checkType(b *buffer, typ int) error {
	if b.typ != typ {
		return errors.New("type mismatch")
	}
	return nil
}

func decodeMessage(b *buffer, m message) error {
	if err := checkType(b, 2); err != nil {
		return err
	}
	dec := m.decoder()
	data := b.data
	for len(data) > 0 {
		// pull varint field# + type
		var err error
		data, err = decodeField(b, data)
		if err != nil {
			return err
		}
		if b.field >= len(dec) || dec[b.field] == nil {
			continue
		}
		if err := dec[b.field](b, m); err != nil {
			return err
		}
	}
	return nil
}

func decodeInt64(b *buffer, x *int64) error {
	if err := checkType(b, 0); err != nil {
		return err
	}
	*x = int64(b.u64)
	return nil
}

func decodeInt64s(b *buffer, x *[]int64) error {
	if b.typ == 2 {
		// Packed encoding
		data := b.data
		tmp := make([]int64, 0, len(data)) // Maximally sized
		for len(data) > 0 {
			var u uint64
			var err error

			if u, data, err = decodeVarint(data); err != nil {
				return err
			}
			tmp = append(tmp, int64(u))
		}
		*x = append(*x, tmp...)
		return nil
	}
	var i int64
	if err := decodeInt64(b, &i); err != nil {
		return err
	}
	*x = append(*x, i)
	return nil
}

func decodeUint64(b *buffer, x *uint64) error {
	if err := checkType(b, 0); err != nil {
		return err
	}
	*x = b.u64
	return nil
}

func decodeUint64s(b *buffer, x *[]uint64) error {
	if b.typ == 2 {
		data := b.data
		// Packed encoding
		tmp := make([]uint64, 0, len(data)) // Maximally sized
		for len(data) > 0 {
			var u uint64
			var err error

			if u, data, err = decodeVarint(data); err != nil {
				return err
			}
			tmp = append(tmp, u)
		}
		*x = append(*x, tmp...)
		return nil
	}
	var u uint64
	if err := decodeUint64(b, &u); err != nil {
		return err
	}
	*x = append(*x, u)
	return nil
}

func decodeString(b *buffer, x *string) error {
	if err := checkType(b, 2); err != nil {
		return err
	}
	*x = string(b.data)
	return nil
}

func decodeStrings(b *buffer, x *[]string) error {
	var s string
	if err := decodeString(b, &s); err != nil {
		return err
	}
	*x = append(*x, s)
	return nil
}

func decodeBool(b *buffer, x *bool) error {
	if err := checkType(b, 0); err != nil {
		return err
	}
	if int64(b.u64) == 0 {
		*x = false
	} else {
		*x = true
	}
	return nil
}
