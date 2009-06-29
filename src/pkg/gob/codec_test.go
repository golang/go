// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gob

import (
	"bytes";
	"gob";
	"io";
	"os";
	"testing";
)

// Guarantee encoding format by comparing some encodings to hand-written values
type EncodeT struct {
	x	uint64;
	b	[]byte;
}
var encodeT = []EncodeT {
	EncodeT{ 0x00,	[]byte{0x80} },
	EncodeT{ 0x0f,	[]byte{0x8f} },
	EncodeT{ 0xff,	[]byte{0x7f, 0x81} },
	EncodeT{ 0xffff,	[]byte{0x7f, 0x7f, 0x83} },
	EncodeT{ 0xffffff,	[]byte{0x7f, 0x7f, 0x7f, 0x87} },
	EncodeT{ 0xffffffff,	[]byte{0x7f, 0x7f, 0x7f, 0x7f, 0x8f} },
	EncodeT{ 0xffffffffff,	[]byte{0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x9f} },
	EncodeT{ 0xffffffffffff,	[]byte{0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0xbf} },
	EncodeT{ 0xffffffffffffff,	[]byte{0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0xff} },
	EncodeT{ 0xffffffffffffffff,	[]byte{0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x81} },
	EncodeT{ 0x1111,	[]byte{0x11, 0xa2} },
	EncodeT{ 0x1111111111111111,	[]byte{0x11, 0x22, 0x44, 0x08, 0x11, 0x22, 0x44, 0x08, 0x91} },
	EncodeT{ 0x8888888888888888,	[]byte{0x08, 0x11, 0x22, 0x44, 0x08, 0x11, 0x22, 0x44, 0x08, 0x81} },
	EncodeT{ 1<<63,	[]byte{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x81} },
}

// Test basic encode/decode routines for unsigned integers
func TestUintCodec(t *testing.T) {
	var b = new(io.ByteBuffer);
	for i, tt := range encodeT {
		b.Reset();
		err := EncodeUint(b, tt.x);
		if err != nil {
			t.Error("EncodeUint:", tt.x, err)
		}
		if !bytes.Equal(tt.b, b.Data()) {
			t.Errorf("EncodeUint: expected % x got % x", tt.b, b.Data())
		}
	}
	for u := uint64(0); ; u = (u+1) * 7 {
		b.Reset();
		err := EncodeUint(b, u);
		if err != nil {
			t.Error("EncodeUint:", u, err)
		}
		v, err := DecodeUint(b);
		if err != nil {
			t.Error("DecodeUint:", u, err)
		}
		if u != v {
			t.Errorf("Encode/Decode: sent %#x received %#x\n", u, v)
		}
		if u & (1<<63) != 0 {
			break
		}
	}
}

func verifyInt(i int64, t *testing.T) {
	var b = new(io.ByteBuffer);
	err := EncodeInt(b, i);
	if err != nil {
		t.Error("EncodeInt:", i, err)
	}
	j, err := DecodeInt(b);
	if err != nil {
		t.Error("DecodeInt:", i, err)
	}
	if i != j {
		t.Errorf("Encode/Decode: sent %#x received %#x\n", uint64(i), uint64(j))
	}
}

// Test basic encode/decode routines for signed integers
func TestIntCodec(t *testing.T) {
	var b = new(io.ByteBuffer);
	for u := uint64(0); ; u = (u+1) * 7 {
		// Do positive and negative values
		i := int64(u);
		verifyInt(i, t);
		verifyInt(-i, t);
		verifyInt(^i, t);
		if u & (1<<63) != 0 {
			break
		}
	}
	verifyInt(-1<<63, t);	// a tricky case
}
