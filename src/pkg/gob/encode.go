// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gob

import (
	"io";
	"math";
	"os";
	"unsafe";
)

// Integers encode as a variant of Google's protocol buffer varint (varvarint?).
// The variant is that the continuation bytes have a zero top bit instead of a one.
// That way there's only one bit to clear and the value is a little easier to see if
// you're the unfortunate sort of person who must read the hex to debug.

// EncodeUint writes an encoded unsigned integer to w.
func EncodeUint(w io.Writer, x uint64) os.Error {
	var buf [16]byte;
	var n int;
	for n = 0; x > 127; n++ {
		buf[n] = uint8(x & 0x7F);
		x >>= 7;
	}
	buf[n] = 0x80 | uint8(x);
	nn, err := w.Write(buf[0:n+1]);
	return err;
}

// EncodeInt writes an encoded signed integer to w.
// The low bit of the encoding says whether to bit complement the (other bits of the) uint to recover the int.
func EncodeInt(w io.Writer, i int64) os.Error {
	var x uint64;
	if i < 0 {
		x = uint64(^i << 1) | 1
	} else {
		x = uint64(i << 1)
	}
	return EncodeUint(w, uint64(x))
}

// The global execution state of an instance of the encoder.
type encState struct {
	w	io.Writer;
	base	uintptr;
}

// The 'instructions' of the encoding machine
type encInstr struct {
	op	func(i *encInstr, state *encState);
	field		int;	// field number
	indir	int;	// how many pointer indirections to reach the value in the struct
	offset	uintptr;	// offset in the structure of the field to encode
}

func encBool(i *encInstr, state *encState) {
	p := unsafe.Pointer(state.base+i.offset);
	for indir := i.indir; indir > 0; indir-- {
		p = *(*unsafe.Pointer)(p);
		if p == nil {
			return
		}
	}
	b := *(*bool)(p);
	if b {
		EncodeUint(state.w, uint64(i.field));
		EncodeUint(state.w, 1);
	}
}

func encInt(i *encInstr, state *encState) {
	p := unsafe.Pointer(state.base+i.offset);
	for indir := i.indir; indir > 0; indir-- {
		p = *(*unsafe.Pointer)(p);
		if p == nil {
			return
		}
	}
	v := int64(*(*int)(p));
	if v != 0 {
		EncodeUint(state.w, uint64(i.field));
		EncodeInt(state.w, v);
	}
}

func encUint(i *encInstr, state *encState) {
	p := unsafe.Pointer(state.base+i.offset);
	for indir := i.indir; indir > 0; indir-- {
		p = *(*unsafe.Pointer)(p);
		if p == nil {
			return
		}
	}
	v := uint64(*(*uint)(p));
	if v != 0 {
		EncodeUint(state.w, uint64(i.field));
		EncodeUint(state.w, v);
	}
}

func encInt8(i *encInstr, state *encState) {
	p := unsafe.Pointer(state.base+i.offset);
	for indir := i.indir; indir > 0; indir-- {
		p = *(*unsafe.Pointer)(p);
		if p == nil {
			return
		}
	}
	v := int64(*(*int8)(p));
	if v != 0 {
		EncodeUint(state.w, uint64(i.field));
		EncodeInt(state.w, v);
	}
}

func encUint8(i *encInstr, state *encState) {
	p := unsafe.Pointer(state.base+i.offset);
	for indir := i.indir; indir > 0; indir-- {
		p = *(*unsafe.Pointer)(p);
		if p == nil {
			return
		}
	}
	v := uint64(*(*uint8)(p));
	if v != 0 {
		EncodeUint(state.w, uint64(i.field));
		EncodeUint(state.w, v);
	}
}

func encInt16(i *encInstr, state *encState) {
	p := unsafe.Pointer(state.base+i.offset);
	for indir := i.indir; indir > 0; indir-- {
		p = *(*unsafe.Pointer)(p);
		if p == nil {
			return
		}
	}
	v := int64(*(*int16)(p));
	if v != 0 {
		EncodeUint(state.w, uint64(i.field));
		EncodeInt(state.w, v);
	}
}

func encUint16(i *encInstr, state *encState) {
	p := unsafe.Pointer(state.base+i.offset);
	for indir := i.indir; indir > 0; indir-- {
		p = *(*unsafe.Pointer)(p);
		if p == nil {
			return
		}
	}
	v := uint64(*(*uint16)(p));
	if v != 0 {
		EncodeUint(state.w, uint64(i.field));
		EncodeUint(state.w, v);
	}
}

func encInt32(i *encInstr, state *encState) {
	p := unsafe.Pointer(state.base+i.offset);
	for indir := i.indir; indir > 0; indir-- {
		p = *(*unsafe.Pointer)(p);
		if p == nil {
			return
		}
	}
	v := int64(*(*int32)(p));
	if v != 0 {
		EncodeUint(state.w, uint64(i.field));
		EncodeInt(state.w, v);
	}
}

func encUint32(i *encInstr, state *encState) {
	p := unsafe.Pointer(state.base+i.offset);
	for indir := i.indir; indir > 0; indir-- {
		p = *(*unsafe.Pointer)(p);
		if p == nil {
			return
		}
	}
	v := uint64(*(*uint32)(p));
	if v != 0 {
		EncodeUint(state.w, uint64(i.field));
		EncodeUint(state.w, v);
	}
}

func encInt64(i *encInstr, state *encState) {
	p := unsafe.Pointer(state.base+i.offset);
	for indir := i.indir; indir > 0; indir-- {
		p = *(*unsafe.Pointer)(p);
		if p == nil {
			return
		}
	}
	v := *(*int64)(p);
	if v != 0 {
		EncodeUint(state.w, uint64(i.field));
		EncodeInt(state.w, v);
	}
}

func encUint64(i *encInstr, state *encState) {
	p := unsafe.Pointer(state.base+i.offset);
	for indir := i.indir; indir > 0; indir-- {
		p = *(*unsafe.Pointer)(p);
		if p == nil {
			return
		}
	}
	v := *(*uint64)(p);
	if v != 0 {
		EncodeUint(state.w, uint64(i.field));
		EncodeUint(state.w, v);
	}
}

// Floating-point numbers are transmitted as uint64s holding the bits
// of the underlying representation.  They are sent byte-reversed, with
// the exponent end coming out first, so integer floating point numbers
// (for example) transmit more compactly.  This routine does the
// swizzling.
func floatBits(f float64) uint64 {
	u := math.Float64bits(f);
	var v uint64;
	for i := 0; i < 8; i++ {
		v <<= 8;
		v |= u & 0xFF;
		u >>= 8;
	}
	return v;
}

func encFloat(i *encInstr, state *encState) {
	p := unsafe.Pointer(state.base+i.offset);
	for indir := i.indir; indir > 0; indir-- {
		p = *(*unsafe.Pointer)(p);
		if p == nil {
			return
		}
	}
	f := float(*(*float)(p));
	if f != 0 {
		v := floatBits(float64(f));
		EncodeUint(state.w, uint64(i.field));
		EncodeUint(state.w, v);
	}
}

func encFloat32(i *encInstr, state *encState) {
	p := unsafe.Pointer(state.base+i.offset);
	for indir := i.indir; indir > 0; indir-- {
		p = *(*unsafe.Pointer)(p);
		if p == nil {
			return
		}
	}
	f := float32(*(*float32)(p));
	if f != 0 {
		v := floatBits(float64(f));
		EncodeUint(state.w, uint64(i.field));
		EncodeUint(state.w, v);
	}
}

func encFloat64(i *encInstr, state *encState) {
	p := unsafe.Pointer(state.base+i.offset);
	for indir := i.indir; indir > 0; indir-- {
		p = *(*unsafe.Pointer)(p);
		if p == nil {
			return
		}
	}
	f := *(*float64)(p);
	if f != 0 {
		v := floatBits(f);
		EncodeUint(state.w, uint64(i.field));
		EncodeUint(state.w, v);
	}
}
