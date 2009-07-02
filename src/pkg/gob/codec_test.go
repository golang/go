// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gob

import (
	"bytes";
	"gob";
	"os";
	"reflect";
	"testing";
	"unsafe";
)
import "fmt" // TODO DELETE

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
	b := new(bytes.Buffer);
	encState := new(EncState);
	encState.w = b;
	for i, tt := range encodeT {
		b.Reset();
		EncodeUint(encState, tt.x);
		if encState.err != nil {
			t.Error("EncodeUint:", tt.x, encState.err)
		}
		if !bytes.Equal(tt.b, b.Data()) {
			t.Errorf("EncodeUint: expected % x got % x", tt.b, b.Data())
		}
	}
	decState := new(DecState);
	decState.r = b;
	for u := uint64(0); ; u = (u+1) * 7 {
		b.Reset();
		EncodeUint(encState, u);
		if encState.err != nil {
			t.Error("EncodeUint:", u, encState.err)
		}
		v := DecodeUint(decState);
		if decState.err != nil {
			t.Error("DecodeUint:", u, decState.err)
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
	var b = new(bytes.Buffer);
	encState := new(EncState);
	encState.w = b;
	EncodeInt(encState, i);
	if encState.err != nil {
		t.Error("EncodeInt:", i, encState.err)
	}
	decState := new(DecState);
	decState.r = b;
	j := DecodeInt(decState);
	if decState.err != nil {
		t.Error("DecodeInt:", i, decState.err)
	}
	if i != j {
		t.Errorf("Encode/Decode: sent %#x received %#x\n", uint64(i), uint64(j))
	}
}

// Test basic encode/decode routines for signed integers
func TestIntCodec(t *testing.T) {
	var b = new(bytes.Buffer);
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

// The result of encoding three true booleans with field numbers 0, 1, 2
var boolResult = []byte{0x81, 0x81, 0x81, 0x81, 0x81, 0x81}
// The result of encoding three numbers = 17 with field numbers 0, 1, 2
var signedResult = []byte{0x81, 0xa2, 0x81, 0xa2, 0x81, 0xa2}
var unsignedResult = []byte{0x81, 0x91, 0x81, 0x91, 0x81, 0x91}
var floatResult = []byte{0x81, 0x40, 0xe2, 0x81, 0x40, 0xe2, 0x81, 0x40, 0xe2}

func newEncState(b *bytes.Buffer) *EncState {
	b.Reset();
	state := new(EncState);
	state.w = b;
	state.fieldnum = -1;
	return state;
}

func encAddrOf(state *EncState, instr *encInstr) unsafe.Pointer {
	p := unsafe.Pointer(state.base+instr.offset);
	return encIndirect(p, instr.indir);
}

// Test instruction execution for encoding.
// Do not run the machine yet; instead do individual instructions crafted by hand.
func TestScalarEncInstructions(t *testing.T) {
	var b = new(bytes.Buffer);

	// bool
	{
		v := true;
		pv := &v;
		ppv := &pv;
		data := (struct { a bool; b *bool; c **bool }){ v, pv, ppv };
		instr := &encInstr{ encBool, 0, 0, 0 };
		state := newEncState(b);
		state.base = uintptr(unsafe.Pointer(&data));
		instr.op(instr, state, encAddrOf(state, instr));
		state.fieldnum = 0;
		instr.field = 1;
		instr.indir = 1;
		instr.offset = uintptr(unsafe.Offsetof(data.b));
		instr.op(instr, state, encAddrOf(state, instr));
		state.fieldnum = 1;
		instr.field = 2;
		instr.indir = 2;
		instr.offset = uintptr(unsafe.Offsetof(data.c));
		instr.op(instr, state, encAddrOf(state, instr));
		if !bytes.Equal(boolResult, b.Data()) {
			t.Errorf("bool enc instructions: expected % x got % x", boolResult, b.Data())
		}
	}

	// int
	{
		b.Reset();
		v := 17;
		pv := &v;
		ppv := &pv;
		data := (struct { a int; b *int; c **int }){ v, pv, ppv };
		instr := &encInstr{ encInt, 0, 0, 0 };
		state := newEncState(b);
		state.base = uintptr(unsafe.Pointer(&data));
		instr.op(instr, state, encAddrOf(state, instr));
		state.fieldnum = 0;
		instr.field = 1;
		instr.indir = 1;
		instr.offset = uintptr(unsafe.Offsetof(data.b));
		instr.op(instr, state, encAddrOf(state, instr));
		state.fieldnum = 1;
		instr.field = 2;
		instr.indir = 2;
		instr.offset = uintptr(unsafe.Offsetof(data.c));
		instr.op(instr, state, encAddrOf(state, instr));
		if !bytes.Equal(signedResult, b.Data()) {
			t.Errorf("int enc instructions: expected % x got % x", signedResult, b.Data())
		}
	}

	// uint
	{
		b.Reset();
		v := uint(17);
		pv := &v;
		ppv := &pv;
		data := (struct { a uint; b *uint; c **uint }){ v, pv, ppv };
		instr := &encInstr{ encUint, 0, 0, 0 };
		state := newEncState(b);
		state.base = uintptr(unsafe.Pointer(&data));
		instr.op(instr, state, encAddrOf(state, instr));
		state.fieldnum = 0;
		instr.field = 1;
		instr.indir = 1;
		instr.offset = uintptr(unsafe.Offsetof(data.b));
		instr.op(instr, state, encAddrOf(state, instr));
		state.fieldnum = 1;
		instr.field = 2;
		instr.indir = 2;
		instr.offset = uintptr(unsafe.Offsetof(data.c));
		instr.op(instr, state, encAddrOf(state, instr));
		if !bytes.Equal(unsignedResult, b.Data()) {
			t.Errorf("uint enc instructions: expected % x got % x", unsignedResult, b.Data())
		}
	}

	// int8
	{
		b.Reset();
		v := int8(17);
		pv := &v;
		ppv := &pv;
		data := (struct { a int8; b *int8; c **int8 }){ v, pv, ppv };
		instr := &encInstr{ encInt, 0, 0, 0 };
		state := newEncState(b);
		state.base = uintptr(unsafe.Pointer(&data));
		instr.op(instr, state, encAddrOf(state, instr));
		state.fieldnum = 0;
		instr.field = 1;
		instr.indir = 1;
		instr.offset = uintptr(unsafe.Offsetof(data.b));
		instr.op(instr, state, encAddrOf(state, instr));
		state.fieldnum = 1;
		instr.field = 2;
		instr.indir = 2;
		instr.offset = uintptr(unsafe.Offsetof(data.c));
		instr.op(instr, state, encAddrOf(state, instr));
		if !bytes.Equal(signedResult, b.Data()) {
			t.Errorf("int8 enc instructions: expected % x got % x", signedResult, b.Data())
		}
	}

	// uint8
	{
		b.Reset();
		v := uint8(17);
		pv := &v;
		ppv := &pv;
		data := (struct { a uint8; b *uint8; c **uint8 }){ v, pv, ppv };
		instr := &encInstr{ encUint, 0, 0, 0 };
		state := newEncState(b);
		state.base = uintptr(unsafe.Pointer(&data));
		instr.op(instr, state, encAddrOf(state, instr));
		state.fieldnum = 0;
		instr.field = 1;
		instr.indir = 1;
		instr.offset = uintptr(unsafe.Offsetof(data.b));
		instr.op(instr, state, encAddrOf(state, instr));
		state.fieldnum = 1;
		instr.field = 2;
		instr.indir = 2;
		instr.offset = uintptr(unsafe.Offsetof(data.c));
		instr.op(instr, state, encAddrOf(state, instr));
		if !bytes.Equal(unsignedResult, b.Data()) {
			t.Errorf("uint8 enc instructions: expected % x got % x", unsignedResult, b.Data())
		}
	}

	// int16
	{
		b.Reset();
		v := int16(17);
		pv := &v;
		ppv := &pv;
		data := (struct { a int16; b *int16; c **int16 }){ v, pv, ppv };
		instr := &encInstr{ encInt16, 0, 0, 0 };
		state := newEncState(b);
		state.base = uintptr(unsafe.Pointer(&data));
		instr.op(instr, state, encAddrOf(state, instr));
		state.fieldnum = 0;
		instr.field = 1;
		instr.indir = 1;
		instr.offset = uintptr(unsafe.Offsetof(data.b));
		instr.op(instr, state, encAddrOf(state, instr));
		state.fieldnum = 1;
		instr.field = 2;
		instr.indir = 2;
		instr.offset = uintptr(unsafe.Offsetof(data.c));
		instr.op(instr, state, encAddrOf(state, instr));
		if !bytes.Equal(signedResult, b.Data()) {
			t.Errorf("int16 enc instructions: expected % x got % x", signedResult, b.Data())
		}
	}

	// uint16
	{
		b.Reset();
		v := uint16(17);
		pv := &v;
		ppv := &pv;
		data := (struct { a uint16; b *uint16; c **uint16 }){ v, pv, ppv };
		instr := &encInstr{ encUint16, 0, 0, 0 };
		state := newEncState(b);
		state.base = uintptr(unsafe.Pointer(&data));
		instr.op(instr, state, encAddrOf(state, instr));
		state.fieldnum = 0;
		instr.field = 1;
		instr.indir = 1;
		instr.offset = uintptr(unsafe.Offsetof(data.b));
		instr.op(instr, state, encAddrOf(state, instr));
		state.fieldnum = 1;
		instr.field = 2;
		instr.indir = 2;
		instr.offset = uintptr(unsafe.Offsetof(data.c));
		instr.op(instr, state, encAddrOf(state, instr));
		if !bytes.Equal(unsignedResult, b.Data()) {
			t.Errorf("uint16 enc instructions: expected % x got % x", unsignedResult, b.Data())
		}
	}

	// int32
	{
		b.Reset();
		v := int32(17);
		pv := &v;
		ppv := &pv;
		data := (struct { a int32; b *int32; c **int32 }){ v, pv, ppv };
		instr := &encInstr{ encInt32, 0, 0, 0 };
		state := newEncState(b);
		state.base = uintptr(unsafe.Pointer(&data));
		instr.op(instr, state, encAddrOf(state, instr));
		state.fieldnum = 0;
		instr.field = 1;
		instr.indir = 1;
		instr.offset = uintptr(unsafe.Offsetof(data.b));
		instr.op(instr, state, encAddrOf(state, instr));
		state.fieldnum = 1;
		instr.field = 2;
		instr.indir = 2;
		instr.offset = uintptr(unsafe.Offsetof(data.c));
		instr.op(instr, state, encAddrOf(state, instr));
		if !bytes.Equal(signedResult, b.Data()) {
			t.Errorf("int32 enc instructions: expected % x got % x", signedResult, b.Data())
		}
	}

	// uint32
	{
		b.Reset();
		v := uint32(17);
		pv := &v;
		ppv := &pv;
		data := (struct { a uint32; b *uint32; c **uint32 }){ v, pv, ppv };
		instr := &encInstr{ encUint32, 0, 0, 0 };
		state := newEncState(b);
		state.base = uintptr(unsafe.Pointer(&data));
		instr.op(instr, state, encAddrOf(state, instr));
		state.fieldnum = 0;
		instr.field = 1;
		instr.indir = 1;
		instr.offset = uintptr(unsafe.Offsetof(data.b));
		instr.op(instr, state, encAddrOf(state, instr));
		state.fieldnum = 1;
		instr.field = 2;
		instr.indir = 2;
		instr.offset = uintptr(unsafe.Offsetof(data.c));
		instr.op(instr, state, encAddrOf(state, instr));
		if !bytes.Equal(unsignedResult, b.Data()) {
			t.Errorf("uint32 enc instructions: expected % x got % x", unsignedResult, b.Data())
		}
	}

	// int64
	{
		b.Reset();
		v := int64(17);
		pv := &v;
		ppv := &pv;
		data := (struct { a int64; b *int64; c **int64 }){ v, pv, ppv };
		instr := &encInstr{ encInt64, 0, 0, 0 };
		state := newEncState(b);
		state.base = uintptr(unsafe.Pointer(&data));
		instr.op(instr, state, encAddrOf(state, instr));
		state.fieldnum = 0;
		instr.field = 1;
		instr.indir = 1;
		instr.offset = uintptr(unsafe.Offsetof(data.b));
		instr.op(instr, state, encAddrOf(state, instr));
		state.fieldnum = 1;
		instr.field = 2;
		instr.indir = 2;
		instr.offset = uintptr(unsafe.Offsetof(data.c));
		instr.op(instr, state, encAddrOf(state, instr));
		if !bytes.Equal(signedResult, b.Data()) {
			t.Errorf("int64 enc instructions: expected % x got % x", signedResult, b.Data())
		}
	}

	// uint64
	{
		b.Reset();
		v := uint64(17);
		pv := &v;
		ppv := &pv;
		data := (struct { a uint64; b *uint64; c **uint64 }){ v, pv, ppv };
		instr := &encInstr{ encUint, 0, 0, 0 };
		state := newEncState(b);
		state.base = uintptr(unsafe.Pointer(&data));
		instr.op(instr, state, encAddrOf(state, instr));
		state.fieldnum = 0;
		instr.field = 1;
		instr.indir = 1;
		instr.offset = uintptr(unsafe.Offsetof(data.b));
		instr.op(instr, state, encAddrOf(state, instr));
		state.fieldnum = 1;
		instr.field = 2;
		instr.indir = 2;
		instr.offset = uintptr(unsafe.Offsetof(data.c));
		instr.op(instr, state, encAddrOf(state, instr));
		if !bytes.Equal(unsignedResult, b.Data()) {
			t.Errorf("uint64 enc instructions: expected % x got % x", unsignedResult, b.Data())
		}
	}

	// float
	{
		b.Reset();
		v := float(17);
		pv := &v;
		ppv := &pv;
		data := (struct { a float; b *float; c **float }){ v, pv, ppv };
		instr := &encInstr{ encFloat, 0, 0, 0 };
		state := newEncState(b);
		state.base = uintptr(unsafe.Pointer(&data));
		instr.op(instr, state, encAddrOf(state, instr));
		state.fieldnum = 0;
		instr.field = 1;
		instr.indir = 1;
		instr.offset = uintptr(unsafe.Offsetof(data.b));
		instr.op(instr, state, encAddrOf(state, instr));
		state.fieldnum = 1;
		instr.field = 2;
		instr.indir = 2;
		instr.offset = uintptr(unsafe.Offsetof(data.c));
		instr.op(instr, state, encAddrOf(state, instr));
		if !bytes.Equal(floatResult, b.Data()) {
			t.Errorf("float enc instructions: expected % x got % x", floatResult, b.Data())
		}
	}

	// float32
	{
		b.Reset();
		v := float32(17);
		pv := &v;
		ppv := &pv;
		data := (struct { a float32; b *float32; c **float32 }){ v, pv, ppv };
		instr := &encInstr{ encFloat32, 0, 0, 0 };
		state := newEncState(b);
		state.base = uintptr(unsafe.Pointer(&data));
		instr.op(instr, state, encAddrOf(state, instr));
		state.fieldnum = 0;
		instr.field = 1;
		instr.indir = 1;
		instr.offset = uintptr(unsafe.Offsetof(data.b));
		instr.op(instr, state, encAddrOf(state, instr));
		state.fieldnum = 1;
		instr.field = 2;
		instr.indir = 2;
		instr.offset = uintptr(unsafe.Offsetof(data.c));
		instr.op(instr, state, encAddrOf(state, instr));
		if !bytes.Equal(floatResult, b.Data()) {
			t.Errorf("float32 enc instructions: expected % x got % x", floatResult, b.Data())
		}
	}

	// float64
	{
		b.Reset();
		v := float64(17);
		pv := &v;
		ppv := &pv;
		data := (struct { a float64; b *float64; c **float64 }){ v, pv, ppv };
		instr := &encInstr{ encFloat64, 0, 0, 0 };
		state := newEncState(b);
		state.base = uintptr(unsafe.Pointer(&data));
		instr.op(instr, state, encAddrOf(state, instr));
		state.fieldnum = 0;
		instr.field = 1;
		instr.indir = 1;
		instr.offset = uintptr(unsafe.Offsetof(data.b));
		instr.op(instr, state, encAddrOf(state, instr));
		state.fieldnum = 1;
		instr.field = 2;
		instr.indir = 2;
		instr.offset = uintptr(unsafe.Offsetof(data.c));
		instr.op(instr, state, encAddrOf(state, instr));
		if !bytes.Equal(floatResult, b.Data()) {
			t.Errorf("float64 enc instructions: expected % x got % x", floatResult, b.Data())
		}
	}
}

func expectField(n int, state *DecState, t *testing.T) {
	v := int(DecodeUint(state));
	if state.err != nil {
		t.Fatalf("decoding field number %d: %v", n, state.err);
	}
	if v + state.fieldnum != n {
		t.Fatalf("decoding field number %d, got %d", n, v);
	}
	state.fieldnum = n;
}

func newDecState(data []byte) *DecState {
	state := new(DecState);
	state.r = bytes.NewBuffer(data);
	state.fieldnum = -1;
	return state;
}

// derive the address of a field, after indirecting indir times.
func decAddrOf(state *DecState, instr *decInstr) unsafe.Pointer {
	p := unsafe.Pointer(state.base+instr.offset);
	return decIndirect(p, instr.indir);
}

// Test instruction execution for decoding.
// Do not run the machine yet; instead do individual instructions crafted by hand.
func TestScalarDecInstructions(t *testing.T) {

	// bool
	{
		var data struct { a bool; b *bool; c **bool };
		instr := &decInstr{ decBool, 0, 0, 0 };
		state := newDecState(boolResult);
		state.base = uintptr(unsafe.Pointer(&data));
		expectField(0, state, t);
		instr.op(instr, state, decAddrOf(state, instr));
		instr.field = 1;
		instr.indir = 1;
		instr.offset = uintptr(unsafe.Offsetof(data.b));
		expectField(1, state, t);
		instr.op(instr, state, decAddrOf(state, instr));
		instr.field = 2;
		instr.indir = 2;
		instr.offset = uintptr(unsafe.Offsetof(data.c));
		expectField(2, state, t);
		instr.op(instr, state, decAddrOf(state, instr));
		if data.a != true {
			t.Errorf("int a = %v not true", data.a)
		}
		if *data.b != true {
			t.Errorf("int b = %v not true", *data.b)
		}
		if **data.c != true {
			t.Errorf("int c = %v not true", **data.c)
		}
	}

	// int
	{
		var data struct { a int; b *int; c **int };
		instr := &decInstr{ decInt, 0, 0, 0 };
		state := newDecState(signedResult);
		state.base = uintptr(unsafe.Pointer(&data));
		expectField(0, state, t);
		instr.op(instr, state, decAddrOf(state, instr));
		instr.field = 1;
		instr.indir = 1;
		instr.offset = uintptr(unsafe.Offsetof(data.b));
		expectField(1, state, t);
		instr.op(instr, state, decAddrOf(state, instr));
		instr.field = 2;
		instr.indir = 2;
		instr.offset = uintptr(unsafe.Offsetof(data.c));
		expectField(2, state, t);
		instr.op(instr, state, decAddrOf(state, instr));
		if data.a != 17 {
			t.Errorf("int a = %v not 17", data.a)
		}
		if *data.b != 17 {
			t.Errorf("int b = %v not 17", *data.b)
		}
		if **data.c != 17 {
			t.Errorf("int c = %v not 17", **data.c)
		}
	}

	// uint
	{
		var data struct { a uint; b *uint; c **uint };
		instr := &decInstr{ decUint, 0, 0, 0 };
		state := newDecState(unsignedResult);
		state.base = uintptr(unsafe.Pointer(&data));
		expectField(0, state, t);
		instr.op(instr, state, decAddrOf(state, instr));
		instr.field = 1;
		instr.indir = 1;
		instr.offset = uintptr(unsafe.Offsetof(data.b));
		expectField(1, state, t);
		instr.op(instr, state, decAddrOf(state, instr));
		instr.field = 2;
		instr.indir = 2;
		instr.offset = uintptr(unsafe.Offsetof(data.c));
		expectField(2, state, t);
		instr.op(instr, state, decAddrOf(state, instr));
		if data.a != 17 {
			t.Errorf("int a = %v not 17", data.a)
		}
		if *data.b != 17 {
			t.Errorf("int b = %v not 17", *data.b)
		}
		if **data.c != 17 {
			t.Errorf("int c = %v not 17", **data.c)
		}
	}

	// int8
	{
		var data struct { a int8; b *int8; c **int8 };
		instr := &decInstr{ decInt8, 0, 0, 0 };
		state := newDecState(signedResult);
		state.base = uintptr(unsafe.Pointer(&data));
		expectField(0, state, t);
		instr.op(instr, state, decAddrOf(state, instr));
		instr.field = 1;
		instr.indir = 1;
		instr.offset = uintptr(unsafe.Offsetof(data.b));
		expectField(1, state, t);
		instr.op(instr, state, decAddrOf(state, instr));
		instr.field = 2;
		instr.indir = 2;
		instr.offset = uintptr(unsafe.Offsetof(data.c));
		expectField(2, state, t);
		instr.op(instr, state, decAddrOf(state, instr));
		if data.a != 17 {
			t.Errorf("int a = %v not 17", data.a)
		}
		if *data.b != 17 {
			t.Errorf("int b = %v not 17", *data.b)
		}
		if **data.c != 17 {
			t.Errorf("int c = %v not 17", **data.c)
		}
	}

	// uint8
	{
		var data struct { a uint8; b *uint8; c **uint8 };
		instr := &decInstr{ decUint8, 0, 0, 0 };
		state := newDecState(unsignedResult);
		state.base = uintptr(unsafe.Pointer(&data));
		expectField(0, state, t);
		instr.op(instr, state, decAddrOf(state, instr));
		instr.field = 1;
		instr.indir = 1;
		instr.offset = uintptr(unsafe.Offsetof(data.b));
		expectField(1, state, t);
		instr.op(instr, state, decAddrOf(state, instr));
		instr.field = 2;
		instr.indir = 2;
		instr.offset = uintptr(unsafe.Offsetof(data.c));
		expectField(2, state, t);
		instr.op(instr, state, decAddrOf(state, instr));
		if data.a != 17 {
			t.Errorf("int a = %v not 17", data.a)
		}
		if *data.b != 17 {
			t.Errorf("int b = %v not 17", *data.b)
		}
		if **data.c != 17 {
			t.Errorf("int c = %v not 17", **data.c)
		}
	}

	// int16
	{
		var data struct { a int16; b *int16; c **int16 };
		instr := &decInstr{ decInt16, 0, 0, 0 };
		state := newDecState(signedResult);
		state.base = uintptr(unsafe.Pointer(&data));
		expectField(0, state, t);
		instr.op(instr, state, decAddrOf(state, instr));
		instr.field = 1;
		instr.indir = 1;
		instr.offset = uintptr(unsafe.Offsetof(data.b));
		expectField(1, state, t);
		instr.op(instr, state, decAddrOf(state, instr));
		instr.field = 2;
		instr.indir = 2;
		instr.offset = uintptr(unsafe.Offsetof(data.c));
		expectField(2, state, t);
		instr.op(instr, state, decAddrOf(state, instr));
		if data.a != 17 {
			t.Errorf("int a = %v not 17", data.a)
		}
		if *data.b != 17 {
			t.Errorf("int b = %v not 17", *data.b)
		}
		if **data.c != 17 {
			t.Errorf("int c = %v not 17", **data.c)
		}
	}

	// uint16
	{
		var data struct { a uint16; b *uint16; c **uint16 };
		instr := &decInstr{ decUint16, 0, 0, 0 };
		state := newDecState(unsignedResult);
		state.base = uintptr(unsafe.Pointer(&data));
		expectField(0, state, t);
		instr.op(instr, state, decAddrOf(state, instr));
		instr.field = 1;
		instr.indir = 1;
		instr.offset = uintptr(unsafe.Offsetof(data.b));
		expectField(1, state, t);
		instr.op(instr, state, decAddrOf(state, instr));
		instr.field = 2;
		instr.indir = 2;
		instr.offset = uintptr(unsafe.Offsetof(data.c));
		expectField(2, state, t);
		instr.op(instr, state, decAddrOf(state, instr));
		if data.a != 17 {
			t.Errorf("int a = %v not 17", data.a)
		}
		if *data.b != 17 {
			t.Errorf("int b = %v not 17", *data.b)
		}
		if **data.c != 17 {
			t.Errorf("int c = %v not 17", **data.c)
		}
	}

	// int32
	{
		var data struct { a int32; b *int32; c **int32 };
		instr := &decInstr{ decInt32, 0, 0, 0 };
		state := newDecState(signedResult);
		state.base = uintptr(unsafe.Pointer(&data));
		expectField(0, state, t);
		instr.op(instr, state, decAddrOf(state, instr));
		instr.field = 1;
		instr.indir = 1;
		instr.offset = uintptr(unsafe.Offsetof(data.b));
		expectField(1, state, t);
		instr.op(instr, state, decAddrOf(state, instr));
		instr.field = 2;
		instr.indir = 2;
		instr.offset = uintptr(unsafe.Offsetof(data.c));
		expectField(2, state, t);
		instr.op(instr, state, decAddrOf(state, instr));
		if data.a != 17 {
			t.Errorf("int a = %v not 17", data.a)
		}
		if *data.b != 17 {
			t.Errorf("int b = %v not 17", *data.b)
		}
		if **data.c != 17 {
			t.Errorf("int c = %v not 17", **data.c)
		}
	}

	// uint32
	{
		var data struct { a uint32; b *uint32; c **uint32 };
		instr := &decInstr{ decUint32, 0, 0, 0 };
		state := newDecState(unsignedResult);
		state.base = uintptr(unsafe.Pointer(&data));
		expectField(0, state, t);
		instr.op(instr, state, decAddrOf(state, instr));
		instr.field = 1;
		instr.indir = 1;
		instr.offset = uintptr(unsafe.Offsetof(data.b));
		expectField(1, state, t);
		instr.op(instr, state, decAddrOf(state, instr));
		instr.field = 2;
		instr.indir = 2;
		instr.offset = uintptr(unsafe.Offsetof(data.c));
		expectField(2, state, t);
		instr.op(instr, state, decAddrOf(state, instr));
		if data.a != 17 {
			t.Errorf("int a = %v not 17", data.a)
		}
		if *data.b != 17 {
			t.Errorf("int b = %v not 17", *data.b)
		}
		if **data.c != 17 {
			t.Errorf("int c = %v not 17", **data.c)
		}
	}

	// int64
	{
		var data struct { a int64; b *int64; c **int64 };
		instr := &decInstr{ decInt64, 0, 0, 0 };
		state := newDecState(signedResult);
		state.base = uintptr(unsafe.Pointer(&data));
		expectField(0, state, t);
		instr.op(instr, state, decAddrOf(state, instr));
		instr.field = 1;
		instr.indir = 1;
		instr.offset = uintptr(unsafe.Offsetof(data.b));
		expectField(1, state, t);
		instr.op(instr, state, decAddrOf(state, instr));
		instr.field = 2;
		instr.indir = 2;
		instr.offset = uintptr(unsafe.Offsetof(data.c));
		expectField(2, state, t);
		instr.op(instr, state, decAddrOf(state, instr));
		if data.a != 17 {
			t.Errorf("int a = %v not 17", data.a)
		}
		if *data.b != 17 {
			t.Errorf("int b = %v not 17", *data.b)
		}
		if **data.c != 17 {
			t.Errorf("int c = %v not 17", **data.c)
		}
	}

	// uint64
	{
		var data struct { a uint64; b *uint64; c **uint64 };
		instr := &decInstr{ decUint64, 0, 0, 0 };
		state := newDecState(unsignedResult);
		state.base = uintptr(unsafe.Pointer(&data));
		expectField(0, state, t);
		instr.op(instr, state, decAddrOf(state, instr));
		instr.field = 1;
		instr.indir = 1;
		instr.offset = uintptr(unsafe.Offsetof(data.b));
		expectField(1, state, t);
		instr.op(instr, state, decAddrOf(state, instr));
		instr.field = 2;
		instr.indir = 2;
		instr.offset = uintptr(unsafe.Offsetof(data.c));
		expectField(2, state, t);
		instr.op(instr, state, decAddrOf(state, instr));
		if data.a != 17 {
			t.Errorf("int a = %v not 17", data.a)
		}
		if *data.b != 17 {
			t.Errorf("int b = %v not 17", *data.b)
		}
		if **data.c != 17 {
			t.Errorf("int c = %v not 17", **data.c)
		}
	}

	// float
	{
		var data struct { a float; b *float; c **float };
		instr := &decInstr{ decFloat, 0, 0, 0 };
		state := newDecState(floatResult);
		state.base = uintptr(unsafe.Pointer(&data));
		expectField(0, state, t);
		instr.op(instr, state, decAddrOf(state, instr));
		instr.field = 1;
		instr.indir = 1;
		instr.offset = uintptr(unsafe.Offsetof(data.b));
		expectField(1, state, t);
		instr.op(instr, state, decAddrOf(state, instr));
		instr.field = 2;
		instr.indir = 2;
		instr.offset = uintptr(unsafe.Offsetof(data.c));
		expectField(2, state, t);
		instr.op(instr, state, decAddrOf(state, instr));
		if data.a != 17 {
			t.Errorf("int a = %v not 17", data.a)
		}
		if *data.b != 17 {
			t.Errorf("int b = %v not 17", *data.b)
		}
		if **data.c != 17 {
			t.Errorf("int c = %v not 17", **data.c)
		}
	}

	// float32
	{
		var data struct { a float32; b *float32; c **float32 };
		instr := &decInstr{ decFloat32, 0, 0, 0 };
		state := newDecState(floatResult);
		state.base = uintptr(unsafe.Pointer(&data));
		expectField(0, state, t);
		instr.op(instr, state, decAddrOf(state, instr));
		instr.field = 1;
		instr.indir = 1;
		instr.offset = uintptr(unsafe.Offsetof(data.b));
		expectField(1, state, t);
		instr.op(instr, state, decAddrOf(state, instr));
		instr.field = 2;
		instr.indir = 2;
		instr.offset = uintptr(unsafe.Offsetof(data.c));
		expectField(2, state, t);
		instr.op(instr, state, decAddrOf(state, instr));
		if data.a != 17 {
			t.Errorf("int a = %v not 17", data.a)
		}
		if *data.b != 17 {
			t.Errorf("int b = %v not 17", *data.b)
		}
		if **data.c != 17 {
			t.Errorf("int c = %v not 17", **data.c)
		}
	}

	// float64
	{
		var data struct { a float64; b *float64; c **float64 };
		instr := &decInstr{ decFloat64, 0, 0, 0 };
		state := newDecState(floatResult);
		state.base = uintptr(unsafe.Pointer(&data));
		expectField(0, state, t);
		instr.op(instr, state, decAddrOf(state, instr));
		instr.field = 1;
		instr.indir = 1;
		instr.offset = uintptr(unsafe.Offsetof(data.b));
		expectField(1, state, t);
		instr.op(instr, state, decAddrOf(state, instr));
		instr.field = 2;
		instr.indir = 2;
		instr.offset = uintptr(unsafe.Offsetof(data.c));
		expectField(2, state, t);
		instr.op(instr, state, decAddrOf(state, instr));
		if data.a != 17 {
			t.Errorf("int a = %v not 17", data.a)
		}
		if *data.b != 17 {
			t.Errorf("int b = %v not 17", *data.b)
		}
		if **data.c != 17 {
			t.Errorf("int c = %v not 17", **data.c)
		}
	}
}

type T1 struct {
	a, b,c int
}

func TestEncode(t *testing.T) {
	t1 := &T1{17,18,-5};
	b := new(bytes.Buffer);
	Encode(b, t1);
	var _t1 T1;
	Decode(b, &_t1);
	if !reflect.DeepEqual(t1, &_t1) {
		t.Errorf("encode expected %v got %v", *t1, _t1);
	}
}
