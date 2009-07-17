// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gob

import (
	"bytes";
	"gob";
	"os";
	"reflect";
	"strings";
	"testing";
	"unsafe";
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
	b := new(bytes.Buffer);
	encState := new(encoderState);
	encState.b = b;
	for i, tt := range encodeT {
		b.Reset();
		encodeUint(encState, tt.x);
		if encState.err != nil {
			t.Error("encodeUint:", tt.x, encState.err)
		}
		if !bytes.Equal(tt.b, b.Data()) {
			t.Errorf("encodeUint: expected % x got % x", tt.b, b.Data())
		}
	}
	decState := new(decodeState);
	decState.b = b;
	for u := uint64(0); ; u = (u+1) * 7 {
		b.Reset();
		encodeUint(encState, u);
		if encState.err != nil {
			t.Error("encodeUint:", u, encState.err)
		}
		v := decodeUint(decState);
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
	encState := new(encoderState);
	encState.b = b;
	encodeInt(encState, i);
	if encState.err != nil {
		t.Error("encodeInt:", i, encState.err)
	}
	decState := new(decodeState);
	decState.b = b;
	j := decodeInt(decState);
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

// The result of encoding a true boolean with field number 6
var boolResult = []byte{0x87, 0x81}
// The result of encoding a number 17 with field number 6
var signedResult = []byte{0x87, 0xa2}
var unsignedResult = []byte{0x87, 0x91}
var floatResult = []byte{0x87, 0x40, 0xe2}
// The result of encoding "hello" with field number 6
var bytesResult = []byte{0x87, 0x85, 'h', 'e', 'l', 'l', 'o'}

func newencoderState(b *bytes.Buffer) *encoderState {
	b.Reset();
	state := new(encoderState);
	state.b = b;
	state.fieldnum = -1;
	return state;
}

// Test instruction execution for encoding.
// Do not run the machine yet; instead do individual instructions crafted by hand.
func TestScalarEncInstructions(t *testing.T) {
	var b = new(bytes.Buffer);

	// bool
	{
		data := struct { a bool } { true };
		instr := &encInstr{ encBool, 6, 0, 0 };
		state := newencoderState(b);
		instr.op(instr, state, unsafe.Pointer(&data));
		if !bytes.Equal(boolResult, b.Data()) {
			t.Errorf("bool enc instructions: expected % x got % x", boolResult, b.Data())
		}
	}

	// int
	{
		b.Reset();
		data := struct { a int } { 17 };
		instr := &encInstr{ encInt, 6, 0, 0 };
		state := newencoderState(b);
		instr.op(instr, state, unsafe.Pointer(&data));
		if !bytes.Equal(signedResult, b.Data()) {
			t.Errorf("int enc instructions: expected % x got % x", signedResult, b.Data())
		}
	}

	// uint
	{
		b.Reset();
		data := struct { a uint } { 17 };
		instr := &encInstr{ encUint, 6, 0, 0 };
		state := newencoderState(b);
		instr.op(instr, state, unsafe.Pointer(&data));
		if !bytes.Equal(unsignedResult, b.Data()) {
			t.Errorf("uint enc instructions: expected % x got % x", unsignedResult, b.Data())
		}
	}

	// int8
	{
		b.Reset();
		data := struct { a int8 } { 17 };
		instr := &encInstr{ encInt, 6, 0, 0 };
		state := newencoderState(b);
		instr.op(instr, state, unsafe.Pointer(&data));
		if !bytes.Equal(signedResult, b.Data()) {
			t.Errorf("int8 enc instructions: expected % x got % x", signedResult, b.Data())
		}
	}

	// uint8
	{
		b.Reset();
		data := struct { a uint8 } { 17 };
		instr := &encInstr{ encUint, 6, 0, 0 };
		state := newencoderState(b);
		instr.op(instr, state, unsafe.Pointer(&data));
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
		data := struct { a int16 } { 17 };
		instr := &encInstr{ encInt16, 6, 0, 0 };
		state := newencoderState(b);
		instr.op(instr, state, unsafe.Pointer(&data));
		if !bytes.Equal(signedResult, b.Data()) {
			t.Errorf("int16 enc instructions: expected % x got % x", signedResult, b.Data())
		}
	}

	// uint16
	{
		b.Reset();
		data := struct { a uint16 } { 17 };
		instr := &encInstr{ encUint16, 6, 0, 0 };
		state := newencoderState(b);
		instr.op(instr, state, unsafe.Pointer(&data));
		if !bytes.Equal(unsignedResult, b.Data()) {
			t.Errorf("uint16 enc instructions: expected % x got % x", unsignedResult, b.Data())
		}
	}

	// int32
	{
		b.Reset();
		data := struct { a int32 } { 17 };
		instr := &encInstr{ encInt32, 6, 0, 0 };
		state := newencoderState(b);
		instr.op(instr, state, unsafe.Pointer(&data));
		if !bytes.Equal(signedResult, b.Data()) {
			t.Errorf("int32 enc instructions: expected % x got % x", signedResult, b.Data())
		}
	}

	// uint32
	{
		b.Reset();
		data := struct { a uint32 } { 17 };
		instr := &encInstr{ encUint32, 6, 0, 0 };
		state := newencoderState(b);
		instr.op(instr, state, unsafe.Pointer(&data));
		if !bytes.Equal(unsignedResult, b.Data()) {
			t.Errorf("uint32 enc instructions: expected % x got % x", unsignedResult, b.Data())
		}
	}

	// int64
	{
		b.Reset();
		data := struct { a int64 } { 17 };
		instr := &encInstr{ encInt64, 6, 0, 0 };
		state := newencoderState(b);
		instr.op(instr, state, unsafe.Pointer(&data));
		if !bytes.Equal(signedResult, b.Data()) {
			t.Errorf("int64 enc instructions: expected % x got % x", signedResult, b.Data())
		}
	}

	// uint64
	{
		b.Reset();
		data := struct { a uint64 } { 17 };
		instr := &encInstr{ encUint, 6, 0, 0 };
		state := newencoderState(b);
		instr.op(instr, state, unsafe.Pointer(&data));
		if !bytes.Equal(unsignedResult, b.Data()) {
			t.Errorf("uint64 enc instructions: expected % x got % x", unsignedResult, b.Data())
		}
	}

	// float
	{
		b.Reset();
		data := struct { a float } { 17 };
		instr := &encInstr{ encFloat, 6, 0, 0 };
		state := newencoderState(b);
		instr.op(instr, state, unsafe.Pointer(&data));
		if !bytes.Equal(floatResult, b.Data()) {
			t.Errorf("float enc instructions: expected % x got % x", floatResult, b.Data())
		}
	}

	// float32
	{
		b.Reset();
		data := struct { a float32 } { 17 };
		instr := &encInstr{ encFloat32, 6, 0, 0 };
		state := newencoderState(b);
		instr.op(instr, state, unsafe.Pointer(&data));
		if !bytes.Equal(floatResult, b.Data()) {
			t.Errorf("float32 enc instructions: expected % x got % x", floatResult, b.Data())
		}
	}

	// float64
	{
		b.Reset();
		data := struct { a float64 } { 17 };
		instr := &encInstr{ encFloat64, 6, 0, 0 };
		state := newencoderState(b);
		instr.op(instr, state, unsafe.Pointer(&data));
		if !bytes.Equal(floatResult, b.Data()) {
			t.Errorf("float64 enc instructions: expected % x got % x", floatResult, b.Data())
		}
	}

	// bytes == []uint8
	{
		b.Reset();
		data := struct { a []byte } { strings.Bytes("hello") };
		instr := &encInstr{ encUint8Array, 6, 0, 0 };
		state := newencoderState(b);
		instr.op(instr, state, unsafe.Pointer(&data));
		if !bytes.Equal(bytesResult, b.Data()) {
			t.Errorf("bytes enc instructions: expected % x got % x", bytesResult, b.Data())
		}
	}

	// string
	{
		b.Reset();
		data := struct { a string } { "hello" };
		instr := &encInstr{ encString, 6, 0, 0 };
		state := newencoderState(b);
		instr.op(instr, state, unsafe.Pointer(&data));
		if !bytes.Equal(bytesResult, b.Data()) {
			t.Errorf("string enc instructions: expected % x got % x", bytesResult, b.Data())
		}
	}
}

func execDec(typ string, instr *decInstr, state *decodeState, t *testing.T, p unsafe.Pointer) {
	v := int(decodeUint(state));
	if state.err != nil {
		t.Fatalf("decoding %s field: %v", typ, state.err);
	}
	if v + state.fieldnum != 6 {
		t.Fatalf("decoding field number %d, got %d", 6, v + state.fieldnum);
	}
	instr.op(instr, state, decIndirect(p, instr.indir));
	state.fieldnum = 6;
}

func newdecodeState(data []byte) *decodeState {
	state := new(decodeState);
	state.b = bytes.NewBuffer(data);
	state.fieldnum = -1;
	return state;
}

// Test instruction execution for decoding.
// Do not run the machine yet; instead do individual instructions crafted by hand.
func TestScalarDecInstructions(t *testing.T) {

	// bool
	{
		var data struct { a bool };
		instr := &decInstr{ decBool, 6, 0, 0 };
		state := newdecodeState(boolResult);
		execDec("bool", instr, state, t, unsafe.Pointer(&data));
		if data.a != true {
			t.Errorf("int a = %v not true", data.a)
		}
	}
	// int
	{
		var data struct { a int };
		instr := &decInstr{ decInt, 6, 0, 0 };
		state := newdecodeState(signedResult);
		execDec("int", instr, state, t, unsafe.Pointer(&data));
		if data.a != 17 {
			t.Errorf("int a = %v not 17", data.a)
		}
	}

	// uint
	{
		var data struct { a uint };
		instr := &decInstr{ decUint, 6, 0, 0 };
		state := newdecodeState(unsignedResult);
		execDec("uint", instr, state, t, unsafe.Pointer(&data));
		if data.a != 17 {
			t.Errorf("int a = %v not 17", data.a)
		}
	}

	// int8
	{
		var data struct { a int8 };
		instr := &decInstr{ decInt8, 6, 0, 0 };
		state := newdecodeState(signedResult);
		execDec("int8", instr, state, t, unsafe.Pointer(&data));
		if data.a != 17 {
			t.Errorf("int a = %v not 17", data.a)
		}
	}

	// uint8
	{
		var data struct { a uint8 };
		instr := &decInstr{ decUint8, 6, 0, 0 };
		state := newdecodeState(unsignedResult);
		execDec("uint8", instr, state, t, unsafe.Pointer(&data));
		if data.a != 17 {
			t.Errorf("int a = %v not 17", data.a)
		}
	}

	// int16
	{
		var data struct { a int16 };
		instr := &decInstr{ decInt16, 6, 0, 0 };
		state := newdecodeState(signedResult);
		execDec("int16", instr, state, t, unsafe.Pointer(&data));
		if data.a != 17 {
			t.Errorf("int a = %v not 17", data.a)
		}
	}

	// uint16
	{
		var data struct { a uint16 };
		instr := &decInstr{ decUint16, 6, 0, 0 };
		state := newdecodeState(unsignedResult);
		execDec("uint16", instr, state, t, unsafe.Pointer(&data));
		if data.a != 17 {
			t.Errorf("int a = %v not 17", data.a)
		}
	}

	// int32
	{
		var data struct { a int32 };
		instr := &decInstr{ decInt32, 6, 0, 0 };
		state := newdecodeState(signedResult);
		execDec("int32", instr, state, t, unsafe.Pointer(&data));
		if data.a != 17 {
			t.Errorf("int a = %v not 17", data.a)
		}
	}

	// uint32
	{
		var data struct { a uint32 };
		instr := &decInstr{ decUint32, 6, 0, 0 };
		state := newdecodeState(unsignedResult);
		execDec("uint32", instr, state, t, unsafe.Pointer(&data));
		if data.a != 17 {
			t.Errorf("int a = %v not 17", data.a)
		}
	}

	// int64
	{
		var data struct { a int64 };
		instr := &decInstr{ decInt64, 6, 0, 0 };
		state := newdecodeState(signedResult);
		execDec("int64", instr, state, t, unsafe.Pointer(&data));
		if data.a != 17 {
			t.Errorf("int a = %v not 17", data.a)
		}
	}

	// uint64
	{
		var data struct { a uint64 };
		instr := &decInstr{ decUint64, 6, 0, 0 };
		state := newdecodeState(unsignedResult);
		execDec("uint64", instr, state, t, unsafe.Pointer(&data));
		if data.a != 17 {
			t.Errorf("int a = %v not 17", data.a)
		}
	}

	// float
	{
		var data struct { a float };
		instr := &decInstr{ decFloat, 6, 0, 0 };
		state := newdecodeState(floatResult);
		execDec("float", instr, state, t, unsafe.Pointer(&data));
		if data.a != 17 {
			t.Errorf("int a = %v not 17", data.a)
		}
	}

	// float32
	{
		var data struct { a float32 };
		instr := &decInstr{ decFloat32, 6, 0, 0 };
		state := newdecodeState(floatResult);
		execDec("float32", instr, state, t, unsafe.Pointer(&data));
		if data.a != 17 {
			t.Errorf("int a = %v not 17", data.a)
		}
	}

	// float64
	{
		var data struct { a float64 };
		instr := &decInstr{ decFloat64, 6, 0, 0 };
		state := newdecodeState(floatResult);
		execDec("float64", instr, state, t, unsafe.Pointer(&data));
		if data.a != 17 {
			t.Errorf("int a = %v not 17", data.a)
		}
	}

	// bytes == []uint8
	{
		var data struct { a []byte };
		instr := &decInstr{ decUint8Array, 6, 0, 0 };
		state := newdecodeState(bytesResult);
		execDec("bytes", instr, state, t, unsafe.Pointer(&data));
		if string(data.a) != "hello" {
			t.Errorf(`bytes a = %q not "hello"`, string(data.a))
		}
	}

	// string
	{
		var data struct { a string };
		instr := &decInstr{ decString, 6, 0, 0 };
		state := newdecodeState(bytesResult);
		execDec("bytes", instr, state, t, unsafe.Pointer(&data));
		if data.a != "hello" {
			t.Errorf(`bytes a = %q not "hello"`, data.a)
		}
	}
}

func TestEndToEnd(t *testing.T) {
	type T2 struct {
		t string
	}
	s1 := "string1";
	s2 := "string2";
	type T1 struct {
		a, b,c int;
		n *[3]float;
		strs *[2]string;
		int64s *[]int64;
		s string;
		y []byte;
		t *T2;
	}
	t1 := &T1{
		a: 17,
		b: 18,
		c: -5,
		n: &[3]float{1.5, 2.5, 3.5},
		strs: &[2]string{s1, s2},
		int64s: &[]int64{77, 89, 123412342134},
		s: "Now is the time",
		y: strings.Bytes("hello, sailor"),
		t: &T2{"this is T2"},
	};
	b := new(bytes.Buffer);
	encode(b, t1);
	var _t1 T1;
	decode(b, getTypeInfo(reflect.Typeof(_t1)).typeId, &_t1);
	if !reflect.DeepEqual(t1, &_t1) {
		t.Errorf("encode expected %v got %v", *t1, _t1);
	}
}

func TestNesting(t *testing.T) {
	type RT struct {
		a string;
		next *RT
	}
	rt := new(RT);
	rt.a = "level1";
	rt.next = new(RT);
	rt.next.a = "level2";
	b := new(bytes.Buffer);
	encode(b, rt);
	var drt RT;
	decode(b, getTypeInfo(reflect.Typeof(drt)).typeId, &drt);
	if drt.a != rt.a {
		t.Errorf("nesting: encode expected %v got %v", *rt, drt);
	}
	if drt.next == nil {
		t.Errorf("nesting: recursion failed");
	}
	if drt.next.a != rt.next.a {
		t.Errorf("nesting: encode expected %v got %v", *rt.next, *drt.next);
	}
}

// These three structures have the same data with different indirections
type T0 struct {
	a int;
	b int;
	c int;
	d int;
}
type T1 struct {
	a int;
	b *int;
	c **int;
	d ***int;
}
type T2 struct {
	a ***int;
	b **int;
	c *int;
	d int;
}

func TestAutoIndirection(t *testing.T) {
	// First transfer t1 into t0
	var t1 T1;
	t1.a = 17;
	t1.b = new(int); *t1.b = 177;
	t1.c = new(*int); *t1.c = new(int); **t1.c = 1777;
	t1.d = new(**int); *t1.d = new(*int); **t1.d = new(int); ***t1.d = 17777;
	b := new(bytes.Buffer);
	encode(b, t1);
	var t0 T0;
	t0Id := getTypeInfo(reflect.Typeof(t0)).typeId;
	decode(b, t0Id, &t0);
	if t0.a != 17 || t0.b != 177 || t0.c != 1777 || t0.d != 17777 {
		t.Errorf("t1->t0: expected {17 177 1777 17777}; got %v", t0);
	}

	// Now transfer t2 into t0
	var t2 T2;
	t2.d = 17777;
	t2.c = new(int); *t2.c = 1777;
	t2.b = new(*int); *t2.b = new(int); **t2.b = 177;
	t2.a = new(**int); *t2.a = new(*int); **t2.a = new(int); ***t2.a = 17;
	b.Reset();
	encode(b, t2);
	t0 = T0{};
	decode(b, t0Id, &t0);
	if t0.a != 17 || t0.b != 177 || t0.c != 1777 || t0.d != 17777 {
		t.Errorf("t2->t0 expected {17 177 1777 17777}; got %v", t0);
	}

	// Now transfer t0 into t1
	t0 = T0{17, 177, 1777, 17777};
	b.Reset();
	encode(b, t0);
	t1 = T1{};
	t1Id := getTypeInfo(reflect.Typeof(t1)).typeId;
	decode(b, t1Id, &t1);
	if t1.a != 17 || *t1.b != 177 || **t1.c != 1777 || ***t1.d != 17777 {
		t.Errorf("t0->t1 expected {17 177 1777 17777}; got {%d %d %d %d}", t1.a, *t1.b, **t1.c, ***t1.d);
	}

	// Now transfer t0 into t2
	b.Reset();
	encode(b, t0);
	t2 = T2{};
	t2Id := getTypeInfo(reflect.Typeof(t2)).typeId;
	decode(b, t2Id, &t2);
	if ***t2.a != 17 || **t2.b != 177 || *t2.c != 1777 || t2.d != 17777 {
		t.Errorf("t0->t2 expected {17 177 1777 17777}; got {%d %d %d %d}", ***t2.a, **t2.b, *t2.c, t2.d);
	}

	// Now do t2 again but without pre-allocated pointers.
	b.Reset();
	encode(b, t0);
	***t2.a = 0;
	**t2.b = 0;
	*t2.c = 0;
	t2.d = 0;
	decode(b, t2Id, &t2);
	if ***t2.a != 17 || **t2.b != 177 || *t2.c != 1777 || t2.d != 17777 {
		t.Errorf("t0->t2 expected {17 177 1777 17777}; got {%d %d %d %d}", ***t2.a, **t2.b, *t2.c, t2.d);
	}
}

type RT0 struct {
	a int;
	b string;
	c float;
}
type RT1 struct {
	c float;
	b string;
	a int;
	notSet string;
}

func TestReorderedFields(t *testing.T) {
	var rt0 RT0;
	rt0.a = 17;
	rt0.b = "hello";
	rt0.c = 3.14159;
	b := new(bytes.Buffer);
	encode(b, rt0);
	rt0Id := getTypeInfo(reflect.Typeof(rt0)).typeId;
	var rt1 RT1;
	// Wire type is RT0, local type is RT1.
	decode(b, rt0Id, &rt1);
	if rt0.a != rt1.a || rt0.b != rt1.b || rt0.c != rt1.c {
		t.Errorf("rt1->rt0: expected %v; got %v", rt0, rt1);
	}
}

// Like an RT0 but with fields we'll ignore on the decode side.
type IT0 struct {
	a int64;
	b string;
	ignore_d []int;
	ignore_e [3]float;
	ignore_f bool;
	ignore_g string;
	ignore_h []byte;
	ignore_i *RT1;
	c float;
}

func TestIgnoredFields(t *testing.T) {
	var it0 IT0;
	it0.a = 17;
	it0.b = "hello";
	it0.c = 3.14159;
	it0.ignore_d = []int{ 1, 2, 3 };
	it0.ignore_e[0]  = 1.0;
	it0.ignore_e[1]  = 2.0;
	it0.ignore_e[2]  = 3.0;
	it0.ignore_f = true;
	it0.ignore_g = "pay no attention";
	it0.ignore_h = strings.Bytes("to the curtain");
	it0.ignore_i = &RT1{ 3.1, "hi", 7, "hello" };

	b := new(bytes.Buffer);
	encode(b, it0);
	rt0Id := getTypeInfo(reflect.Typeof(it0)).typeId;
	var rt1 RT1;
	// Wire type is IT0, local type is RT1.
	err := decode(b, rt0Id, &rt1);
	if err != nil {
		t.Error("error: ", err);
	}
	if int(it0.a) != rt1.a || it0.b != rt1.b || it0.c != rt1.c {
		t.Errorf("rt1->rt0: expected %v; got %v", it0, rt1);
	}
}
