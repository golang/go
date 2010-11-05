// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gob

import (
	"bytes"
	"math"
	"os"
	"reflect"
	"strings"
	"testing"
	"unsafe"
)

// Guarantee encoding format by comparing some encodings to hand-written values
type EncodeT struct {
	x uint64
	b []byte
}

var encodeT = []EncodeT{
	{0x00, []byte{0x00}},
	{0x0F, []byte{0x0F}},
	{0xFF, []byte{0xFF, 0xFF}},
	{0xFFFF, []byte{0xFE, 0xFF, 0xFF}},
	{0xFFFFFF, []byte{0xFD, 0xFF, 0xFF, 0xFF}},
	{0xFFFFFFFF, []byte{0xFC, 0xFF, 0xFF, 0xFF, 0xFF}},
	{0xFFFFFFFFFF, []byte{0xFB, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF}},
	{0xFFFFFFFFFFFF, []byte{0xFA, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF}},
	{0xFFFFFFFFFFFFFF, []byte{0xF9, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF}},
	{0xFFFFFFFFFFFFFFFF, []byte{0xF8, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF}},
	{0x1111, []byte{0xFE, 0x11, 0x11}},
	{0x1111111111111111, []byte{0xF8, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11}},
	{0x8888888888888888, []byte{0xF8, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88}},
	{1 << 63, []byte{0xF8, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}},
}

// testError is meant to be used as a deferred function to turn a panic(gobError) into a
// plain test.Error call.
func testError(t *testing.T) {
	if e := recover(); e != nil {
		t.Error(e.(gobError).Error) // Will re-panic if not one of our errors, such as a runtime error.
	}
	return
}

// Test basic encode/decode routines for unsigned integers
func TestUintCodec(t *testing.T) {
	defer testError(t)
	b := new(bytes.Buffer)
	encState := newEncoderState(nil, b)
	for _, tt := range encodeT {
		b.Reset()
		encodeUint(encState, tt.x)
		if !bytes.Equal(tt.b, b.Bytes()) {
			t.Errorf("encodeUint: %#x encode: expected % x got % x", tt.x, tt.b, b.Bytes())
		}
	}
	decState := newDecodeState(nil, &b)
	for u := uint64(0); ; u = (u + 1) * 7 {
		b.Reset()
		encodeUint(encState, u)
		v := decodeUint(decState)
		if u != v {
			t.Errorf("Encode/Decode: sent %#x received %#x", u, v)
		}
		if u&(1<<63) != 0 {
			break
		}
	}
}

func verifyInt(i int64, t *testing.T) {
	defer testError(t)
	var b = new(bytes.Buffer)
	encState := newEncoderState(nil, b)
	encodeInt(encState, i)
	decState := newDecodeState(nil, &b)
	decState.buf = make([]byte, 8)
	j := decodeInt(decState)
	if i != j {
		t.Errorf("Encode/Decode: sent %#x received %#x", uint64(i), uint64(j))
	}
}

// Test basic encode/decode routines for signed integers
func TestIntCodec(t *testing.T) {
	for u := uint64(0); ; u = (u + 1) * 7 {
		// Do positive and negative values
		i := int64(u)
		verifyInt(i, t)
		verifyInt(-i, t)
		verifyInt(^i, t)
		if u&(1<<63) != 0 {
			break
		}
	}
	verifyInt(-1<<63, t) // a tricky case
}

// The result of encoding a true boolean with field number 7
var boolResult = []byte{0x07, 0x01}
// The result of encoding a number 17 with field number 7
var signedResult = []byte{0x07, 2 * 17}
var unsignedResult = []byte{0x07, 17}
var floatResult = []byte{0x07, 0xFE, 0x31, 0x40}
// The result of encoding a number 17+19i with field number 7
var complexResult = []byte{0x07, 0xFE, 0x31, 0x40, 0xFE, 0x33, 0x40}
// The result of encoding "hello" with field number 7
var bytesResult = []byte{0x07, 0x05, 'h', 'e', 'l', 'l', 'o'}

func newencoderState(b *bytes.Buffer) *encoderState {
	b.Reset()
	state := newEncoderState(nil, b)
	state.fieldnum = -1
	return state
}

// Test instruction execution for encoding.
// Do not run the machine yet; instead do individual instructions crafted by hand.
func TestScalarEncInstructions(t *testing.T) {
	var b = new(bytes.Buffer)

	// bool
	{
		data := struct{ a bool }{true}
		instr := &encInstr{encBool, 6, 0, 0}
		state := newencoderState(b)
		instr.op(instr, state, unsafe.Pointer(&data))
		if !bytes.Equal(boolResult, b.Bytes()) {
			t.Errorf("bool enc instructions: expected % x got % x", boolResult, b.Bytes())
		}
	}

	// int
	{
		b.Reset()
		data := struct{ a int }{17}
		instr := &encInstr{encInt, 6, 0, 0}
		state := newencoderState(b)
		instr.op(instr, state, unsafe.Pointer(&data))
		if !bytes.Equal(signedResult, b.Bytes()) {
			t.Errorf("int enc instructions: expected % x got % x", signedResult, b.Bytes())
		}
	}

	// uint
	{
		b.Reset()
		data := struct{ a uint }{17}
		instr := &encInstr{encUint, 6, 0, 0}
		state := newencoderState(b)
		instr.op(instr, state, unsafe.Pointer(&data))
		if !bytes.Equal(unsignedResult, b.Bytes()) {
			t.Errorf("uint enc instructions: expected % x got % x", unsignedResult, b.Bytes())
		}
	}

	// int8
	{
		b.Reset()
		data := struct{ a int8 }{17}
		instr := &encInstr{encInt8, 6, 0, 0}
		state := newencoderState(b)
		instr.op(instr, state, unsafe.Pointer(&data))
		if !bytes.Equal(signedResult, b.Bytes()) {
			t.Errorf("int8 enc instructions: expected % x got % x", signedResult, b.Bytes())
		}
	}

	// uint8
	{
		b.Reset()
		data := struct{ a uint8 }{17}
		instr := &encInstr{encUint8, 6, 0, 0}
		state := newencoderState(b)
		instr.op(instr, state, unsafe.Pointer(&data))
		if !bytes.Equal(unsignedResult, b.Bytes()) {
			t.Errorf("uint8 enc instructions: expected % x got % x", unsignedResult, b.Bytes())
		}
	}

	// int16
	{
		b.Reset()
		data := struct{ a int16 }{17}
		instr := &encInstr{encInt16, 6, 0, 0}
		state := newencoderState(b)
		instr.op(instr, state, unsafe.Pointer(&data))
		if !bytes.Equal(signedResult, b.Bytes()) {
			t.Errorf("int16 enc instructions: expected % x got % x", signedResult, b.Bytes())
		}
	}

	// uint16
	{
		b.Reset()
		data := struct{ a uint16 }{17}
		instr := &encInstr{encUint16, 6, 0, 0}
		state := newencoderState(b)
		instr.op(instr, state, unsafe.Pointer(&data))
		if !bytes.Equal(unsignedResult, b.Bytes()) {
			t.Errorf("uint16 enc instructions: expected % x got % x", unsignedResult, b.Bytes())
		}
	}

	// int32
	{
		b.Reset()
		data := struct{ a int32 }{17}
		instr := &encInstr{encInt32, 6, 0, 0}
		state := newencoderState(b)
		instr.op(instr, state, unsafe.Pointer(&data))
		if !bytes.Equal(signedResult, b.Bytes()) {
			t.Errorf("int32 enc instructions: expected % x got % x", signedResult, b.Bytes())
		}
	}

	// uint32
	{
		b.Reset()
		data := struct{ a uint32 }{17}
		instr := &encInstr{encUint32, 6, 0, 0}
		state := newencoderState(b)
		instr.op(instr, state, unsafe.Pointer(&data))
		if !bytes.Equal(unsignedResult, b.Bytes()) {
			t.Errorf("uint32 enc instructions: expected % x got % x", unsignedResult, b.Bytes())
		}
	}

	// int64
	{
		b.Reset()
		data := struct{ a int64 }{17}
		instr := &encInstr{encInt64, 6, 0, 0}
		state := newencoderState(b)
		instr.op(instr, state, unsafe.Pointer(&data))
		if !bytes.Equal(signedResult, b.Bytes()) {
			t.Errorf("int64 enc instructions: expected % x got % x", signedResult, b.Bytes())
		}
	}

	// uint64
	{
		b.Reset()
		data := struct{ a uint64 }{17}
		instr := &encInstr{encUint64, 6, 0, 0}
		state := newencoderState(b)
		instr.op(instr, state, unsafe.Pointer(&data))
		if !bytes.Equal(unsignedResult, b.Bytes()) {
			t.Errorf("uint64 enc instructions: expected % x got % x", unsignedResult, b.Bytes())
		}
	}

	// float
	{
		b.Reset()
		data := struct{ a float }{17}
		instr := &encInstr{encFloat, 6, 0, 0}
		state := newencoderState(b)
		instr.op(instr, state, unsafe.Pointer(&data))
		if !bytes.Equal(floatResult, b.Bytes()) {
			t.Errorf("float enc instructions: expected % x got % x", floatResult, b.Bytes())
		}
	}

	// float32
	{
		b.Reset()
		data := struct{ a float32 }{17}
		instr := &encInstr{encFloat32, 6, 0, 0}
		state := newencoderState(b)
		instr.op(instr, state, unsafe.Pointer(&data))
		if !bytes.Equal(floatResult, b.Bytes()) {
			t.Errorf("float32 enc instructions: expected % x got % x", floatResult, b.Bytes())
		}
	}

	// float64
	{
		b.Reset()
		data := struct{ a float64 }{17}
		instr := &encInstr{encFloat64, 6, 0, 0}
		state := newencoderState(b)
		instr.op(instr, state, unsafe.Pointer(&data))
		if !bytes.Equal(floatResult, b.Bytes()) {
			t.Errorf("float64 enc instructions: expected % x got % x", floatResult, b.Bytes())
		}
	}

	// bytes == []uint8
	{
		b.Reset()
		data := struct{ a []byte }{[]byte("hello")}
		instr := &encInstr{encUint8Array, 6, 0, 0}
		state := newencoderState(b)
		instr.op(instr, state, unsafe.Pointer(&data))
		if !bytes.Equal(bytesResult, b.Bytes()) {
			t.Errorf("bytes enc instructions: expected % x got % x", bytesResult, b.Bytes())
		}
	}

	// string
	{
		b.Reset()
		data := struct{ a string }{"hello"}
		instr := &encInstr{encString, 6, 0, 0}
		state := newencoderState(b)
		instr.op(instr, state, unsafe.Pointer(&data))
		if !bytes.Equal(bytesResult, b.Bytes()) {
			t.Errorf("string enc instructions: expected % x got % x", bytesResult, b.Bytes())
		}
	}
}

func execDec(typ string, instr *decInstr, state *decodeState, t *testing.T, p unsafe.Pointer) {
	defer testError(t)
	v := int(decodeUint(state))
	if v+state.fieldnum != 6 {
		t.Fatalf("decoding field number %d, got %d", 6, v+state.fieldnum)
	}
	instr.op(instr, state, decIndirect(p, instr.indir))
	state.fieldnum = 6
}

func newDecodeStateFromData(data []byte) *decodeState {
	b := bytes.NewBuffer(data)
	state := newDecodeState(nil, &b)
	state.fieldnum = -1
	return state
}

// Test instruction execution for decoding.
// Do not run the machine yet; instead do individual instructions crafted by hand.
func TestScalarDecInstructions(t *testing.T) {
	ovfl := os.ErrorString("overflow")

	// bool
	{
		var data struct {
			a bool
		}
		instr := &decInstr{decBool, 6, 0, 0, ovfl}
		state := newDecodeStateFromData(boolResult)
		execDec("bool", instr, state, t, unsafe.Pointer(&data))
		if data.a != true {
			t.Errorf("bool a = %v not true", data.a)
		}
	}
	// int
	{
		var data struct {
			a int
		}
		instr := &decInstr{decOpMap[reflect.Int], 6, 0, 0, ovfl}
		state := newDecodeStateFromData(signedResult)
		execDec("int", instr, state, t, unsafe.Pointer(&data))
		if data.a != 17 {
			t.Errorf("int a = %v not 17", data.a)
		}
	}

	// uint
	{
		var data struct {
			a uint
		}
		instr := &decInstr{decOpMap[reflect.Uint], 6, 0, 0, ovfl}
		state := newDecodeStateFromData(unsignedResult)
		execDec("uint", instr, state, t, unsafe.Pointer(&data))
		if data.a != 17 {
			t.Errorf("uint a = %v not 17", data.a)
		}
	}

	// int8
	{
		var data struct {
			a int8
		}
		instr := &decInstr{decInt8, 6, 0, 0, ovfl}
		state := newDecodeStateFromData(signedResult)
		execDec("int8", instr, state, t, unsafe.Pointer(&data))
		if data.a != 17 {
			t.Errorf("int8 a = %v not 17", data.a)
		}
	}

	// uint8
	{
		var data struct {
			a uint8
		}
		instr := &decInstr{decUint8, 6, 0, 0, ovfl}
		state := newDecodeStateFromData(unsignedResult)
		execDec("uint8", instr, state, t, unsafe.Pointer(&data))
		if data.a != 17 {
			t.Errorf("uint8 a = %v not 17", data.a)
		}
	}

	// int16
	{
		var data struct {
			a int16
		}
		instr := &decInstr{decInt16, 6, 0, 0, ovfl}
		state := newDecodeStateFromData(signedResult)
		execDec("int16", instr, state, t, unsafe.Pointer(&data))
		if data.a != 17 {
			t.Errorf("int16 a = %v not 17", data.a)
		}
	}

	// uint16
	{
		var data struct {
			a uint16
		}
		instr := &decInstr{decUint16, 6, 0, 0, ovfl}
		state := newDecodeStateFromData(unsignedResult)
		execDec("uint16", instr, state, t, unsafe.Pointer(&data))
		if data.a != 17 {
			t.Errorf("uint16 a = %v not 17", data.a)
		}
	}

	// int32
	{
		var data struct {
			a int32
		}
		instr := &decInstr{decInt32, 6, 0, 0, ovfl}
		state := newDecodeStateFromData(signedResult)
		execDec("int32", instr, state, t, unsafe.Pointer(&data))
		if data.a != 17 {
			t.Errorf("int32 a = %v not 17", data.a)
		}
	}

	// uint32
	{
		var data struct {
			a uint32
		}
		instr := &decInstr{decUint32, 6, 0, 0, ovfl}
		state := newDecodeStateFromData(unsignedResult)
		execDec("uint32", instr, state, t, unsafe.Pointer(&data))
		if data.a != 17 {
			t.Errorf("uint32 a = %v not 17", data.a)
		}
	}

	// uintptr
	{
		var data struct {
			a uintptr
		}
		instr := &decInstr{decOpMap[reflect.Uintptr], 6, 0, 0, ovfl}
		state := newDecodeStateFromData(unsignedResult)
		execDec("uintptr", instr, state, t, unsafe.Pointer(&data))
		if data.a != 17 {
			t.Errorf("uintptr a = %v not 17", data.a)
		}
	}

	// int64
	{
		var data struct {
			a int64
		}
		instr := &decInstr{decInt64, 6, 0, 0, ovfl}
		state := newDecodeStateFromData(signedResult)
		execDec("int64", instr, state, t, unsafe.Pointer(&data))
		if data.a != 17 {
			t.Errorf("int64 a = %v not 17", data.a)
		}
	}

	// uint64
	{
		var data struct {
			a uint64
		}
		instr := &decInstr{decUint64, 6, 0, 0, ovfl}
		state := newDecodeStateFromData(unsignedResult)
		execDec("uint64", instr, state, t, unsafe.Pointer(&data))
		if data.a != 17 {
			t.Errorf("uint64 a = %v not 17", data.a)
		}
	}

	// float
	{
		var data struct {
			a float
		}
		instr := &decInstr{decOpMap[reflect.Float], 6, 0, 0, ovfl}
		state := newDecodeStateFromData(floatResult)
		execDec("float", instr, state, t, unsafe.Pointer(&data))
		if data.a != 17 {
			t.Errorf("float a = %v not 17", data.a)
		}
	}

	// float32
	{
		var data struct {
			a float32
		}
		instr := &decInstr{decFloat32, 6, 0, 0, ovfl}
		state := newDecodeStateFromData(floatResult)
		execDec("float32", instr, state, t, unsafe.Pointer(&data))
		if data.a != 17 {
			t.Errorf("float32 a = %v not 17", data.a)
		}
	}

	// float64
	{
		var data struct {
			a float64
		}
		instr := &decInstr{decFloat64, 6, 0, 0, ovfl}
		state := newDecodeStateFromData(floatResult)
		execDec("float64", instr, state, t, unsafe.Pointer(&data))
		if data.a != 17 {
			t.Errorf("float64 a = %v not 17", data.a)
		}
	}

	// complex
	{
		var data struct {
			a complex
		}
		instr := &decInstr{decOpMap[reflect.Complex], 6, 0, 0, ovfl}
		state := newDecodeStateFromData(complexResult)
		execDec("complex", instr, state, t, unsafe.Pointer(&data))
		if data.a != 17+19i {
			t.Errorf("complex a = %v not 17+19i", data.a)
		}
	}

	// complex64
	{
		var data struct {
			a complex64
		}
		instr := &decInstr{decOpMap[reflect.Complex64], 6, 0, 0, ovfl}
		state := newDecodeStateFromData(complexResult)
		execDec("complex", instr, state, t, unsafe.Pointer(&data))
		if data.a != 17+19i {
			t.Errorf("complex a = %v not 17+19i", data.a)
		}
	}

	// complex128
	{
		var data struct {
			a complex128
		}
		instr := &decInstr{decOpMap[reflect.Complex128], 6, 0, 0, ovfl}
		state := newDecodeStateFromData(complexResult)
		execDec("complex", instr, state, t, unsafe.Pointer(&data))
		if data.a != 17+19i {
			t.Errorf("complex a = %v not 17+19i", data.a)
		}
	}

	// bytes == []uint8
	{
		var data struct {
			a []byte
		}
		instr := &decInstr{decUint8Array, 6, 0, 0, ovfl}
		state := newDecodeStateFromData(bytesResult)
		execDec("bytes", instr, state, t, unsafe.Pointer(&data))
		if string(data.a) != "hello" {
			t.Errorf(`bytes a = %q not "hello"`, string(data.a))
		}
	}

	// string
	{
		var data struct {
			a string
		}
		instr := &decInstr{decString, 6, 0, 0, ovfl}
		state := newDecodeStateFromData(bytesResult)
		execDec("bytes", instr, state, t, unsafe.Pointer(&data))
		if data.a != "hello" {
			t.Errorf(`bytes a = %q not "hello"`, data.a)
		}
	}
}

func TestEndToEnd(t *testing.T) {
	type T2 struct {
		t string
	}
	s1 := "string1"
	s2 := "string2"
	type T1 struct {
		a, b, c int
		m       map[string]*float
		n       *[3]float
		strs    *[2]string
		int64s  *[]int64
		ri      complex64
		s       string
		y       []byte
		t       *T2
	}
	pi := 3.14159
	e := 2.71828
	t1 := &T1{
		a:      17,
		b:      18,
		c:      -5,
		m:      map[string]*float{"pi": &pi, "e": &e},
		n:      &[3]float{1.5, 2.5, 3.5},
		strs:   &[2]string{s1, s2},
		int64s: &[]int64{77, 89, 123412342134},
		ri:     17 - 23i,
		s:      "Now is the time",
		y:      []byte("hello, sailor"),
		t:      &T2{"this is T2"},
	}
	b := new(bytes.Buffer)
	err := NewEncoder(b).Encode(t1)
	if err != nil {
		t.Error("encode:", err)
	}
	var _t1 T1
	err = NewDecoder(b).Decode(&_t1)
	if err != nil {
		t.Fatal("decode:", err)
	}
	if !reflect.DeepEqual(t1, &_t1) {
		t.Errorf("encode expected %v got %v", *t1, _t1)
	}
}

func TestOverflow(t *testing.T) {
	type inputT struct {
		maxi int64
		mini int64
		maxu uint64
		maxf float64
		minf float64
		maxc complex128
		minc complex128
	}
	var it inputT
	var err os.Error
	b := new(bytes.Buffer)
	enc := NewEncoder(b)
	dec := NewDecoder(b)

	// int8
	b.Reset()
	it = inputT{
		maxi: math.MaxInt8 + 1,
	}
	type outi8 struct {
		maxi int8
		mini int8
	}
	var o1 outi8
	enc.Encode(it)
	err = dec.Decode(&o1)
	if err == nil || err.String() != `value for "maxi" out of range` {
		t.Error("wrong overflow error for int8:", err)
	}
	it = inputT{
		mini: math.MinInt8 - 1,
	}
	b.Reset()
	enc.Encode(it)
	err = dec.Decode(&o1)
	if err == nil || err.String() != `value for "mini" out of range` {
		t.Error("wrong underflow error for int8:", err)
	}

	// int16
	b.Reset()
	it = inputT{
		maxi: math.MaxInt16 + 1,
	}
	type outi16 struct {
		maxi int16
		mini int16
	}
	var o2 outi16
	enc.Encode(it)
	err = dec.Decode(&o2)
	if err == nil || err.String() != `value for "maxi" out of range` {
		t.Error("wrong overflow error for int16:", err)
	}
	it = inputT{
		mini: math.MinInt16 - 1,
	}
	b.Reset()
	enc.Encode(it)
	err = dec.Decode(&o2)
	if err == nil || err.String() != `value for "mini" out of range` {
		t.Error("wrong underflow error for int16:", err)
	}

	// int32
	b.Reset()
	it = inputT{
		maxi: math.MaxInt32 + 1,
	}
	type outi32 struct {
		maxi int32
		mini int32
	}
	var o3 outi32
	enc.Encode(it)
	err = dec.Decode(&o3)
	if err == nil || err.String() != `value for "maxi" out of range` {
		t.Error("wrong overflow error for int32:", err)
	}
	it = inputT{
		mini: math.MinInt32 - 1,
	}
	b.Reset()
	enc.Encode(it)
	err = dec.Decode(&o3)
	if err == nil || err.String() != `value for "mini" out of range` {
		t.Error("wrong underflow error for int32:", err)
	}

	// uint8
	b.Reset()
	it = inputT{
		maxu: math.MaxUint8 + 1,
	}
	type outu8 struct {
		maxu uint8
	}
	var o4 outu8
	enc.Encode(it)
	err = dec.Decode(&o4)
	if err == nil || err.String() != `value for "maxu" out of range` {
		t.Error("wrong overflow error for uint8:", err)
	}

	// uint16
	b.Reset()
	it = inputT{
		maxu: math.MaxUint16 + 1,
	}
	type outu16 struct {
		maxu uint16
	}
	var o5 outu16
	enc.Encode(it)
	err = dec.Decode(&o5)
	if err == nil || err.String() != `value for "maxu" out of range` {
		t.Error("wrong overflow error for uint16:", err)
	}

	// uint32
	b.Reset()
	it = inputT{
		maxu: math.MaxUint32 + 1,
	}
	type outu32 struct {
		maxu uint32
	}
	var o6 outu32
	enc.Encode(it)
	err = dec.Decode(&o6)
	if err == nil || err.String() != `value for "maxu" out of range` {
		t.Error("wrong overflow error for uint32:", err)
	}

	// float32
	b.Reset()
	it = inputT{
		maxf: math.MaxFloat32 * 2,
	}
	type outf32 struct {
		maxf float32
		minf float32
	}
	var o7 outf32
	enc.Encode(it)
	err = dec.Decode(&o7)
	if err == nil || err.String() != `value for "maxf" out of range` {
		t.Error("wrong overflow error for float32:", err)
	}

	// complex64
	b.Reset()
	it = inputT{
		maxc: cmplx(math.MaxFloat32*2, math.MaxFloat32*2),
	}
	type outc64 struct {
		maxc complex64
		minc complex64
	}
	var o8 outc64
	enc.Encode(it)
	err = dec.Decode(&o8)
	if err == nil || err.String() != `value for "maxc" out of range` {
		t.Error("wrong overflow error for complex64:", err)
	}
}


func TestNesting(t *testing.T) {
	type RT struct {
		a    string
		next *RT
	}
	rt := new(RT)
	rt.a = "level1"
	rt.next = new(RT)
	rt.next.a = "level2"
	b := new(bytes.Buffer)
	NewEncoder(b).Encode(rt)
	var drt RT
	dec := NewDecoder(b)
	err := dec.Decode(&drt)
	if err != nil {
		t.Errorf("decoder error:", err)
	}
	if drt.a != rt.a {
		t.Errorf("nesting: encode expected %v got %v", *rt, drt)
	}
	if drt.next == nil {
		t.Errorf("nesting: recursion failed")
	}
	if drt.next.a != rt.next.a {
		t.Errorf("nesting: encode expected %v got %v", *rt.next, *drt.next)
	}
}

// These three structures have the same data with different indirections
type T0 struct {
	a int
	b int
	c int
	d int
}
type T1 struct {
	a int
	b *int
	c **int
	d ***int
}
type T2 struct {
	a ***int
	b **int
	c *int
	d int
}

func TestAutoIndirection(t *testing.T) {
	// First transfer t1 into t0
	var t1 T1
	t1.a = 17
	t1.b = new(int)
	*t1.b = 177
	t1.c = new(*int)
	*t1.c = new(int)
	**t1.c = 1777
	t1.d = new(**int)
	*t1.d = new(*int)
	**t1.d = new(int)
	***t1.d = 17777
	b := new(bytes.Buffer)
	enc := NewEncoder(b)
	enc.Encode(t1)
	dec := NewDecoder(b)
	var t0 T0
	dec.Decode(&t0)
	if t0.a != 17 || t0.b != 177 || t0.c != 1777 || t0.d != 17777 {
		t.Errorf("t1->t0: expected {17 177 1777 17777}; got %v", t0)
	}

	// Now transfer t2 into t0
	var t2 T2
	t2.d = 17777
	t2.c = new(int)
	*t2.c = 1777
	t2.b = new(*int)
	*t2.b = new(int)
	**t2.b = 177
	t2.a = new(**int)
	*t2.a = new(*int)
	**t2.a = new(int)
	***t2.a = 17
	b.Reset()
	enc.Encode(t2)
	t0 = T0{}
	dec.Decode(&t0)
	if t0.a != 17 || t0.b != 177 || t0.c != 1777 || t0.d != 17777 {
		t.Errorf("t2->t0 expected {17 177 1777 17777}; got %v", t0)
	}

	// Now transfer t0 into t1
	t0 = T0{17, 177, 1777, 17777}
	b.Reset()
	enc.Encode(t0)
	t1 = T1{}
	dec.Decode(&t1)
	if t1.a != 17 || *t1.b != 177 || **t1.c != 1777 || ***t1.d != 17777 {
		t.Errorf("t0->t1 expected {17 177 1777 17777}; got {%d %d %d %d}", t1.a, *t1.b, **t1.c, ***t1.d)
	}

	// Now transfer t0 into t2
	b.Reset()
	enc.Encode(t0)
	t2 = T2{}
	dec.Decode(&t2)
	if ***t2.a != 17 || **t2.b != 177 || *t2.c != 1777 || t2.d != 17777 {
		t.Errorf("t0->t2 expected {17 177 1777 17777}; got {%d %d %d %d}", ***t2.a, **t2.b, *t2.c, t2.d)
	}

	// Now do t2 again but without pre-allocated pointers.
	b.Reset()
	enc.Encode(t0)
	***t2.a = 0
	**t2.b = 0
	*t2.c = 0
	t2.d = 0
	dec.Decode(&t2)
	if ***t2.a != 17 || **t2.b != 177 || *t2.c != 1777 || t2.d != 17777 {
		t.Errorf("t0->t2 expected {17 177 1777 17777}; got {%d %d %d %d}", ***t2.a, **t2.b, *t2.c, t2.d)
	}
}

type RT0 struct {
	a int
	b string
	c float
}
type RT1 struct {
	c      float
	b      string
	a      int
	notSet string
}

func TestReorderedFields(t *testing.T) {
	var rt0 RT0
	rt0.a = 17
	rt0.b = "hello"
	rt0.c = 3.14159
	b := new(bytes.Buffer)
	NewEncoder(b).Encode(rt0)
	dec := NewDecoder(b)
	var rt1 RT1
	// Wire type is RT0, local type is RT1.
	err := dec.Decode(&rt1)
	if err != nil {
		t.Error("decode error:", err)
	}
	if rt0.a != rt1.a || rt0.b != rt1.b || rt0.c != rt1.c {
		t.Errorf("rt1->rt0: expected %v; got %v", rt0, rt1)
	}
}

// Like an RT0 but with fields we'll ignore on the decode side.
type IT0 struct {
	a        int64
	b        string
	ignore_d []int
	ignore_e [3]float
	ignore_f bool
	ignore_g string
	ignore_h []byte
	ignore_i *RT1
	ignore_m map[string]int
	c        float
}

func TestIgnoredFields(t *testing.T) {
	var it0 IT0
	it0.a = 17
	it0.b = "hello"
	it0.c = 3.14159
	it0.ignore_d = []int{1, 2, 3}
	it0.ignore_e[0] = 1.0
	it0.ignore_e[1] = 2.0
	it0.ignore_e[2] = 3.0
	it0.ignore_f = true
	it0.ignore_g = "pay no attention"
	it0.ignore_h = []byte("to the curtain")
	it0.ignore_i = &RT1{3.1, "hi", 7, "hello"}
	it0.ignore_m = map[string]int{"one": 1, "two": 2}

	b := new(bytes.Buffer)
	NewEncoder(b).Encode(it0)
	dec := NewDecoder(b)
	var rt1 RT1
	// Wire type is IT0, local type is RT1.
	err := dec.Decode(&rt1)
	if err != nil {
		t.Error("error: ", err)
	}
	if int(it0.a) != rt1.a || it0.b != rt1.b || it0.c != rt1.c {
		t.Errorf("rt1->rt0: expected %v; got %v", it0, rt1)
	}
}

type Bad0 struct {
	ch chan int
	c  float
}

var nilEncoder *Encoder

func TestInvalidField(t *testing.T) {
	var bad0 Bad0
	bad0.ch = make(chan int)
	b := new(bytes.Buffer)
	err := nilEncoder.encode(b, reflect.NewValue(&bad0))
	if err == nil {
		t.Error("expected error; got none")
	} else if strings.Index(err.String(), "type") < 0 {
		t.Error("expected type error; got", err)
	}
}

type Indirect struct {
	a ***[3]int
	s ***[]int
	m ****map[string]int
}

type Direct struct {
	a [3]int
	s []int
	m map[string]int
}

func TestIndirectSliceMapArray(t *testing.T) {
	// Marshal indirect, unmarshal to direct.
	i := new(Indirect)
	i.a = new(**[3]int)
	*i.a = new(*[3]int)
	**i.a = new([3]int)
	***i.a = [3]int{1, 2, 3}
	i.s = new(**[]int)
	*i.s = new(*[]int)
	**i.s = new([]int)
	***i.s = []int{4, 5, 6}
	i.m = new(***map[string]int)
	*i.m = new(**map[string]int)
	**i.m = new(*map[string]int)
	***i.m = new(map[string]int)
	****i.m = map[string]int{"one": 1, "two": 2, "three": 3}
	b := new(bytes.Buffer)
	NewEncoder(b).Encode(i)
	dec := NewDecoder(b)
	var d Direct
	err := dec.Decode(&d)
	if err != nil {
		t.Error("error: ", err)
	}
	if len(d.a) != 3 || d.a[0] != 1 || d.a[1] != 2 || d.a[2] != 3 {
		t.Errorf("indirect to direct: d.a is %v not %v", d.a, ***i.a)
	}
	if len(d.s) != 3 || d.s[0] != 4 || d.s[1] != 5 || d.s[2] != 6 {
		t.Errorf("indirect to direct: d.s is %v not %v", d.s, ***i.s)
	}
	if len(d.m) != 3 || d.m["one"] != 1 || d.m["two"] != 2 || d.m["three"] != 3 {
		t.Errorf("indirect to direct: d.m is %v not %v", d.m, ***i.m)
	}
	// Marshal direct, unmarshal to indirect.
	d.a = [3]int{11, 22, 33}
	d.s = []int{44, 55, 66}
	d.m = map[string]int{"four": 4, "five": 5, "six": 6}
	i = new(Indirect)
	b.Reset()
	NewEncoder(b).Encode(d)
	dec = NewDecoder(b)
	err = dec.Decode(&i)
	if err != nil {
		t.Error("error: ", err)
	}
	if len(***i.a) != 3 || (***i.a)[0] != 11 || (***i.a)[1] != 22 || (***i.a)[2] != 33 {
		t.Errorf("direct to indirect: ***i.a is %v not %v", ***i.a, d.a)
	}
	if len(***i.s) != 3 || (***i.s)[0] != 44 || (***i.s)[1] != 55 || (***i.s)[2] != 66 {
		t.Errorf("direct to indirect: ***i.s is %v not %v", ***i.s, ***i.s)
	}
	if len(****i.m) != 3 || (****i.m)["four"] != 4 || (****i.m)["five"] != 5 || (****i.m)["six"] != 6 {
		t.Errorf("direct to indirect: ****i.m is %v not %v", ****i.m, d.m)
	}
}

// An interface with several implementations
type Squarer interface {
	Square() int
}

type Int int

func (i Int) Square() int {
	return int(i * i)
}

type Float float

func (f Float) Square() int {
	return int(f * f)
}

type Vector []int

func (v Vector) Square() int {
	sum := 0
	for _, x := range v {
		sum += x * x
	}
	return sum
}

type Point struct {
	a, b int
}

func (p Point) Square() int {
	return p.a*p.a + p.b*p.b
}

// A struct with interfaces in it.
type InterfaceItem struct {
	i             int
	sq1, sq2, sq3 Squarer
	f             float
	sq            []Squarer
}

// The same struct without interfaces
type NoInterfaceItem struct {
	i int
	f float
}

func TestInterface(t *testing.T) {
	iVal := Int(3)
	fVal := Float(5)
	// Sending a Vector will require that the receiver define a type in the middle of
	// receiving the value for item2.
	vVal := Vector{1, 2, 3}
	b := new(bytes.Buffer)
	item1 := &InterfaceItem{1, iVal, fVal, vVal, 11.5, []Squarer{iVal, fVal, nil, vVal}}
	// Register the types.
	Register(Int(0))
	Register(Float(0))
	Register(Vector{})
	err := NewEncoder(b).Encode(item1)
	if err != nil {
		t.Error("expected no encode error; got", err)
	}

	item2 := InterfaceItem{}
	err = NewDecoder(b).Decode(&item2)
	if err != nil {
		t.Fatal("decode:", err)
	}
	if item2.i != item1.i {
		t.Error("normal int did not decode correctly")
	}
	if item2.sq1 == nil || item2.sq1.Square() != iVal.Square() {
		t.Error("Int did not decode correctly")
	}
	if item2.sq2 == nil || item2.sq2.Square() != fVal.Square() {
		t.Error("Float did not decode correctly")
	}
	if item2.sq3 == nil || item2.sq3.Square() != vVal.Square() {
		t.Error("Vector did not decode correctly")
	}
	if item2.f != item1.f {
		t.Error("normal float did not decode correctly")
	}
	// Now check that we received a slice of Squarers correctly, including a nil element
	if len(item1.sq) != len(item2.sq) {
		t.Fatalf("[]Squarer length wrong: got %d; expected %d", len(item2.sq), len(item1.sq))
	}
	for i, v1 := range item1.sq {
		v2 := item2.sq[i]
		if v1 == nil || v2 == nil {
			if v1 != nil || v2 != nil {
				t.Errorf("item %d inconsistent nils", i)
			}
			continue
			if v1.Square() != v2.Square() {
				t.Errorf("item %d inconsistent values: %v %v", v1, v2)
			}
		}
	}

}

// A struct with all basic types, stored in interfaces.
type BasicInterfaceItem struct {
	Int, Int8, Int16, Int32, Int64      interface{}
	Uint, Uint8, Uint16, Uint32, Uint64 interface{}
	Float, Float32, Float64             interface{}
	Complex, Complex64, Complex128      interface{}
	Bool                                interface{}
	String                              interface{}
	Bytes                               interface{}
}

func TestInterfaceBasic(t *testing.T) {
	b := new(bytes.Buffer)
	item1 := &BasicInterfaceItem{
		int(1), int8(1), int16(1), int32(1), int64(1),
		uint(1), uint8(1), uint16(1), uint32(1), uint64(1),
		float(1), float32(1), float64(1),
		complex(0i), complex64(0i), complex128(0i),
		true,
		"hello",
		[]byte("sailor"),
	}
	err := NewEncoder(b).Encode(item1)
	if err != nil {
		t.Error("expected no encode error; got", err)
	}

	item2 := &BasicInterfaceItem{}
	err = NewDecoder(b).Decode(&item2)
	if err != nil {
		t.Fatal("decode:", err)
	}
	if !reflect.DeepEqual(item1, item2) {
		t.Errorf("encode expected %v got %v", item1, item2)
	}
	// Hand check a couple for correct types.
	if v, ok := item2.Bool.(bool); !ok || !v {
		t.Error("boolean should be true")
	}
	if v, ok := item2.String.(string); !ok || v != item1.String.(string) {
		t.Errorf("string should be %v is %v", item1.String, v)
	}
}

type String string

type PtrInterfaceItem struct {
	str interface{} // basic
	Str interface{} // derived
}

// We'll send pointers; should receive values.
// Also check that we can register T but send *T.
func TestInterfacePointer(t *testing.T) {
	b := new(bytes.Buffer)
	str1 := "howdy"
	str2 := String("kiddo")
	item1 := &PtrInterfaceItem{
		&str1,
		&str2,
	}
	// Register the type.
	Register(str2)
	err := NewEncoder(b).Encode(item1)
	if err != nil {
		t.Error("expected no encode error; got", err)
	}

	item2 := &PtrInterfaceItem{}
	err = NewDecoder(b).Decode(&item2)
	if err != nil {
		t.Fatal("decode:", err)
	}
	// Hand test for correct types and values.
	if v, ok := item2.str.(string); !ok || v != str1 {
		t.Errorf("basic string failed: %q should be %q", v, str1)
	}
	if v, ok := item2.Str.(String); !ok || v != str2 {
		t.Errorf("derived type String failed: %q should be %q", v, str2)
	}
}

func TestIgnoreInterface(t *testing.T) {
	iVal := Int(3)
	fVal := Float(5)
	// Sending a Point will require that the receiver define a type in the middle of
	// receiving the value for item2.
	pVal := Point{2, 3}
	b := new(bytes.Buffer)
	item1 := &InterfaceItem{1, iVal, fVal, pVal, 11.5, nil}
	// Register the types.
	Register(Int(0))
	Register(Float(0))
	Register(Point{})
	err := NewEncoder(b).Encode(item1)
	if err != nil {
		t.Error("expected no encode error; got", err)
	}

	item2 := NoInterfaceItem{}
	err = NewDecoder(b).Decode(&item2)
	if err != nil {
		t.Fatal("decode:", err)
	}
	if item2.i != item1.i {
		t.Error("normal int did not decode correctly")
	}
	if item2.f != item2.f {
		t.Error("normal float did not decode correctly")
	}
}

// A type that won't be defined in the gob until we send it in an interface value.
type OnTheFly struct {
	a int
}

type DT struct {
	//	X OnTheFly
	a     int
	b     string
	c     float
	i     interface{}
	j     interface{}
	i_nil interface{}
	m     map[string]int
	r     [3]int
	s     []string
}

func TestDebug(t *testing.T) {
	if debugFunc == nil {
		return
	}
	Register(OnTheFly{})
	var dt DT
	dt.a = 17
	dt.b = "hello"
	dt.c = 3.14159
	dt.i = 271828
	dt.j = OnTheFly{3}
	dt.i_nil = nil
	dt.m = map[string]int{"one": 1, "two": 2}
	dt.r = [3]int{11, 22, 33}
	dt.s = []string{"hi", "joe"}
	b := new(bytes.Buffer)
	err := NewEncoder(b).Encode(dt)
	if err != nil {
		t.Fatal("encode:", err)
	}
	debugBuffer := bytes.NewBuffer(b.Bytes())
	dt2 := &DT{}
	err = NewDecoder(b).Decode(&dt2)
	if err != nil {
		t.Error("decode:", err)
	}
	debugFunc(debugBuffer)
}
