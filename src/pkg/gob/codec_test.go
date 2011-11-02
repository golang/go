// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gob

import (
	"bytes"
	"errors"
	"math"
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
		t.Error(e.(gobError).err) // Will re-panic if not one of our errors, such as a runtime error.
	}
	return
}

// Test basic encode/decode routines for unsigned integers
func TestUintCodec(t *testing.T) {
	defer testError(t)
	b := new(bytes.Buffer)
	encState := newEncoderState(b)
	for _, tt := range encodeT {
		b.Reset()
		encState.encodeUint(tt.x)
		if !bytes.Equal(tt.b, b.Bytes()) {
			t.Errorf("encodeUint: %#x encode: expected % x got % x", tt.x, tt.b, b.Bytes())
		}
	}
	decState := newDecodeState(b)
	for u := uint64(0); ; u = (u + 1) * 7 {
		b.Reset()
		encState.encodeUint(u)
		v := decState.decodeUint()
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
	encState := newEncoderState(b)
	encState.encodeInt(i)
	decState := newDecodeState(b)
	decState.buf = make([]byte, 8)
	j := decState.decodeInt()
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

func newDecodeState(buf *bytes.Buffer) *decoderState {
	d := new(decoderState)
	d.b = buf
	d.buf = make([]byte, uint64Size)
	return d
}

func newEncoderState(b *bytes.Buffer) *encoderState {
	b.Reset()
	state := &encoderState{enc: nil, b: b}
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
		state := newEncoderState(b)
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
		state := newEncoderState(b)
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
		state := newEncoderState(b)
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
		state := newEncoderState(b)
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
		state := newEncoderState(b)
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
		state := newEncoderState(b)
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
		state := newEncoderState(b)
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
		state := newEncoderState(b)
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
		state := newEncoderState(b)
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
		state := newEncoderState(b)
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
		state := newEncoderState(b)
		instr.op(instr, state, unsafe.Pointer(&data))
		if !bytes.Equal(unsignedResult, b.Bytes()) {
			t.Errorf("uint64 enc instructions: expected % x got % x", unsignedResult, b.Bytes())
		}
	}

	// float32
	{
		b.Reset()
		data := struct{ a float32 }{17}
		instr := &encInstr{encFloat32, 6, 0, 0}
		state := newEncoderState(b)
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
		state := newEncoderState(b)
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
		state := newEncoderState(b)
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
		state := newEncoderState(b)
		instr.op(instr, state, unsafe.Pointer(&data))
		if !bytes.Equal(bytesResult, b.Bytes()) {
			t.Errorf("string enc instructions: expected % x got % x", bytesResult, b.Bytes())
		}
	}
}

func execDec(typ string, instr *decInstr, state *decoderState, t *testing.T, p unsafe.Pointer) {
	defer testError(t)
	v := int(state.decodeUint())
	if v+state.fieldnum != 6 {
		t.Fatalf("decoding field number %d, got %d", 6, v+state.fieldnum)
	}
	instr.op(instr, state, decIndirect(p, instr.indir))
	state.fieldnum = 6
}

func newDecodeStateFromData(data []byte) *decoderState {
	b := bytes.NewBuffer(data)
	state := newDecodeState(b)
	state.fieldnum = -1
	return state
}

// Test instruction execution for decoding.
// Do not run the machine yet; instead do individual instructions crafted by hand.
func TestScalarDecInstructions(t *testing.T) {
	ovfl := errors.New("overflow")

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
		instr := &decInstr{decOpTable[reflect.Int], 6, 0, 0, ovfl}
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
		instr := &decInstr{decOpTable[reflect.Uint], 6, 0, 0, ovfl}
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
		instr := &decInstr{decOpTable[reflect.Uintptr], 6, 0, 0, ovfl}
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

	// complex64
	{
		var data struct {
			a complex64
		}
		instr := &decInstr{decOpTable[reflect.Complex64], 6, 0, 0, ovfl}
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
		instr := &decInstr{decOpTable[reflect.Complex128], 6, 0, 0, ovfl}
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
		instr := &decInstr{decUint8Slice, 6, 0, 0, ovfl}
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
		T string
	}
	s1 := "string1"
	s2 := "string2"
	type T1 struct {
		A, B, C  int
		M        map[string]*float64
		EmptyMap map[string]int // to check that we receive a non-nil map.
		N        *[3]float64
		Strs     *[2]string
		Int64s   *[]int64
		RI       complex64
		S        string
		Y        []byte
		T        *T2
	}
	pi := 3.14159
	e := 2.71828
	t1 := &T1{
		A:        17,
		B:        18,
		C:        -5,
		M:        map[string]*float64{"pi": &pi, "e": &e},
		EmptyMap: make(map[string]int),
		N:        &[3]float64{1.5, 2.5, 3.5},
		Strs:     &[2]string{s1, s2},
		Int64s:   &[]int64{77, 89, 123412342134},
		RI:       17 - 23i,
		S:        "Now is the time",
		Y:        []byte("hello, sailor"),
		T:        &T2{"this is T2"},
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
	// Be absolutely sure the received map is non-nil.
	if t1.EmptyMap == nil {
		t.Errorf("nil map sent")
	}
	if _t1.EmptyMap == nil {
		t.Errorf("nil map received")
	}
}

func TestOverflow(t *testing.T) {
	type inputT struct {
		Maxi int64
		Mini int64
		Maxu uint64
		Maxf float64
		Minf float64
		Maxc complex128
		Minc complex128
	}
	var it inputT
	var err error
	b := new(bytes.Buffer)
	enc := NewEncoder(b)
	dec := NewDecoder(b)

	// int8
	b.Reset()
	it = inputT{
		Maxi: math.MaxInt8 + 1,
	}
	type outi8 struct {
		Maxi int8
		Mini int8
	}
	var o1 outi8
	enc.Encode(it)
	err = dec.Decode(&o1)
	if err == nil || err.Error() != `value for "Maxi" out of range` {
		t.Error("wrong overflow error for int8:", err)
	}
	it = inputT{
		Mini: math.MinInt8 - 1,
	}
	b.Reset()
	enc.Encode(it)
	err = dec.Decode(&o1)
	if err == nil || err.Error() != `value for "Mini" out of range` {
		t.Error("wrong underflow error for int8:", err)
	}

	// int16
	b.Reset()
	it = inputT{
		Maxi: math.MaxInt16 + 1,
	}
	type outi16 struct {
		Maxi int16
		Mini int16
	}
	var o2 outi16
	enc.Encode(it)
	err = dec.Decode(&o2)
	if err == nil || err.Error() != `value for "Maxi" out of range` {
		t.Error("wrong overflow error for int16:", err)
	}
	it = inputT{
		Mini: math.MinInt16 - 1,
	}
	b.Reset()
	enc.Encode(it)
	err = dec.Decode(&o2)
	if err == nil || err.Error() != `value for "Mini" out of range` {
		t.Error("wrong underflow error for int16:", err)
	}

	// int32
	b.Reset()
	it = inputT{
		Maxi: math.MaxInt32 + 1,
	}
	type outi32 struct {
		Maxi int32
		Mini int32
	}
	var o3 outi32
	enc.Encode(it)
	err = dec.Decode(&o3)
	if err == nil || err.Error() != `value for "Maxi" out of range` {
		t.Error("wrong overflow error for int32:", err)
	}
	it = inputT{
		Mini: math.MinInt32 - 1,
	}
	b.Reset()
	enc.Encode(it)
	err = dec.Decode(&o3)
	if err == nil || err.Error() != `value for "Mini" out of range` {
		t.Error("wrong underflow error for int32:", err)
	}

	// uint8
	b.Reset()
	it = inputT{
		Maxu: math.MaxUint8 + 1,
	}
	type outu8 struct {
		Maxu uint8
	}
	var o4 outu8
	enc.Encode(it)
	err = dec.Decode(&o4)
	if err == nil || err.Error() != `value for "Maxu" out of range` {
		t.Error("wrong overflow error for uint8:", err)
	}

	// uint16
	b.Reset()
	it = inputT{
		Maxu: math.MaxUint16 + 1,
	}
	type outu16 struct {
		Maxu uint16
	}
	var o5 outu16
	enc.Encode(it)
	err = dec.Decode(&o5)
	if err == nil || err.Error() != `value for "Maxu" out of range` {
		t.Error("wrong overflow error for uint16:", err)
	}

	// uint32
	b.Reset()
	it = inputT{
		Maxu: math.MaxUint32 + 1,
	}
	type outu32 struct {
		Maxu uint32
	}
	var o6 outu32
	enc.Encode(it)
	err = dec.Decode(&o6)
	if err == nil || err.Error() != `value for "Maxu" out of range` {
		t.Error("wrong overflow error for uint32:", err)
	}

	// float32
	b.Reset()
	it = inputT{
		Maxf: math.MaxFloat32 * 2,
	}
	type outf32 struct {
		Maxf float32
		Minf float32
	}
	var o7 outf32
	enc.Encode(it)
	err = dec.Decode(&o7)
	if err == nil || err.Error() != `value for "Maxf" out of range` {
		t.Error("wrong overflow error for float32:", err)
	}

	// complex64
	b.Reset()
	it = inputT{
		Maxc: complex(math.MaxFloat32*2, math.MaxFloat32*2),
	}
	type outc64 struct {
		Maxc complex64
		Minc complex64
	}
	var o8 outc64
	enc.Encode(it)
	err = dec.Decode(&o8)
	if err == nil || err.Error() != `value for "Maxc" out of range` {
		t.Error("wrong overflow error for complex64:", err)
	}
}

func TestNesting(t *testing.T) {
	type RT struct {
		A    string
		Next *RT
	}
	rt := new(RT)
	rt.A = "level1"
	rt.Next = new(RT)
	rt.Next.A = "level2"
	b := new(bytes.Buffer)
	NewEncoder(b).Encode(rt)
	var drt RT
	dec := NewDecoder(b)
	err := dec.Decode(&drt)
	if err != nil {
		t.Fatal("decoder error:", err)
	}
	if drt.A != rt.A {
		t.Errorf("nesting: encode expected %v got %v", *rt, drt)
	}
	if drt.Next == nil {
		t.Errorf("nesting: recursion failed")
	}
	if drt.Next.A != rt.Next.A {
		t.Errorf("nesting: encode expected %v got %v", *rt.Next, *drt.Next)
	}
}

// These three structures have the same data with different indirections
type T0 struct {
	A int
	B int
	C int
	D int
}
type T1 struct {
	A int
	B *int
	C **int
	D ***int
}
type T2 struct {
	A ***int
	B **int
	C *int
	D int
}

func TestAutoIndirection(t *testing.T) {
	// First transfer t1 into t0
	var t1 T1
	t1.A = 17
	t1.B = new(int)
	*t1.B = 177
	t1.C = new(*int)
	*t1.C = new(int)
	**t1.C = 1777
	t1.D = new(**int)
	*t1.D = new(*int)
	**t1.D = new(int)
	***t1.D = 17777
	b := new(bytes.Buffer)
	enc := NewEncoder(b)
	enc.Encode(t1)
	dec := NewDecoder(b)
	var t0 T0
	dec.Decode(&t0)
	if t0.A != 17 || t0.B != 177 || t0.C != 1777 || t0.D != 17777 {
		t.Errorf("t1->t0: expected {17 177 1777 17777}; got %v", t0)
	}

	// Now transfer t2 into t0
	var t2 T2
	t2.D = 17777
	t2.C = new(int)
	*t2.C = 1777
	t2.B = new(*int)
	*t2.B = new(int)
	**t2.B = 177
	t2.A = new(**int)
	*t2.A = new(*int)
	**t2.A = new(int)
	***t2.A = 17
	b.Reset()
	enc.Encode(t2)
	t0 = T0{}
	dec.Decode(&t0)
	if t0.A != 17 || t0.B != 177 || t0.C != 1777 || t0.D != 17777 {
		t.Errorf("t2->t0 expected {17 177 1777 17777}; got %v", t0)
	}

	// Now transfer t0 into t1
	t0 = T0{17, 177, 1777, 17777}
	b.Reset()
	enc.Encode(t0)
	t1 = T1{}
	dec.Decode(&t1)
	if t1.A != 17 || *t1.B != 177 || **t1.C != 1777 || ***t1.D != 17777 {
		t.Errorf("t0->t1 expected {17 177 1777 17777}; got {%d %d %d %d}", t1.A, *t1.B, **t1.C, ***t1.D)
	}

	// Now transfer t0 into t2
	b.Reset()
	enc.Encode(t0)
	t2 = T2{}
	dec.Decode(&t2)
	if ***t2.A != 17 || **t2.B != 177 || *t2.C != 1777 || t2.D != 17777 {
		t.Errorf("t0->t2 expected {17 177 1777 17777}; got {%d %d %d %d}", ***t2.A, **t2.B, *t2.C, t2.D)
	}

	// Now do t2 again but without pre-allocated pointers.
	b.Reset()
	enc.Encode(t0)
	***t2.A = 0
	**t2.B = 0
	*t2.C = 0
	t2.D = 0
	dec.Decode(&t2)
	if ***t2.A != 17 || **t2.B != 177 || *t2.C != 1777 || t2.D != 17777 {
		t.Errorf("t0->t2 expected {17 177 1777 17777}; got {%d %d %d %d}", ***t2.A, **t2.B, *t2.C, t2.D)
	}
}

type RT0 struct {
	A int
	B string
	C float64
}
type RT1 struct {
	C      float64
	B      string
	A      int
	NotSet string
}

func TestReorderedFields(t *testing.T) {
	var rt0 RT0
	rt0.A = 17
	rt0.B = "hello"
	rt0.C = 3.14159
	b := new(bytes.Buffer)
	NewEncoder(b).Encode(rt0)
	dec := NewDecoder(b)
	var rt1 RT1
	// Wire type is RT0, local type is RT1.
	err := dec.Decode(&rt1)
	if err != nil {
		t.Fatal("decode error:", err)
	}
	if rt0.A != rt1.A || rt0.B != rt1.B || rt0.C != rt1.C {
		t.Errorf("rt1->rt0: expected %v; got %v", rt0, rt1)
	}
}

// Like an RT0 but with fields we'll ignore on the decode side.
type IT0 struct {
	A        int64
	B        string
	Ignore_d []int
	Ignore_e [3]float64
	Ignore_f bool
	Ignore_g string
	Ignore_h []byte
	Ignore_i *RT1
	Ignore_m map[string]int
	C        float64
}

func TestIgnoredFields(t *testing.T) {
	var it0 IT0
	it0.A = 17
	it0.B = "hello"
	it0.C = 3.14159
	it0.Ignore_d = []int{1, 2, 3}
	it0.Ignore_e[0] = 1.0
	it0.Ignore_e[1] = 2.0
	it0.Ignore_e[2] = 3.0
	it0.Ignore_f = true
	it0.Ignore_g = "pay no attention"
	it0.Ignore_h = []byte("to the curtain")
	it0.Ignore_i = &RT1{3.1, "hi", 7, "hello"}
	it0.Ignore_m = map[string]int{"one": 1, "two": 2}

	b := new(bytes.Buffer)
	NewEncoder(b).Encode(it0)
	dec := NewDecoder(b)
	var rt1 RT1
	// Wire type is IT0, local type is RT1.
	err := dec.Decode(&rt1)
	if err != nil {
		t.Error("error: ", err)
	}
	if int(it0.A) != rt1.A || it0.B != rt1.B || it0.C != rt1.C {
		t.Errorf("rt0->rt1: expected %v; got %v", it0, rt1)
	}
}

func TestBadRecursiveType(t *testing.T) {
	type Rec ***Rec
	var rec Rec
	b := new(bytes.Buffer)
	err := NewEncoder(b).Encode(&rec)
	if err == nil {
		t.Error("expected error; got none")
	} else if strings.Index(err.Error(), "recursive") < 0 {
		t.Error("expected recursive type error; got", err)
	}
	// Can't test decode easily because we can't encode one, so we can't pass one to a Decoder.
}

type Bad0 struct {
	CH chan int
	C  float64
}

func TestInvalidField(t *testing.T) {
	var bad0 Bad0
	bad0.CH = make(chan int)
	b := new(bytes.Buffer)
	dummyEncoder := new(Encoder) // sufficient for this purpose.
	dummyEncoder.encode(b, reflect.ValueOf(&bad0), userType(reflect.TypeOf(&bad0)))
	if err := dummyEncoder.err; err == nil {
		t.Error("expected error; got none")
	} else if strings.Index(err.Error(), "type") < 0 {
		t.Error("expected type error; got", err)
	}
}

type Indirect struct {
	A ***[3]int
	S ***[]int
	M ****map[string]int
}

type Direct struct {
	A [3]int
	S []int
	M map[string]int
}

func TestIndirectSliceMapArray(t *testing.T) {
	// Marshal indirect, unmarshal to direct.
	i := new(Indirect)
	i.A = new(**[3]int)
	*i.A = new(*[3]int)
	**i.A = new([3]int)
	***i.A = [3]int{1, 2, 3}
	i.S = new(**[]int)
	*i.S = new(*[]int)
	**i.S = new([]int)
	***i.S = []int{4, 5, 6}
	i.M = new(***map[string]int)
	*i.M = new(**map[string]int)
	**i.M = new(*map[string]int)
	***i.M = new(map[string]int)
	****i.M = map[string]int{"one": 1, "two": 2, "three": 3}
	b := new(bytes.Buffer)
	NewEncoder(b).Encode(i)
	dec := NewDecoder(b)
	var d Direct
	err := dec.Decode(&d)
	if err != nil {
		t.Error("error: ", err)
	}
	if len(d.A) != 3 || d.A[0] != 1 || d.A[1] != 2 || d.A[2] != 3 {
		t.Errorf("indirect to direct: d.A is %v not %v", d.A, ***i.A)
	}
	if len(d.S) != 3 || d.S[0] != 4 || d.S[1] != 5 || d.S[2] != 6 {
		t.Errorf("indirect to direct: d.S is %v not %v", d.S, ***i.S)
	}
	if len(d.M) != 3 || d.M["one"] != 1 || d.M["two"] != 2 || d.M["three"] != 3 {
		t.Errorf("indirect to direct: d.M is %v not %v", d.M, ***i.M)
	}
	// Marshal direct, unmarshal to indirect.
	d.A = [3]int{11, 22, 33}
	d.S = []int{44, 55, 66}
	d.M = map[string]int{"four": 4, "five": 5, "six": 6}
	i = new(Indirect)
	b.Reset()
	NewEncoder(b).Encode(d)
	dec = NewDecoder(b)
	err = dec.Decode(&i)
	if err != nil {
		t.Fatal("error: ", err)
	}
	if len(***i.A) != 3 || (***i.A)[0] != 11 || (***i.A)[1] != 22 || (***i.A)[2] != 33 {
		t.Errorf("direct to indirect: ***i.A is %v not %v", ***i.A, d.A)
	}
	if len(***i.S) != 3 || (***i.S)[0] != 44 || (***i.S)[1] != 55 || (***i.S)[2] != 66 {
		t.Errorf("direct to indirect: ***i.S is %v not %v", ***i.S, ***i.S)
	}
	if len(****i.M) != 3 || (****i.M)["four"] != 4 || (****i.M)["five"] != 5 || (****i.M)["six"] != 6 {
		t.Errorf("direct to indirect: ****i.M is %v not %v", ****i.M, d.M)
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

type Float float64

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
	X, Y int
}

func (p Point) Square() int {
	return p.X*p.X + p.Y*p.Y
}

// A struct with interfaces in it.
type InterfaceItem struct {
	I             int
	Sq1, Sq2, Sq3 Squarer
	F             float64
	Sq            []Squarer
}

// The same struct without interfaces
type NoInterfaceItem struct {
	I int
	F float64
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
	if item2.I != item1.I {
		t.Error("normal int did not decode correctly")
	}
	if item2.Sq1 == nil || item2.Sq1.Square() != iVal.Square() {
		t.Error("Int did not decode correctly")
	}
	if item2.Sq2 == nil || item2.Sq2.Square() != fVal.Square() {
		t.Error("Float did not decode correctly")
	}
	if item2.Sq3 == nil || item2.Sq3.Square() != vVal.Square() {
		t.Error("Vector did not decode correctly")
	}
	if item2.F != item1.F {
		t.Error("normal float did not decode correctly")
	}
	// Now check that we received a slice of Squarers correctly, including a nil element
	if len(item1.Sq) != len(item2.Sq) {
		t.Fatalf("[]Squarer length wrong: got %d; expected %d", len(item2.Sq), len(item1.Sq))
	}
	for i, v1 := range item1.Sq {
		v2 := item2.Sq[i]
		if v1 == nil || v2 == nil {
			if v1 != nil || v2 != nil {
				t.Errorf("item %d inconsistent nils", i)
			}
			continue
			if v1.Square() != v2.Square() {
				t.Errorf("item %d inconsistent values: %v %v", i, v1, v2)
			}
		}
	}
}

// A struct with all basic types, stored in interfaces.
type BasicInterfaceItem struct {
	Int, Int8, Int16, Int32, Int64      interface{}
	Uint, Uint8, Uint16, Uint32, Uint64 interface{}
	Float32, Float64                    interface{}
	Complex64, Complex128               interface{}
	Bool                                interface{}
	String                              interface{}
	Bytes                               interface{}
}

func TestInterfaceBasic(t *testing.T) {
	b := new(bytes.Buffer)
	item1 := &BasicInterfaceItem{
		int(1), int8(1), int16(1), int32(1), int64(1),
		uint(1), uint8(1), uint16(1), uint32(1), uint64(1),
		float32(1), 1.0,
		complex64(1i), complex128(1i),
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
	Str1 interface{} // basic
	Str2 interface{} // derived
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
	if v, ok := item2.Str1.(string); !ok || v != str1 {
		t.Errorf("basic string failed: %q should be %q", v, str1)
	}
	if v, ok := item2.Str2.(String); !ok || v != str2 {
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
	if item2.I != item1.I {
		t.Error("normal int did not decode correctly")
	}
	if item2.F != item2.F {
		t.Error("normal float did not decode correctly")
	}
}

type U struct {
	A int
	B string
	c float64
	D uint
}

func TestUnexportedFields(t *testing.T) {
	var u0 U
	u0.A = 17
	u0.B = "hello"
	u0.c = 3.14159
	u0.D = 23
	b := new(bytes.Buffer)
	NewEncoder(b).Encode(u0)
	dec := NewDecoder(b)
	var u1 U
	u1.c = 1234.
	err := dec.Decode(&u1)
	if err != nil {
		t.Fatal("decode error:", err)
	}
	if u0.A != u0.A || u0.B != u1.B || u0.D != u1.D {
		t.Errorf("u1->u0: expected %v; got %v", u0, u1)
	}
	if u1.c != 1234. {
		t.Error("u1.c modified")
	}
}

var singletons = []interface{}{
	true,
	7,
	3.2,
	"hello",
	[3]int{11, 22, 33},
	[]float32{0.5, 0.25, 0.125},
	map[string]int{"one": 1, "two": 2},
}

func TestDebugSingleton(t *testing.T) {
	if debugFunc == nil {
		return
	}
	b := new(bytes.Buffer)
	// Accumulate a number of values and print them out all at once.
	for _, x := range singletons {
		err := NewEncoder(b).Encode(x)
		if err != nil {
			t.Fatal("encode:", err)
		}
	}
	debugFunc(b)
}

// A type that won't be defined in the gob until we send it in an interface value.
type OnTheFly struct {
	A int
}

type DT struct {
	//	X OnTheFly
	A     int
	B     string
	C     float64
	I     interface{}
	J     interface{}
	I_nil interface{}
	M     map[string]int
	T     [3]int
	S     []string
}

func TestDebugStruct(t *testing.T) {
	if debugFunc == nil {
		return
	}
	Register(OnTheFly{})
	var dt DT
	dt.A = 17
	dt.B = "hello"
	dt.C = 3.14159
	dt.I = 271828
	dt.J = OnTheFly{3}
	dt.I_nil = nil
	dt.M = map[string]int{"one": 1, "two": 2}
	dt.T = [3]int{11, 22, 33}
	dt.S = []string{"hi", "joe"}
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
