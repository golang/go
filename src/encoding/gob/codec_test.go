// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gob

import (
	"bytes"
	"errors"
	"flag"
	"math"
	"math/rand"
	"reflect"
	"strings"
	"testing"
	"time"
	"unsafe"
)

var doFuzzTests = flag.Bool("gob.fuzz", false, "run the fuzz tests, which are large and very slow")

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
}

func newDecBuffer(data []byte) *decBuffer {
	return &decBuffer{
		data: data,
	}
}

// Test basic encode/decode routines for unsigned integers
func TestUintCodec(t *testing.T) {
	defer testError(t)
	b := new(encBuffer)
	encState := newEncoderState(b)
	for _, tt := range encodeT {
		b.Reset()
		encState.encodeUint(tt.x)
		if !bytes.Equal(tt.b, b.Bytes()) {
			t.Errorf("encodeUint: %#x encode: expected % x got % x", tt.x, tt.b, b.Bytes())
		}
	}
	for u := uint64(0); ; u = (u + 1) * 7 {
		b.Reset()
		encState.encodeUint(u)
		decState := newDecodeState(newDecBuffer(b.Bytes()))
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
	var b = new(encBuffer)
	encState := newEncoderState(b)
	encState.encodeInt(i)
	decState := newDecodeState(newDecBuffer(b.Bytes()))
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

func newDecodeState(buf *decBuffer) *decoderState {
	d := new(decoderState)
	d.b = buf
	return d
}

func newEncoderState(b *encBuffer) *encoderState {
	b.Reset()
	state := &encoderState{enc: nil, b: b}
	state.fieldnum = -1
	return state
}

// Test instruction execution for encoding.
// Do not run the machine yet; instead do individual instructions crafted by hand.
func TestScalarEncInstructions(t *testing.T) {
	var b = new(encBuffer)

	// bool
	{
		var data bool = true
		instr := &encInstr{encBool, 6, nil, 0}
		state := newEncoderState(b)
		instr.op(instr, state, reflect.ValueOf(data))
		if !bytes.Equal(boolResult, b.Bytes()) {
			t.Errorf("bool enc instructions: expected % x got % x", boolResult, b.Bytes())
		}
	}

	// int
	{
		b.Reset()
		var data int = 17
		instr := &encInstr{encInt, 6, nil, 0}
		state := newEncoderState(b)
		instr.op(instr, state, reflect.ValueOf(data))
		if !bytes.Equal(signedResult, b.Bytes()) {
			t.Errorf("int enc instructions: expected % x got % x", signedResult, b.Bytes())
		}
	}

	// uint
	{
		b.Reset()
		var data uint = 17
		instr := &encInstr{encUint, 6, nil, 0}
		state := newEncoderState(b)
		instr.op(instr, state, reflect.ValueOf(data))
		if !bytes.Equal(unsignedResult, b.Bytes()) {
			t.Errorf("uint enc instructions: expected % x got % x", unsignedResult, b.Bytes())
		}
	}

	// int8
	{
		b.Reset()
		var data int8 = 17
		instr := &encInstr{encInt, 6, nil, 0}
		state := newEncoderState(b)
		instr.op(instr, state, reflect.ValueOf(data))
		if !bytes.Equal(signedResult, b.Bytes()) {
			t.Errorf("int8 enc instructions: expected % x got % x", signedResult, b.Bytes())
		}
	}

	// uint8
	{
		b.Reset()
		var data uint8 = 17
		instr := &encInstr{encUint, 6, nil, 0}
		state := newEncoderState(b)
		instr.op(instr, state, reflect.ValueOf(data))
		if !bytes.Equal(unsignedResult, b.Bytes()) {
			t.Errorf("uint8 enc instructions: expected % x got % x", unsignedResult, b.Bytes())
		}
	}

	// int16
	{
		b.Reset()
		var data int16 = 17
		instr := &encInstr{encInt, 6, nil, 0}
		state := newEncoderState(b)
		instr.op(instr, state, reflect.ValueOf(data))
		if !bytes.Equal(signedResult, b.Bytes()) {
			t.Errorf("int16 enc instructions: expected % x got % x", signedResult, b.Bytes())
		}
	}

	// uint16
	{
		b.Reset()
		var data uint16 = 17
		instr := &encInstr{encUint, 6, nil, 0}
		state := newEncoderState(b)
		instr.op(instr, state, reflect.ValueOf(data))
		if !bytes.Equal(unsignedResult, b.Bytes()) {
			t.Errorf("uint16 enc instructions: expected % x got % x", unsignedResult, b.Bytes())
		}
	}

	// int32
	{
		b.Reset()
		var data int32 = 17
		instr := &encInstr{encInt, 6, nil, 0}
		state := newEncoderState(b)
		instr.op(instr, state, reflect.ValueOf(data))
		if !bytes.Equal(signedResult, b.Bytes()) {
			t.Errorf("int32 enc instructions: expected % x got % x", signedResult, b.Bytes())
		}
	}

	// uint32
	{
		b.Reset()
		var data uint32 = 17
		instr := &encInstr{encUint, 6, nil, 0}
		state := newEncoderState(b)
		instr.op(instr, state, reflect.ValueOf(data))
		if !bytes.Equal(unsignedResult, b.Bytes()) {
			t.Errorf("uint32 enc instructions: expected % x got % x", unsignedResult, b.Bytes())
		}
	}

	// int64
	{
		b.Reset()
		var data int64 = 17
		instr := &encInstr{encInt, 6, nil, 0}
		state := newEncoderState(b)
		instr.op(instr, state, reflect.ValueOf(data))
		if !bytes.Equal(signedResult, b.Bytes()) {
			t.Errorf("int64 enc instructions: expected % x got % x", signedResult, b.Bytes())
		}
	}

	// uint64
	{
		b.Reset()
		var data uint64 = 17
		instr := &encInstr{encUint, 6, nil, 0}
		state := newEncoderState(b)
		instr.op(instr, state, reflect.ValueOf(data))
		if !bytes.Equal(unsignedResult, b.Bytes()) {
			t.Errorf("uint64 enc instructions: expected % x got % x", unsignedResult, b.Bytes())
		}
	}

	// float32
	{
		b.Reset()
		var data float32 = 17
		instr := &encInstr{encFloat, 6, nil, 0}
		state := newEncoderState(b)
		instr.op(instr, state, reflect.ValueOf(data))
		if !bytes.Equal(floatResult, b.Bytes()) {
			t.Errorf("float32 enc instructions: expected % x got % x", floatResult, b.Bytes())
		}
	}

	// float64
	{
		b.Reset()
		var data float64 = 17
		instr := &encInstr{encFloat, 6, nil, 0}
		state := newEncoderState(b)
		instr.op(instr, state, reflect.ValueOf(data))
		if !bytes.Equal(floatResult, b.Bytes()) {
			t.Errorf("float64 enc instructions: expected % x got % x", floatResult, b.Bytes())
		}
	}

	// bytes == []uint8
	{
		b.Reset()
		data := []byte("hello")
		instr := &encInstr{encUint8Array, 6, nil, 0}
		state := newEncoderState(b)
		instr.op(instr, state, reflect.ValueOf(data))
		if !bytes.Equal(bytesResult, b.Bytes()) {
			t.Errorf("bytes enc instructions: expected % x got % x", bytesResult, b.Bytes())
		}
	}

	// string
	{
		b.Reset()
		var data string = "hello"
		instr := &encInstr{encString, 6, nil, 0}
		state := newEncoderState(b)
		instr.op(instr, state, reflect.ValueOf(data))
		if !bytes.Equal(bytesResult, b.Bytes()) {
			t.Errorf("string enc instructions: expected % x got % x", bytesResult, b.Bytes())
		}
	}
}

func execDec(instr *decInstr, state *decoderState, t *testing.T, value reflect.Value) {
	defer testError(t)
	v := int(state.decodeUint())
	if v+state.fieldnum != 6 {
		t.Fatalf("decoding field number %d, got %d", 6, v+state.fieldnum)
	}
	instr.op(instr, state, value.Elem())
	state.fieldnum = 6
}

func newDecodeStateFromData(data []byte) *decoderState {
	b := newDecBuffer(data)
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
		var data bool
		instr := &decInstr{decBool, 6, nil, ovfl}
		state := newDecodeStateFromData(boolResult)
		execDec(instr, state, t, reflect.ValueOf(&data))
		if data != true {
			t.Errorf("bool a = %v not true", data)
		}
	}
	// int
	{
		var data int
		instr := &decInstr{decOpTable[reflect.Int], 6, nil, ovfl}
		state := newDecodeStateFromData(signedResult)
		execDec(instr, state, t, reflect.ValueOf(&data))
		if data != 17 {
			t.Errorf("int a = %v not 17", data)
		}
	}

	// uint
	{
		var data uint
		instr := &decInstr{decOpTable[reflect.Uint], 6, nil, ovfl}
		state := newDecodeStateFromData(unsignedResult)
		execDec(instr, state, t, reflect.ValueOf(&data))
		if data != 17 {
			t.Errorf("uint a = %v not 17", data)
		}
	}

	// int8
	{
		var data int8
		instr := &decInstr{decInt8, 6, nil, ovfl}
		state := newDecodeStateFromData(signedResult)
		execDec(instr, state, t, reflect.ValueOf(&data))
		if data != 17 {
			t.Errorf("int8 a = %v not 17", data)
		}
	}

	// uint8
	{
		var data uint8
		instr := &decInstr{decUint8, 6, nil, ovfl}
		state := newDecodeStateFromData(unsignedResult)
		execDec(instr, state, t, reflect.ValueOf(&data))
		if data != 17 {
			t.Errorf("uint8 a = %v not 17", data)
		}
	}

	// int16
	{
		var data int16
		instr := &decInstr{decInt16, 6, nil, ovfl}
		state := newDecodeStateFromData(signedResult)
		execDec(instr, state, t, reflect.ValueOf(&data))
		if data != 17 {
			t.Errorf("int16 a = %v not 17", data)
		}
	}

	// uint16
	{
		var data uint16
		instr := &decInstr{decUint16, 6, nil, ovfl}
		state := newDecodeStateFromData(unsignedResult)
		execDec(instr, state, t, reflect.ValueOf(&data))
		if data != 17 {
			t.Errorf("uint16 a = %v not 17", data)
		}
	}

	// int32
	{
		var data int32
		instr := &decInstr{decInt32, 6, nil, ovfl}
		state := newDecodeStateFromData(signedResult)
		execDec(instr, state, t, reflect.ValueOf(&data))
		if data != 17 {
			t.Errorf("int32 a = %v not 17", data)
		}
	}

	// uint32
	{
		var data uint32
		instr := &decInstr{decUint32, 6, nil, ovfl}
		state := newDecodeStateFromData(unsignedResult)
		execDec(instr, state, t, reflect.ValueOf(&data))
		if data != 17 {
			t.Errorf("uint32 a = %v not 17", data)
		}
	}

	// uintptr
	{
		var data uintptr
		instr := &decInstr{decOpTable[reflect.Uintptr], 6, nil, ovfl}
		state := newDecodeStateFromData(unsignedResult)
		execDec(instr, state, t, reflect.ValueOf(&data))
		if data != 17 {
			t.Errorf("uintptr a = %v not 17", data)
		}
	}

	// int64
	{
		var data int64
		instr := &decInstr{decInt64, 6, nil, ovfl}
		state := newDecodeStateFromData(signedResult)
		execDec(instr, state, t, reflect.ValueOf(&data))
		if data != 17 {
			t.Errorf("int64 a = %v not 17", data)
		}
	}

	// uint64
	{
		var data uint64
		instr := &decInstr{decUint64, 6, nil, ovfl}
		state := newDecodeStateFromData(unsignedResult)
		execDec(instr, state, t, reflect.ValueOf(&data))
		if data != 17 {
			t.Errorf("uint64 a = %v not 17", data)
		}
	}

	// float32
	{
		var data float32
		instr := &decInstr{decFloat32, 6, nil, ovfl}
		state := newDecodeStateFromData(floatResult)
		execDec(instr, state, t, reflect.ValueOf(&data))
		if data != 17 {
			t.Errorf("float32 a = %v not 17", data)
		}
	}

	// float64
	{
		var data float64
		instr := &decInstr{decFloat64, 6, nil, ovfl}
		state := newDecodeStateFromData(floatResult)
		execDec(instr, state, t, reflect.ValueOf(&data))
		if data != 17 {
			t.Errorf("float64 a = %v not 17", data)
		}
	}

	// complex64
	{
		var data complex64
		instr := &decInstr{decOpTable[reflect.Complex64], 6, nil, ovfl}
		state := newDecodeStateFromData(complexResult)
		execDec(instr, state, t, reflect.ValueOf(&data))
		if data != 17+19i {
			t.Errorf("complex a = %v not 17+19i", data)
		}
	}

	// complex128
	{
		var data complex128
		instr := &decInstr{decOpTable[reflect.Complex128], 6, nil, ovfl}
		state := newDecodeStateFromData(complexResult)
		execDec(instr, state, t, reflect.ValueOf(&data))
		if data != 17+19i {
			t.Errorf("complex a = %v not 17+19i", data)
		}
	}

	// bytes == []uint8
	{
		var data []byte
		instr := &decInstr{decUint8Slice, 6, nil, ovfl}
		state := newDecodeStateFromData(bytesResult)
		execDec(instr, state, t, reflect.ValueOf(&data))
		if string(data) != "hello" {
			t.Errorf(`bytes a = %q not "hello"`, string(data))
		}
	}

	// string
	{
		var data string
		instr := &decInstr{decString, 6, nil, ovfl}
		state := newDecodeStateFromData(bytesResult)
		execDec(instr, state, t, reflect.ValueOf(&data))
		if data != "hello" {
			t.Errorf(`bytes a = %q not "hello"`, data)
		}
	}
}

func TestEndToEnd(t *testing.T) {
	type T2 struct {
		T string
	}
	type T3 struct {
		X float64
		Z *int
	}
	type T1 struct {
		A, B, C  int
		M        map[string]*float64
		M2       map[int]T3
		Mstring  map[string]string
		Mintptr  map[int]*int
		Mcomp    map[complex128]complex128
		Marr     map[[2]string][2]*float64
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
	two := 2.0
	meaning := 42
	fingers := 5
	s1 := "string1"
	s2 := "string2"
	var comp1 complex128 = complex(1.0, 1.0)
	var comp2 complex128 = complex(1.0, 1.0)
	var arr1 [2]string
	arr1[0] = s1
	arr1[1] = s2
	var arr2 [2]string
	arr2[0] = s2
	arr2[1] = s1
	var floatArr1 [2]*float64
	floatArr1[0] = &pi
	floatArr1[1] = &e
	var floatArr2 [2]*float64
	floatArr2[0] = &e
	floatArr2[1] = &two
	t1 := &T1{
		A:        17,
		B:        18,
		C:        -5,
		M:        map[string]*float64{"pi": &pi, "e": &e},
		M2:       map[int]T3{4: {X: pi, Z: &meaning}, 10: {X: e, Z: &fingers}},
		Mstring:  map[string]string{"pi": "3.14", "e": "2.71"},
		Mintptr:  map[int]*int{meaning: &fingers, fingers: &meaning},
		Mcomp:    map[complex128]complex128{comp1: comp2, comp2: comp1},
		Marr:     map[[2]string][2]*float64{arr1: floatArr1, arr2: floatArr2},
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
	} else if !strings.Contains(err.Error(), "recursive") {
		t.Error("expected recursive type error; got", err)
	}
	// Can't test decode easily because we can't encode one, so we can't pass one to a Decoder.
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
		} else if v1.Square() != v2.Square() {
			t.Errorf("item %d inconsistent values: %v %v", i, v1, v2)
		}
	}
}

// A struct with all basic types, stored in interfaces.
type BasicInterfaceItem struct {
	Int, Int8, Int16, Int32, Int64      any
	Uint, Uint8, Uint16, Uint32, Uint64 any
	Float32, Float64                    any
	Complex64, Complex128               any
	Bool                                any
	String                              any
	Bytes                               any
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
	Str1 any // basic
	Str2 any // derived
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
	if item2.F != item1.F {
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
	if u0.A != u1.A || u0.B != u1.B || u0.D != u1.D {
		t.Errorf("u1->u0: expected %v; got %v", u0, u1)
	}
	if u1.c != 1234. {
		t.Error("u1.c modified")
	}
}

var singletons = []any{
	true,
	7,
	uint(10),
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
	I     any
	J     any
	I_nil any
	M     map[string]int
	T     [3]int
	S     []string
}

func newDT() DT {
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
	return dt
}

func TestDebugStruct(t *testing.T) {
	if debugFunc == nil {
		return
	}
	Register(OnTheFly{})
	dt := newDT()
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

func encFuzzDec(rng *rand.Rand, in any) error {
	buf := new(bytes.Buffer)
	enc := NewEncoder(buf)
	if err := enc.Encode(&in); err != nil {
		return err
	}

	b := buf.Bytes()
	for i, bi := range b {
		if rng.Intn(10) < 3 {
			b[i] = bi + uint8(rng.Intn(256))
		}
	}

	dec := NewDecoder(buf)
	var e any
	if err := dec.Decode(&e); err != nil {
		return err
	}
	return nil
}

// This does some "fuzz testing" by attempting to decode a sequence of random bytes.
func TestFuzz(t *testing.T) {
	if !*doFuzzTests {
		t.Skipf("disabled; run with -gob.fuzz to enable")
	}

	// all possible inputs
	input := []any{
		new(int),
		new(float32),
		new(float64),
		new(complex128),
		&ByteStruct{255},
		&ArrayStruct{},
		&StringStruct{"hello"},
		&GobTest1{0, &StringStruct{"hello"}},
	}
	testFuzz(t, time.Now().UnixNano(), 100, input...)
}

func TestFuzzRegressions(t *testing.T) {
	if !*doFuzzTests {
		t.Skipf("disabled; run with -gob.fuzz to enable")
	}

	// An instance triggering a type name of length ~102 GB.
	testFuzz(t, 1328492090837718000, 100, new(float32))
	// An instance triggering a type name of 1.6 GB.
	// Note: can take several minutes to run.
	testFuzz(t, 1330522872628565000, 100, new(int))
}

func testFuzz(t *testing.T, seed int64, n int, input ...any) {
	for _, e := range input {
		t.Logf("seed=%d n=%d e=%T", seed, n, e)
		rng := rand.New(rand.NewSource(seed))
		for i := 0; i < n; i++ {
			encFuzzDec(rng, e)
		}
	}
}

// TestFuzzOneByte tries to decode corrupted input sequences
// and checks that no panic occurs.
func TestFuzzOneByte(t *testing.T) {
	if !*doFuzzTests {
		t.Skipf("disabled; run with -gob.fuzz to enable")
	}

	buf := new(strings.Builder)
	Register(OnTheFly{})
	dt := newDT()
	if err := NewEncoder(buf).Encode(dt); err != nil {
		t.Fatal(err)
	}
	s := buf.String()

	indices := make([]int, 0, len(s))
	for i := 0; i < len(s); i++ {
		switch i {
		case 14, 167, 231, 265: // a slice length, corruptions are not handled yet.
			continue
		case 248:
			// Large map size, which currently causes an out of memory panic.
			// See golang.org/issue/24308 and golang.org/issue/20221.
			continue
		}
		indices = append(indices, i)
	}
	if testing.Short() {
		indices = []int{1, 111, 178} // known fixed panics
	}
	for _, i := range indices {
		for j := 0; j < 256; j += 3 {
			b := []byte(s)
			b[i] ^= byte(j)
			var e DT
			func() {
				defer func() {
					if p := recover(); p != nil {
						t.Errorf("crash for b[%d] ^= 0x%x", i, j)
						panic(p)
					}
				}()
				err := NewDecoder(bytes.NewReader(b)).Decode(&e)
				_ = err
			}()
		}
	}
}

// Don't crash, just give error with invalid type id.
// Issue 9649.
func TestErrorInvalidTypeId(t *testing.T) {
	data := []byte{0x01, 0x00, 0x01, 0x00}
	d := NewDecoder(bytes.NewReader(data))
	// When running d.Decode(&foo) the first time the decoder stops
	// after []byte{0x01, 0x00} and reports an errBadType. Running
	// d.Decode(&foo) again on exactly the same input sequence should
	// give another errBadType, but instead caused a panic because
	// decoderMap wasn't cleaned up properly after the first error.
	for i := 0; i < 2; i++ {
		var foo struct{}
		err := d.Decode(&foo)
		if err != errBadType {
			t.Fatalf("decode: expected %s, got %s", errBadType, err)
		}
	}
}

type LargeSliceByte struct {
	S []byte
}

type LargeSliceInt8 struct {
	S []int8
}

type StringPair struct {
	A, B string
}

type LargeSliceStruct struct {
	S []StringPair
}

type LargeSliceString struct {
	S []string
}

func testEncodeDecode(t *testing.T, in, out any) {
	t.Helper()
	var b bytes.Buffer
	err := NewEncoder(&b).Encode(in)
	if err != nil {
		t.Fatal("encode:", err)
	}
	err = NewDecoder(&b).Decode(out)
	if err != nil {
		t.Fatal("decode:", err)
	}
	if !reflect.DeepEqual(in, out) {
		t.Errorf("output mismatch")
	}
}

func TestLargeSlice(t *testing.T) {
	t.Run("byte", func(t *testing.T) {
		if unsafe.Sizeof(uintptr(0)) > 4 {
			t.Parallel() // Only run in parallel in a large address space
		}
		s := make([]byte, 10<<21)
		for i := range s {
			s[i] = byte(i)
		}
		st := &LargeSliceByte{S: s}
		rt := &LargeSliceByte{}
		testEncodeDecode(t, st, rt)
	})
	t.Run("int8", func(t *testing.T) {
		if unsafe.Sizeof(uintptr(0)) > 4 {
			t.Parallel()
		}
		s := make([]int8, 10<<21)
		for i := range s {
			s[i] = int8(i)
		}
		st := &LargeSliceInt8{S: s}
		rt := &LargeSliceInt8{}
		testEncodeDecode(t, st, rt)
	})
	t.Run("struct", func(t *testing.T) {
		if unsafe.Sizeof(uintptr(0)) > 4 {
			t.Parallel()
		}
		s := make([]StringPair, 1<<21)
		for i := range s {
			s[i].A = string(rune(i))
			s[i].B = s[i].A
		}
		st := &LargeSliceStruct{S: s}
		rt := &LargeSliceStruct{}
		testEncodeDecode(t, st, rt)
	})
	t.Run("string", func(t *testing.T) {
		if unsafe.Sizeof(uintptr(0)) > 4 {
			t.Parallel()
		}
		s := make([]string, 1<<21)
		for i := range s {
			s[i] = string(rune(i))
		}
		st := &LargeSliceString{S: s}
		rt := &LargeSliceString{}
		testEncodeDecode(t, st, rt)
	})
}

func TestLocalRemoteTypesMismatch(t *testing.T) {
	// Test data is from https://go.dev/issue/62117.
	testData := []byte{9, 127, 3, 1, 2, 255, 128, 0, 0, 0, 3, 255, 128, 0}

	var v []*struct{}
	buf := bytes.NewBuffer(testData)
	err := NewDecoder(buf).Decode(&v)
	if err == nil {
		t.Error("Encode/Decode: expected error but got err == nil")
	}
}
