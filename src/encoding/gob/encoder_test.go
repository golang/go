// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gob

import (
	"bytes"
	"cmp"
	"encoding/hex"
	"fmt"
	"io"
	"math"
	"reflect"
	"slices"
	"strings"
	"testing"
)

// Test basic operations in a safe manner.
func TestBasicEncoderDecoder(t *testing.T) {
	var values = []any{
		true,
		int(123),
		int8(123),
		int16(-12345),
		int32(123456),
		int64(-1234567),
		uint(123),
		uint8(123),
		uint16(12345),
		uint32(123456),
		uint64(1234567),
		uintptr(12345678),
		float32(1.2345),
		float64(1.2345678),
		complex64(1.2345 + 2.3456i),
		complex128(1.2345678 + 2.3456789i),
		[]byte("hello"),
		string("hello"),
	}
	for _, value := range values {
		b := new(bytes.Buffer)
		enc := NewEncoder(b)
		err := enc.Encode(value)
		if err != nil {
			t.Error("encoder fail:", err)
		}
		dec := NewDecoder(b)
		result := reflect.New(reflect.TypeOf(value))
		err = dec.Decode(result.Interface())
		if err != nil {
			t.Fatalf("error decoding %T: %v:", reflect.TypeOf(value), err)
		}
		if !reflect.DeepEqual(value, result.Elem().Interface()) {
			t.Fatalf("%T: expected %v got %v", value, value, result.Elem().Interface())
		}
	}
}

func TestEncodeIntSlice(t *testing.T) {

	s8 := []int8{1, 5, 12, 22, 35, 51, 70, 92, 117}
	s16 := []int16{145, 176, 210, 247, 287, 330, 376, 425, 477}
	s32 := []int32{532, 590, 651, 715, 782, 852, 925, 1001, 1080}
	s64 := []int64{1162, 1247, 1335, 1426, 1520, 1617, 1717, 1820, 1926}

	t.Run("int8", func { t ->
		var sink bytes.Buffer
		enc := NewEncoder(&sink)
		enc.Encode(s8)

		dec := NewDecoder(&sink)
		res := make([]int8, 9)
		dec.Decode(&res)

		if !reflect.DeepEqual(s8, res) {
			t.Fatalf("EncodeIntSlice: expected %v, got %v", s8, res)
		}
	})

	t.Run("int16", func { t ->
		var sink bytes.Buffer
		enc := NewEncoder(&sink)
		enc.Encode(s16)

		dec := NewDecoder(&sink)
		res := make([]int16, 9)
		dec.Decode(&res)

		if !reflect.DeepEqual(s16, res) {
			t.Fatalf("EncodeIntSlice: expected %v, got %v", s16, res)
		}
	})

	t.Run("int32", func { t ->
		var sink bytes.Buffer
		enc := NewEncoder(&sink)
		enc.Encode(s32)

		dec := NewDecoder(&sink)
		res := make([]int32, 9)
		dec.Decode(&res)

		if !reflect.DeepEqual(s32, res) {
			t.Fatalf("EncodeIntSlice: expected %v, got %v", s32, res)
		}
	})

	t.Run("int64", func { t ->
		var sink bytes.Buffer
		enc := NewEncoder(&sink)
		enc.Encode(s64)

		dec := NewDecoder(&sink)
		res := make([]int64, 9)
		dec.Decode(&res)

		if !reflect.DeepEqual(s64, res) {
			t.Fatalf("EncodeIntSlice: expected %v, got %v", s64, res)
		}
	})

}

type ET0 struct {
	A int
	B string
}

type ET2 struct {
	X string
}

type ET1 struct {
	A    int
	Et2  *ET2
	Next *ET1
}

// Like ET1 but with a different name for a field
type ET3 struct {
	A             int
	Et2           *ET2
	DifferentNext *ET1
}

// Like ET1 but with a different type for a field
type ET4 struct {
	A    int
	Et2  float64
	Next int
}

func TestEncoderDecoder(t *testing.T) {
	b := new(bytes.Buffer)
	enc := NewEncoder(b)
	et0 := new(ET0)
	et0.A = 7
	et0.B = "gobs of fun"
	err := enc.Encode(et0)
	if err != nil {
		t.Error("encoder fail:", err)
	}
	//fmt.Printf("% x %q\n", b, b)
	//Debug(b)
	dec := NewDecoder(b)
	newEt0 := new(ET0)
	err = dec.Decode(newEt0)
	if err != nil {
		t.Fatal("error decoding ET0:", err)
	}

	if !reflect.DeepEqual(et0, newEt0) {
		t.Fatalf("invalid data for et0: expected %+v; got %+v", *et0, *newEt0)
	}
	if b.Len() != 0 {
		t.Error("not at eof;", b.Len(), "bytes left")
	}
	//	t.FailNow()

	b = new(bytes.Buffer)
	enc = NewEncoder(b)
	et1 := new(ET1)
	et1.A = 7
	et1.Et2 = new(ET2)
	err = enc.Encode(et1)
	if err != nil {
		t.Error("encoder fail:", err)
	}
	dec = NewDecoder(b)
	newEt1 := new(ET1)
	err = dec.Decode(newEt1)
	if err != nil {
		t.Fatal("error decoding ET1:", err)
	}

	if !reflect.DeepEqual(et1, newEt1) {
		t.Fatalf("invalid data for et1: expected %+v; got %+v", *et1, *newEt1)
	}
	if b.Len() != 0 {
		t.Error("not at eof;", b.Len(), "bytes left")
	}

	enc.Encode(et1)
	newEt1 = new(ET1)
	err = dec.Decode(newEt1)
	if err != nil {
		t.Fatal("round 2: error decoding ET1:", err)
	}
	if !reflect.DeepEqual(et1, newEt1) {
		t.Fatalf("round 2: invalid data for et1: expected %+v; got %+v", *et1, *newEt1)
	}
	if b.Len() != 0 {
		t.Error("round 2: not at eof;", b.Len(), "bytes left")
	}

	// Now test with a running encoder/decoder pair that we recognize a type mismatch.
	err = enc.Encode(et1)
	if err != nil {
		t.Error("round 3: encoder fail:", err)
	}
	newEt2 := new(ET2)
	err = dec.Decode(newEt2)
	if err == nil {
		t.Fatal("round 3: expected `bad type' error decoding ET2")
	}
}

// Run one value through the encoder/decoder, but use the wrong type.
// Input is always an ET1; we compare it to whatever is under 'e'.
func badTypeCheck(e any, shouldFail bool, msg string, t *testing.T) {
	b := new(bytes.Buffer)
	enc := NewEncoder(b)
	et1 := new(ET1)
	et1.A = 7
	et1.Et2 = new(ET2)
	err := enc.Encode(et1)
	if err != nil {
		t.Error("encoder fail:", err)
	}
	dec := NewDecoder(b)
	err = dec.Decode(e)
	if shouldFail && err == nil {
		t.Error("expected error for", msg)
	}
	if !shouldFail && err != nil {
		t.Error("unexpected error for", msg, err)
	}
}

// Test that we recognize a bad type the first time.
func TestWrongTypeDecoder(t *testing.T) {
	badTypeCheck(new(ET2), true, "no fields in common", t)
	badTypeCheck(new(ET3), false, "different name of field", t)
	badTypeCheck(new(ET4), true, "different type of field", t)
}

// Types not supported at top level by the Encoder.
var unsupportedValues = []any{
	make(chan int),
	func(a int) bool { return true },
}

func TestUnsupported(t *testing.T) {
	var b bytes.Buffer
	enc := NewEncoder(&b)
	for _, v := range unsupportedValues {
		err := enc.Encode(v)
		if err == nil {
			t.Errorf("expected error for %T; got none", v)
		}
	}
}

func encAndDec(in, out any) error {
	b := new(bytes.Buffer)
	enc := NewEncoder(b)
	err := enc.Encode(in)
	if err != nil {
		return err
	}
	dec := NewDecoder(b)
	err = dec.Decode(out)
	if err != nil {
		return err
	}
	return nil
}

func TestTypeToPtrType(t *testing.T) {
	// Encode a T, decode a *T
	type Type0 struct {
		A int
	}
	t0 := Type0{7}
	t0p := new(Type0)
	if err := encAndDec(t0, t0p); err != nil {
		t.Error(err)
	}
}

func TestPtrTypeToType(t *testing.T) {
	// Encode a *T, decode a T
	type Type1 struct {
		A uint
	}
	t1p := &Type1{17}
	var t1 Type1
	if err := encAndDec(t1, t1p); err != nil {
		t.Error(err)
	}
}

func TestTypeToPtrPtrPtrPtrType(t *testing.T) {
	type Type2 struct {
		A ****float64
	}
	t2 := Type2{}
	t2.A = new(***float64)
	*t2.A = new(**float64)
	**t2.A = new(*float64)
	***t2.A = new(float64)
	****t2.A = 27.4
	t2pppp := new(***Type2)
	if err := encAndDec(t2, t2pppp); err != nil {
		t.Fatal(err)
	}
	if ****(****t2pppp).A != ****t2.A {
		t.Errorf("wrong value after decode: %g not %g", ****(****t2pppp).A, ****t2.A)
	}
}

func TestSlice(t *testing.T) {
	type Type3 struct {
		A []string
	}
	t3p := &Type3{[]string{"hello", "world"}}
	var t3 Type3
	if err := encAndDec(t3, t3p); err != nil {
		t.Error(err)
	}
}

func TestValueError(t *testing.T) {
	// Encode a *T, decode a T
	type Type4 struct {
		A int
	}
	t4p := &Type4{3}
	var t4 Type4 // note: not a pointer.
	if err := encAndDec(t4p, t4); err == nil || !strings.Contains(err.Error(), "pointer") {
		t.Error("expected error about pointer; got", err)
	}
}

func TestArray(t *testing.T) {
	type Type5 struct {
		A [3]string
		B [3]byte
	}
	type Type6 struct {
		A [2]string // can't hold t5.a
	}
	t5 := Type5{[3]string{"hello", ",", "world"}, [3]byte{1, 2, 3}}
	var t5p Type5
	if err := encAndDec(t5, &t5p); err != nil {
		t.Error(err)
	}
	var t6 Type6
	if err := encAndDec(t5, &t6); err == nil {
		t.Error("should fail with mismatched array sizes")
	}
}

func TestRecursiveMapType(t *testing.T) {
	type recursiveMap map[string]recursiveMap
	r1 := recursiveMap{"A": recursiveMap{"B": nil, "C": nil}, "D": nil}
	r2 := make(recursiveMap)
	if err := encAndDec(r1, &r2); err != nil {
		t.Error(err)
	}
}

func TestRecursiveSliceType(t *testing.T) {
	type recursiveSlice []recursiveSlice
	r1 := recursiveSlice{0: recursiveSlice{0: nil}, 1: nil}
	r2 := make(recursiveSlice, 0)
	if err := encAndDec(r1, &r2); err != nil {
		t.Error(err)
	}
}

// Regression test for bug: must send zero values inside arrays
func TestDefaultsInArray(t *testing.T) {
	type Type7 struct {
		B []bool
		I []int
		S []string
		F []float64
	}
	t7 := Type7{
		[]bool{false, false, true},
		[]int{0, 0, 1},
		[]string{"hi", "", "there"},
		[]float64{0, 0, 1},
	}
	var t7p Type7
	if err := encAndDec(t7, &t7p); err != nil {
		t.Error(err)
	}
}

var testInt int
var testFloat32 float32
var testString string
var testSlice []string
var testMap map[string]int
var testArray [7]int

type SingleTest struct {
	in  any
	out any
	err string
}

var singleTests = []SingleTest{
	{17, &testInt, ""},
	{float32(17.5), &testFloat32, ""},
	{"bike shed", &testString, ""},
	{[]string{"bike", "shed", "paint", "color"}, &testSlice, ""},
	{map[string]int{"seven": 7, "twelve": 12}, &testMap, ""},
	{[7]int{4, 55, 0, 0, 0, 0, 0}, &testArray, ""}, // case that once triggered a bug
	{[7]int{4, 55, 1, 44, 22, 66, 1234}, &testArray, ""},

	// Decode errors
	{172, &testFloat32, "type"},
}

func TestSingletons(t *testing.T) {
	b := new(bytes.Buffer)
	enc := NewEncoder(b)
	dec := NewDecoder(b)
	for _, test := range singleTests {
		b.Reset()
		err := enc.Encode(test.in)
		if err != nil {
			t.Errorf("error encoding %v: %s", test.in, err)
			continue
		}
		err = dec.Decode(test.out)
		switch {
		case err != nil && test.err == "":
			t.Errorf("error decoding %v: %s", test.in, err)
			continue
		case err == nil && test.err != "":
			t.Errorf("expected error decoding %v: %s", test.in, test.err)
			continue
		case err != nil && test.err != "":
			if !strings.Contains(err.Error(), test.err) {
				t.Errorf("wrong error decoding %v: wanted %s, got %v", test.in, test.err, err)
			}
			continue
		}
		// Get rid of the pointer in the rhs
		val := reflect.ValueOf(test.out).Elem().Interface()
		if !reflect.DeepEqual(test.in, val) {
			t.Errorf("decoding singleton: expected %v got %v", test.in, val)
		}
	}
}

func TestStructNonStruct(t *testing.T) {
	type Struct struct {
		A string
	}
	type NonStruct string
	s := Struct{"hello"}
	var sp Struct
	if err := encAndDec(s, &sp); err != nil {
		t.Error(err)
	}
	var ns NonStruct
	if err := encAndDec(s, &ns); err == nil {
		t.Error("should get error for struct/non-struct")
	} else if !strings.Contains(err.Error(), "type") {
		t.Error("for struct/non-struct expected type error; got", err)
	}
	// Now try the other way
	var nsp NonStruct
	if err := encAndDec(ns, &nsp); err != nil {
		t.Error(err)
	}
	if err := encAndDec(ns, &s); err == nil {
		t.Error("should get error for non-struct/struct")
	} else if !strings.Contains(err.Error(), "type") {
		t.Error("for non-struct/struct expected type error; got", err)
	}
}

type interfaceIndirectTestI interface {
	F() bool
}

type interfaceIndirectTestT struct{}

func (this *interfaceIndirectTestT) F() bool {
	return true
}

// A version of a bug reported on golang-nuts. Also tests top-level
// slice of interfaces. The issue was registering *T caused T to be
// stored as the concrete type.
func TestInterfaceIndirect(t *testing.T) {
	Register(&interfaceIndirectTestT{})
	b := new(bytes.Buffer)
	w := []interfaceIndirectTestI{&interfaceIndirectTestT{}}
	err := NewEncoder(b).Encode(w)
	if err != nil {
		t.Fatal("encode error:", err)
	}

	var r []interfaceIndirectTestI
	err = NewDecoder(b).Decode(&r)
	if err != nil {
		t.Fatal("decode error:", err)
	}
}

// Now follow various tests that decode into things that can't represent the
// encoded value, all of which should be legal.

// Also, when the ignored object contains an interface value, it may define
// types. Make sure that skipping the value still defines the types by using
// the encoder/decoder pair to send a value afterwards. If an interface
// is sent, its type in the test is always NewType0, so this checks that the
// encoder and decoder don't skew with respect to type definitions.

type Struct0 struct {
	I any
}

type NewType0 struct {
	S string
}

type ignoreTest struct {
	in, out any
}

var ignoreTests = []ignoreTest{
	// Decode normal struct into an empty struct
	{&struct{ A int }{23}, &struct{}{}},
	// Decode normal struct into a nil.
	{&struct{ A int }{23}, nil},
	// Decode singleton string into a nil.
	{"hello, world", nil},
	// Decode singleton slice into a nil.
	{[]int{1, 2, 3, 4}, nil},
	// Decode struct containing an interface into a nil.
	{&Struct0{&NewType0{"value0"}}, nil},
	// Decode singleton slice of interfaces into a nil.
	{[]any{"hi", &NewType0{"value1"}, 23}, nil},
}

func TestDecodeIntoNothing(t *testing.T) {
	Register(new(NewType0))
	for i, test := range ignoreTests {
		b := new(bytes.Buffer)
		enc := NewEncoder(b)
		err := enc.Encode(test.in)
		if err != nil {
			t.Errorf("%d: encode error %s:", i, err)
			continue
		}
		dec := NewDecoder(b)
		err = dec.Decode(test.out)
		if err != nil {
			t.Errorf("%d: decode error: %s", i, err)
			continue
		}
		// Now see if the encoder and decoder are in a consistent state.
		str := fmt.Sprintf("Value %d", i)
		err = enc.Encode(&NewType0{str})
		if err != nil {
			t.Fatalf("%d: NewType0 encode error: %s", i, err)
		}
		ns := new(NewType0)
		err = dec.Decode(ns)
		if err != nil {
			t.Fatalf("%d: NewType0 decode error: %s", i, err)
		}
		if ns.S != str {
			t.Fatalf("%d: expected %q got %q", i, str, ns.S)
		}
	}
}

func TestIgnoreRecursiveType(t *testing.T) {
	// It's hard to build a self-contained test for this because
	// we can't build compatible types in one package with
	// different items so something is ignored. Here is
	// some data that represents, according to debug.go:
	// type definition {
	//	slice "recursiveSlice" id=106
	//		elem id=106
	// }
	data := []byte{
		0x1d, 0xff, 0xd3, 0x02, 0x01, 0x01, 0x0e, 0x72,
		0x65, 0x63, 0x75, 0x72, 0x73, 0x69, 0x76, 0x65,
		0x53, 0x6c, 0x69, 0x63, 0x65, 0x01, 0xff, 0xd4,
		0x00, 0x01, 0xff, 0xd4, 0x00, 0x00, 0x07, 0xff,
		0xd4, 0x00, 0x02, 0x01, 0x00, 0x00,
	}
	dec := NewDecoder(bytes.NewReader(data))
	// Issue 10415: This caused infinite recursion.
	err := dec.Decode(nil)
	if err != nil {
		t.Fatal(err)
	}
}

// Another bug from golang-nuts, involving nested interfaces.
type Bug0Outer struct {
	Bug0Field any
}

type Bug0Inner struct {
	A int
}

func TestNestedInterfaces(t *testing.T) {
	var buf bytes.Buffer
	e := NewEncoder(&buf)
	d := NewDecoder(&buf)
	Register(new(Bug0Outer))
	Register(new(Bug0Inner))
	f := &Bug0Outer{&Bug0Outer{&Bug0Inner{7}}}
	var v any = f
	err := e.Encode(&v)
	if err != nil {
		t.Fatal("Encode:", err)
	}
	err = d.Decode(&v)
	if err != nil {
		t.Fatal("Decode:", err)
	}
	// Make sure it decoded correctly.
	outer1, ok := v.(*Bug0Outer)
	if !ok {
		t.Fatalf("v not Bug0Outer: %T", v)
	}
	outer2, ok := outer1.Bug0Field.(*Bug0Outer)
	if !ok {
		t.Fatalf("v.Bug0Field not Bug0Outer: %T", outer1.Bug0Field)
	}
	inner, ok := outer2.Bug0Field.(*Bug0Inner)
	if !ok {
		t.Fatalf("v.Bug0Field.Bug0Field not Bug0Inner: %T", outer2.Bug0Field)
	}
	if inner.A != 7 {
		t.Fatalf("final value %d; expected %d", inner.A, 7)
	}
}

// The bugs keep coming. We forgot to send map subtypes before the map.

type Bug1Elem struct {
	Name string
	Id   int
}

type Bug1StructMap map[string]Bug1Elem

func TestMapBug1(t *testing.T) {
	in := make(Bug1StructMap)
	in["val1"] = Bug1Elem{"elem1", 1}
	in["val2"] = Bug1Elem{"elem2", 2}

	b := new(bytes.Buffer)
	enc := NewEncoder(b)
	err := enc.Encode(in)
	if err != nil {
		t.Fatal("encode:", err)
	}
	dec := NewDecoder(b)
	out := make(Bug1StructMap)
	err = dec.Decode(&out)
	if err != nil {
		t.Fatal("decode:", err)
	}
	if !reflect.DeepEqual(in, out) {
		t.Errorf("mismatch: %v %v", in, out)
	}
}

func TestGobMapInterfaceEncode(t *testing.T) {
	m := map[string]any{
		"up": uintptr(0),
		"i0": []int{-1},
		"i1": []int8{-1},
		"i2": []int16{-1},
		"i3": []int32{-1},
		"i4": []int64{-1},
		"u0": []uint{1},
		"u1": []uint8{1},
		"u2": []uint16{1},
		"u3": []uint32{1},
		"u4": []uint64{1},
		"f0": []float32{1},
		"f1": []float64{1},
		"c0": []complex64{complex(2, -2)},
		"c1": []complex128{complex(2, float64(-2))},
		"us": []uintptr{0},
		"bo": []bool{false},
		"st": []string{"s"},
	}
	enc := NewEncoder(new(bytes.Buffer))
	err := enc.Encode(m)
	if err != nil {
		t.Errorf("encode map: %s", err)
	}
}

func TestSliceReusesMemory(t *testing.T) {
	buf := new(bytes.Buffer)
	// Bytes
	{
		x := []byte("abcd")
		enc := NewEncoder(buf)
		err := enc.Encode(x)
		if err != nil {
			t.Errorf("bytes: encode: %s", err)
		}
		// Decode into y, which is big enough.
		y := []byte("ABCDE")
		addr := &y[0]
		dec := NewDecoder(buf)
		err = dec.Decode(&y)
		if err != nil {
			t.Fatal("bytes: decode:", err)
		}
		if !bytes.Equal(x, y) {
			t.Errorf("bytes: expected %q got %q\n", x, y)
		}
		if addr != &y[0] {
			t.Errorf("bytes: unnecessary reallocation")
		}
	}
	// general slice
	{
		x := []rune("abcd")
		enc := NewEncoder(buf)
		err := enc.Encode(x)
		if err != nil {
			t.Errorf("ints: encode: %s", err)
		}
		// Decode into y, which is big enough.
		y := []rune("ABCDE")
		addr := &y[0]
		dec := NewDecoder(buf)
		err = dec.Decode(&y)
		if err != nil {
			t.Fatal("ints: decode:", err)
		}
		if !reflect.DeepEqual(x, y) {
			t.Errorf("ints: expected %q got %q\n", x, y)
		}
		if addr != &y[0] {
			t.Errorf("ints: unnecessary reallocation")
		}
	}
}

// Used to crash: negative count in recvMessage.
func TestBadCount(t *testing.T) {
	b := []byte{0xfb, 0xa5, 0x82, 0x2f, 0xca, 0x1}
	if err := NewDecoder(bytes.NewReader(b)).Decode(nil); err == nil {
		t.Error("expected error from bad count")
	} else if err.Error() != errBadCount.Error() {
		t.Error("expected bad count error; got", err)
	}
}

// Verify that sequential Decoders built on a single input will
// succeed if the input implements ReadByte and there is no
// type information in the stream.
func TestSequentialDecoder(t *testing.T) {
	b := new(bytes.Buffer)
	enc := NewEncoder(b)
	const count = 10
	for i := 0; i < count; i++ {
		s := fmt.Sprintf("%d", i)
		if err := enc.Encode(s); err != nil {
			t.Error("encoder fail:", err)
		}
	}
	for i := 0; i < count; i++ {
		dec := NewDecoder(b)
		var s string
		if err := dec.Decode(&s); err != nil {
			t.Fatal("decoder fail:", err)
		}
		if s != fmt.Sprintf("%d", i) {
			t.Fatalf("decode expected %d got %s", i, s)
		}
	}
}

// Should be able to have unrepresentable fields (chan, func, *chan etc.); we just ignore them.
type Bug2 struct {
	A   int
	C   chan int
	CP  *chan int
	F   func()
	FPP **func()
}

func TestChanFuncIgnored(t *testing.T) {
	c := make(chan int)
	f := func() {}
	fp := &f
	b0 := Bug2{23, c, &c, f, &fp}
	var buf bytes.Buffer
	enc := NewEncoder(&buf)
	if err := enc.Encode(b0); err != nil {
		t.Fatal("error encoding:", err)
	}
	var b1 Bug2
	err := NewDecoder(&buf).Decode(&b1)
	if err != nil {
		t.Fatal("decode:", err)
	}
	if b1.A != b0.A {
		t.Fatalf("got %d want %d", b1.A, b0.A)
	}
	if b1.C != nil || b1.CP != nil || b1.F != nil || b1.FPP != nil {
		t.Fatal("unexpected value for chan or func")
	}
}

func TestSliceIncompatibility(t *testing.T) {
	var in = []byte{1, 2, 3}
	var out []int
	if err := encAndDec(in, &out); err == nil {
		t.Error("expected compatibility error")
	}
}

// Mutually recursive slices of structs caused problems.
type Bug3 struct {
	Num      int
	Children []*Bug3
}

func TestGobPtrSlices(t *testing.T) {
	in := []*Bug3{
		{1, nil},
		{2, nil},
	}
	b := new(bytes.Buffer)
	err := NewEncoder(b).Encode(&in)
	if err != nil {
		t.Fatal("encode:", err)
	}

	var out []*Bug3
	err = NewDecoder(b).Decode(&out)
	if err != nil {
		t.Fatal("decode:", err)
	}
	if !reflect.DeepEqual(in, out) {
		t.Fatalf("got %v; wanted %v", out, in)
	}
}

// getDecEnginePtr cached engine for ut.base instead of ut.user so we passed
// a *map and then tried to reuse its engine to decode the inner map.
func TestPtrToMapOfMap(t *testing.T) {
	Register(make(map[string]any))
	subdata := make(map[string]any)
	subdata["bar"] = "baz"
	data := make(map[string]any)
	data["foo"] = subdata

	b := new(bytes.Buffer)
	err := NewEncoder(b).Encode(data)
	if err != nil {
		t.Fatal("encode:", err)
	}
	var newData map[string]any
	err = NewDecoder(b).Decode(&newData)
	if err != nil {
		t.Fatal("decode:", err)
	}
	if !reflect.DeepEqual(data, newData) {
		t.Fatalf("expected %v got %v", data, newData)
	}
}

// Test that untyped nils generate an error, not a panic.
// See Issue 16204.
func TestCatchInvalidNilValue(t *testing.T) {
	encodeErr, panicErr := encodeAndRecover(nil)
	if panicErr != nil {
		t.Fatalf("panicErr=%v, should not panic encoding untyped nil", panicErr)
	}
	if encodeErr == nil {
		t.Errorf("got err=nil, want non-nil error when encoding untyped nil value")
	} else if !strings.Contains(encodeErr.Error(), "nil value") {
		t.Errorf("expected 'nil value' error; got err=%v", encodeErr)
	}
}

// A top-level nil pointer generates a panic with a helpful string-valued message.
func TestTopLevelNilPointer(t *testing.T) {
	var ip *int
	encodeErr, panicErr := encodeAndRecover(ip)
	if encodeErr != nil {
		t.Fatal("error in encode:", encodeErr)
	}
	if panicErr == nil {
		t.Fatal("top-level nil pointer did not panic")
	}
	errMsg := panicErr.Error()
	if !strings.Contains(errMsg, "nil pointer") {
		t.Fatal("expected nil pointer error, got:", errMsg)
	}
}

func encodeAndRecover(value any) (encodeErr, panicErr error) {
	defer func() {
		e := recover()
		if e != nil {
			switch err := e.(type) {
			case error:
				panicErr = err
			default:
				panicErr = fmt.Errorf("%v", err)
			}
		}
	}()

	encodeErr = NewEncoder(io.Discard).Encode(value)
	return
}

func TestNilPointerPanics(t *testing.T) {
	var (
		nilStringPtr      *string
		intMap            = make(map[int]int)
		intMapPtr         = &intMap
		nilIntMapPtr      *map[int]int
		zero              int
		nilBoolChannel    chan bool
		nilBoolChannelPtr *chan bool
		nilStringSlice    []string
		stringSlice       = make([]string, 1)
		nilStringSlicePtr *[]string
	)

	testCases := []struct {
		value     any
		mustPanic bool
	}{
		{nilStringPtr, true},
		{intMap, false},
		{intMapPtr, false},
		{nilIntMapPtr, true},
		{zero, false},
		{nilStringSlice, false},
		{stringSlice, false},
		{nilStringSlicePtr, true},
		{nilBoolChannel, false},
		{nilBoolChannelPtr, true},
	}

	for _, tt := range testCases {
		_, panicErr := encodeAndRecover(tt.value)
		if tt.mustPanic {
			if panicErr == nil {
				t.Errorf("expected panic with input %#v, did not panic", tt.value)
			}
			continue
		}
		if panicErr != nil {
			t.Fatalf("expected no panic with input %#v, got panic=%v", tt.value, panicErr)
		}
	}
}

func TestNilPointerInsideInterface(t *testing.T) {
	var ip *int
	si := struct {
		I any
	}{
		I: ip,
	}
	buf := new(bytes.Buffer)
	err := NewEncoder(buf).Encode(si)
	if err == nil {
		t.Fatal("expected error, got none")
	}
	errMsg := err.Error()
	if !strings.Contains(errMsg, "nil pointer") || !strings.Contains(errMsg, "interface") {
		t.Fatal("expected error about nil pointer and interface, got:", errMsg)
	}
}

type Bug4Public struct {
	Name   string
	Secret Bug4Secret
}

type Bug4Secret struct {
	a int // error: no exported fields.
}

// Test that a failed compilation doesn't leave around an executable encoder.
// Issue 3723.
func TestMultipleEncodingsOfBadType(t *testing.T) {
	x := Bug4Public{
		Name:   "name",
		Secret: Bug4Secret{1},
	}
	buf := new(bytes.Buffer)
	enc := NewEncoder(buf)
	err := enc.Encode(x)
	if err == nil {
		t.Fatal("first encoding: expected error")
	}
	buf.Reset()
	enc = NewEncoder(buf)
	err = enc.Encode(x)
	if err == nil {
		t.Fatal("second encoding: expected error")
	}
	if !strings.Contains(err.Error(), "no exported fields") {
		t.Errorf("expected error about no exported fields; got %v", err)
	}
}

// There was an error check comparing the length of the input with the
// length of the slice being decoded. It was wrong because the next
// thing in the input might be a type definition, which would lead to
// an incorrect length check. This test reproduces the corner case.

type Z struct {
}

func Test29ElementSlice(t *testing.T) {
	Register(Z{})
	src := make([]any, 100) // Size needs to be bigger than size of type definition.
	for i := range src {
		src[i] = Z{}
	}
	buf := new(bytes.Buffer)
	err := NewEncoder(buf).Encode(src)
	if err != nil {
		t.Fatalf("encode: %v", err)
		return
	}

	var dst []any
	err = NewDecoder(buf).Decode(&dst)
	if err != nil {
		t.Errorf("decode: %v", err)
		return
	}
}

// Don't crash, just give error when allocating a huge slice.
// Issue 8084.
func TestErrorForHugeSlice(t *testing.T) {
	// Encode an int slice.
	buf := new(bytes.Buffer)
	slice := []int{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
	err := NewEncoder(buf).Encode(slice)
	if err != nil {
		t.Fatal("encode:", err)
	}
	// Reach into the buffer and smash the count to make the encoded slice very long.
	buf.Bytes()[buf.Len()-len(slice)-1] = 0xfa
	// Decode and see error.
	err = NewDecoder(buf).Decode(&slice)
	if err == nil {
		t.Fatal("decode: no error")
	}
	if !strings.Contains(err.Error(), "slice too big") {
		t.Fatalf("decode: expected slice too big error, got %s", err.Error())
	}
}

type badDataTest struct {
	input string // The input encoded as a hex string.
	error string // A substring of the error that should result.
	data  any    // What to decode into.
}

var badDataTests = []badDataTest{
	{"", "EOF", nil},
	{"7F6869", "unexpected EOF", nil},
	{"036e6f77206973207468652074696d6520666f7220616c6c20676f6f64206d656e", "unknown type id", new(ET2)},
	{"0424666f6f", "field numbers out of bounds", new(ET2)}, // Issue 6323.
	{"05100028557b02027f8302", "interface encoding", nil},   // Issue 10270.
	// Issue 10273.
	{"130a00fb5dad0bf8ff020263e70002fa28020202a89859", "slice length too large", nil},
	{"0f1000fb285d003316020735ff023a65c5", "interface encoding", nil},
	{"03fffb0616fffc00f902ff02ff03bf005d02885802a311a8120228022c028ee7", "GobDecoder", nil},
	// Issue 10491.
	{"10fe010f020102fe01100001fe010e000016fe010d030102fe010e00010101015801fe01100000000bfe011000f85555555555555555", "exceeds input size", nil},
}

// TestBadData tests that various problems caused by malformed input
// are caught as errors and do not cause panics.
func TestBadData(t *testing.T) {
	for i, test := range badDataTests {
		data, err := hex.DecodeString(test.input)
		if err != nil {
			t.Fatalf("#%d: hex error: %s", i, err)
		}
		d := NewDecoder(bytes.NewReader(data))
		err = d.Decode(test.data)
		if err == nil {
			t.Errorf("decode: no error")
			continue
		}
		if !strings.Contains(err.Error(), test.error) {
			t.Errorf("#%d: decode: expected %q error, got %s", i, test.error, err.Error())
		}
	}
}

func TestDecodeErrorMultipleTypes(t *testing.T) {
	type Test struct {
		A string
		B int
	}
	var b bytes.Buffer
	NewEncoder(&b).Encode(Test{"one", 1})

	var result, result2 Test
	dec := NewDecoder(&b)
	err := dec.Decode(&result)
	if err != nil {
		t.Errorf("decode: unexpected error %v", err)
	}

	b.Reset()
	NewEncoder(&b).Encode(Test{"two", 2})
	err = dec.Decode(&result2)
	if err == nil {
		t.Errorf("decode: expected duplicate type error, got nil")
	} else if !strings.Contains(err.Error(), "duplicate type") {
		t.Errorf("decode: expected duplicate type error, got %s", err.Error())
	}
}

// Issue 24075
func TestMarshalFloatMap(t *testing.T) {
	nan1 := math.NaN()
	nan2 := math.Float64frombits(math.Float64bits(nan1) ^ 1) // A different NaN in the same class.

	in := map[float64]string{
		nan1: "a",
		nan1: "b",
		nan2: "c",
	}

	var b bytes.Buffer
	enc := NewEncoder(&b)
	if err := enc.Encode(in); err != nil {
		t.Errorf("Encode : %v", err)
	}

	out := map[float64]string{}
	dec := NewDecoder(&b)
	if err := dec.Decode(&out); err != nil {
		t.Fatalf("Decode : %v", err)
	}

	type mapEntry struct {
		keyBits uint64
		value   string
	}
	readMap := func(m map[float64]string) (entries []mapEntry) {
		for k, v := range m {
			entries = append(entries, mapEntry{math.Float64bits(k), v})
		}
		slices.SortFunc(entries, func { a, b ->
			r := cmp.Compare(a.keyBits, b.keyBits)
			if r != 0 {
				return r
			}
			return cmp.Compare(a.value, b.value)
		})
		return entries
	}

	got := readMap(out)
	want := readMap(in)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("\nEncode: %v\nDecode: %v", want, got)
	}
}

func TestDecodePartial(t *testing.T) {
	type T struct {
		X []int
		Y string
	}

	var buf bytes.Buffer
	t1 := T{X: []int{1, 2, 3}, Y: "foo"}
	t2 := T{X: []int{4, 5, 6}, Y: "bar"}
	enc := NewEncoder(&buf)

	t1start := 0
	if err := enc.Encode(&t1); err != nil {
		t.Fatal(err)
	}

	t2start := buf.Len()
	if err := enc.Encode(&t2); err != nil {
		t.Fatal(err)
	}

	data := buf.Bytes()
	for i := 0; i <= len(data); i++ {
		bufr := bytes.NewReader(data[:i])

		// Decode both values, stopping at the first error.
		var t1b, t2b T
		dec := NewDecoder(bufr)
		var err error
		err = dec.Decode(&t1b)
		if err == nil {
			err = dec.Decode(&t2b)
		}

		switch i {
		case t1start, t2start:
			// Either the first or the second Decode calls had zero input.
			if err != io.EOF {
				t.Errorf("%d/%d: expected io.EOF: %v", i, len(data), err)
			}
		case len(data):
			// We reached the end of the entire input.
			if err != nil {
				t.Errorf("%d/%d: unexpected error: %v", i, len(data), err)
			}
			if !reflect.DeepEqual(t1b, t1) {
				t.Fatalf("t1 value mismatch: got %v, want %v", t1b, t1)
			}
			if !reflect.DeepEqual(t2b, t2) {
				t.Fatalf("t2 value mismatch: got %v, want %v", t2b, t2)
			}
		default:
			// In between, we must see io.ErrUnexpectedEOF.
			// The decoder used to erroneously return io.EOF in some cases here,
			// such as if the input was cut off right after some type specs,
			// but before any value was actually transmitted.
			if err != io.ErrUnexpectedEOF {
				t.Errorf("%d/%d: expected io.ErrUnexpectedEOF: %v", i, len(data), err)
			}
		}
	}
}

func TestDecoderOverflow(t *testing.T) {
	// Issue 55337.
	dec := NewDecoder(bytes.NewReader([]byte{
		0x12, 0xff, 0xff, 0x2, 0x2, 0x20, 0x0, 0xf8, 0x7f, 0xff, 0xff, 0xff,
		0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x20, 0x20, 0x20, 0x20, 0x20,
	}))
	var r interface{}
	err := dec.Decode(r)
	if err == nil {
		t.Fatalf("expected an error")
	}
}
