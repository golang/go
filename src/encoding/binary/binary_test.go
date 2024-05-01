// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package binary

import (
	"bytes"
	"fmt"
	"io"
	"math"
	"reflect"
	"strings"
	"sync"
	"testing"
	"unsafe"
)

type Struct struct {
	Int8       int8
	Int16      int16
	Int32      int32
	Int64      int64
	Uint8      uint8
	Uint16     uint16
	Uint32     uint32
	Uint64     uint64
	Float32    float32
	Float64    float64
	Complex64  complex64
	Complex128 complex128
	Array      [4]uint8
	Bool       bool
	BoolArray  [4]bool
}

type T struct {
	Int     int
	Uint    uint
	Uintptr uintptr
	Array   [4]int
}

var s = Struct{
	0x01,
	0x0203,
	0x04050607,
	0x08090a0b0c0d0e0f,
	0x10,
	0x1112,
	0x13141516,
	0x1718191a1b1c1d1e,

	math.Float32frombits(0x1f202122),
	math.Float64frombits(0x232425262728292a),
	complex(
		math.Float32frombits(0x2b2c2d2e),
		math.Float32frombits(0x2f303132),
	),
	complex(
		math.Float64frombits(0x333435363738393a),
		math.Float64frombits(0x3b3c3d3e3f404142),
	),

	[4]uint8{0x43, 0x44, 0x45, 0x46},

	true,
	[4]bool{true, false, true, false},
}

var big = []byte{
	1,
	2, 3,
	4, 5, 6, 7,
	8, 9, 10, 11, 12, 13, 14, 15,
	16,
	17, 18,
	19, 20, 21, 22,
	23, 24, 25, 26, 27, 28, 29, 30,

	31, 32, 33, 34,
	35, 36, 37, 38, 39, 40, 41, 42,
	43, 44, 45, 46, 47, 48, 49, 50,
	51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,

	67, 68, 69, 70,

	1,
	1, 0, 1, 0,
}

var little = []byte{
	1,
	3, 2,
	7, 6, 5, 4,
	15, 14, 13, 12, 11, 10, 9, 8,
	16,
	18, 17,
	22, 21, 20, 19,
	30, 29, 28, 27, 26, 25, 24, 23,

	34, 33, 32, 31,
	42, 41, 40, 39, 38, 37, 36, 35,
	46, 45, 44, 43, 50, 49, 48, 47,
	58, 57, 56, 55, 54, 53, 52, 51, 66, 65, 64, 63, 62, 61, 60, 59,

	67, 68, 69, 70,

	1,
	1, 0, 1, 0,
}

var src = []byte{1, 2, 3, 4, 5, 6, 7, 8}
var res = []int32{0x01020304, 0x05060708}
var putbuf = []byte{0, 0, 0, 0, 0, 0, 0, 0}

func checkResult(t *testing.T, dir string, order ByteOrder, err error, have, want any) {
	if err != nil {
		t.Errorf("%v %v: %v", dir, order, err)
		return
	}
	if !reflect.DeepEqual(have, want) {
		t.Errorf("%v %v:\n\thave %+v\n\twant %+v", dir, order, have, want)
	}
}

func testRead(t *testing.T, order ByteOrder, b []byte, s1 any) {
	var s2 Struct
	err := Read(bytes.NewReader(b), order, &s2)
	checkResult(t, "Read", order, err, s2, s1)
}

func testWrite(t *testing.T, order ByteOrder, b []byte, s1 any) {
	buf := new(bytes.Buffer)
	err := Write(buf, order, s1)
	checkResult(t, "Write", order, err, buf.Bytes(), b)
}

func TestLittleEndianRead(t *testing.T)     { testRead(t, LittleEndian, little, s) }
func TestLittleEndianWrite(t *testing.T)    { testWrite(t, LittleEndian, little, s) }
func TestLittleEndianPtrWrite(t *testing.T) { testWrite(t, LittleEndian, little, &s) }

func TestBigEndianRead(t *testing.T)     { testRead(t, BigEndian, big, s) }
func TestBigEndianWrite(t *testing.T)    { testWrite(t, BigEndian, big, s) }
func TestBigEndianPtrWrite(t *testing.T) { testWrite(t, BigEndian, big, &s) }

func TestReadSlice(t *testing.T) {
	slice := make([]int32, 2)
	err := Read(bytes.NewReader(src), BigEndian, slice)
	checkResult(t, "ReadSlice", BigEndian, err, slice, res)
}

func TestWriteSlice(t *testing.T) {
	buf := new(bytes.Buffer)
	err := Write(buf, BigEndian, res)
	checkResult(t, "WriteSlice", BigEndian, err, buf.Bytes(), src)
}

func TestReadBool(t *testing.T) {
	var res bool
	var err error
	err = Read(bytes.NewReader([]byte{0}), BigEndian, &res)
	checkResult(t, "ReadBool", BigEndian, err, res, false)
	res = false
	err = Read(bytes.NewReader([]byte{1}), BigEndian, &res)
	checkResult(t, "ReadBool", BigEndian, err, res, true)
	res = false
	err = Read(bytes.NewReader([]byte{2}), BigEndian, &res)
	checkResult(t, "ReadBool", BigEndian, err, res, true)
}

func TestReadBoolSlice(t *testing.T) {
	slice := make([]bool, 4)
	err := Read(bytes.NewReader([]byte{0, 1, 2, 255}), BigEndian, slice)
	checkResult(t, "ReadBoolSlice", BigEndian, err, slice, []bool{false, true, true, true})
}

// Addresses of arrays are easier to manipulate with reflection than are slices.
var intArrays = []any{
	&[100]int8{},
	&[100]int16{},
	&[100]int32{},
	&[100]int64{},
	&[100]uint8{},
	&[100]uint16{},
	&[100]uint32{},
	&[100]uint64{},
}

func TestSliceRoundTrip(t *testing.T) {
	buf := new(bytes.Buffer)
	for _, array := range intArrays {
		src := reflect.ValueOf(array).Elem()
		unsigned := false
		switch src.Index(0).Kind() {
		case reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
			unsigned = true
		}
		for i := 0; i < src.Len(); i++ {
			if unsigned {
				src.Index(i).SetUint(uint64(i * 0x07654321))
			} else {
				src.Index(i).SetInt(int64(i * 0x07654321))
			}
		}
		buf.Reset()
		srcSlice := src.Slice(0, src.Len())
		err := Write(buf, BigEndian, srcSlice.Interface())
		if err != nil {
			t.Fatal(err)
		}
		dst := reflect.New(src.Type()).Elem()
		dstSlice := dst.Slice(0, dst.Len())
		err = Read(buf, BigEndian, dstSlice.Interface())
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(src.Interface(), dst.Interface()) {
			t.Fatal(src)
		}
	}
}

func TestWriteT(t *testing.T) {
	buf := new(bytes.Buffer)
	ts := T{}
	if err := Write(buf, BigEndian, ts); err == nil {
		t.Errorf("WriteT: have err == nil, want non-nil")
	}

	tv := reflect.Indirect(reflect.ValueOf(ts))
	for i, n := 0, tv.NumField(); i < n; i++ {
		typ := tv.Field(i).Type().String()
		if typ == "[4]int" {
			typ = "int" // the problem is int, not the [4]
		}
		if err := Write(buf, BigEndian, tv.Field(i).Interface()); err == nil {
			t.Errorf("WriteT.%v: have err == nil, want non-nil", tv.Field(i).Type())
		} else if !strings.Contains(err.Error(), typ) {
			t.Errorf("WriteT: have err == %q, want it to mention %s", err, typ)
		}
	}
}

type BlankFields struct {
	A uint32
	_ int32
	B float64
	_ [4]int16
	C byte
	_ [7]byte
	_ struct {
		f [8]float32
	}
}

type BlankFieldsProbe struct {
	A  uint32
	P0 int32
	B  float64
	P1 [4]int16
	C  byte
	P2 [7]byte
	P3 struct {
		F [8]float32
	}
}

func TestBlankFields(t *testing.T) {
	buf := new(bytes.Buffer)
	b1 := BlankFields{A: 1234567890, B: 2.718281828, C: 42}
	if err := Write(buf, LittleEndian, &b1); err != nil {
		t.Error(err)
	}

	// zero values must have been written for blank fields
	var p BlankFieldsProbe
	if err := Read(buf, LittleEndian, &p); err != nil {
		t.Error(err)
	}

	// quick test: only check first value of slices
	if p.P0 != 0 || p.P1[0] != 0 || p.P2[0] != 0 || p.P3.F[0] != 0 {
		t.Errorf("non-zero values for originally blank fields: %#v", p)
	}

	// write p and see if we can probe only some fields
	if err := Write(buf, LittleEndian, &p); err != nil {
		t.Error(err)
	}

	// read should ignore blank fields in b2
	var b2 BlankFields
	if err := Read(buf, LittleEndian, &b2); err != nil {
		t.Error(err)
	}
	if b1.A != b2.A || b1.B != b2.B || b1.C != b2.C {
		t.Errorf("%#v != %#v", b1, b2)
	}
}

func TestSizeStructCache(t *testing.T) {
	// Reset the cache, otherwise multiple test runs fail.
	structSize = sync.Map{}

	count := func() int {
		var i int
		structSize.Range(func(_, _ any) bool {
			i++
			return true
		})
		return i
	}

	var total int
	added := func() int {
		delta := count() - total
		total += delta
		return delta
	}

	type foo struct {
		A uint32
	}

	type bar struct {
		A Struct
		B foo
		C Struct
	}

	testcases := []struct {
		val  any
		want int
	}{
		{new(foo), 1},
		{new(bar), 1},
		{new(bar), 0},
		{new(struct{ A Struct }), 1},
		{new(struct{ A Struct }), 0},
	}

	for _, tc := range testcases {
		if Size(tc.val) == -1 {
			t.Fatalf("Can't get the size of %T", tc.val)
		}

		if n := added(); n != tc.want {
			t.Errorf("Sizing %T added %d entries to the cache, want %d", tc.val, n, tc.want)
		}
	}
}

func TestSizeInvalid(t *testing.T) {
	testcases := []any{
		int(0),
		new(int),
		(*int)(nil),
		[1]uint{},
		new([1]uint),
		(*[1]uint)(nil),
		[]int{},
		[]int(nil),
		new([]int),
		(*[]int)(nil),
	}
	for _, tc := range testcases {
		if got := Size(tc); got != -1 {
			t.Errorf("Size(%T) = %d, want -1", tc, got)
		}
	}
}

// An attempt to read into a struct with an unexported field will
// panic. This is probably not the best choice, but at this point
// anything else would be an API change.

type Unexported struct {
	a int32
}

func TestUnexportedRead(t *testing.T) {
	var buf bytes.Buffer
	u1 := Unexported{a: 1}
	if err := Write(&buf, LittleEndian, &u1); err != nil {
		t.Fatal(err)
	}

	defer func() {
		if recover() == nil {
			t.Fatal("did not panic")
		}
	}()
	var u2 Unexported
	Read(&buf, LittleEndian, &u2)
}

func TestReadErrorMsg(t *testing.T) {
	var buf bytes.Buffer
	read := func(data any) {
		err := Read(&buf, LittleEndian, data)
		want := "binary.Read: invalid type " + reflect.TypeOf(data).String()
		if err == nil {
			t.Errorf("%T: got no error; want %q", data, want)
			return
		}
		if got := err.Error(); got != want {
			t.Errorf("%T: got %q; want %q", data, got, want)
		}
	}
	read(0)
	s := new(struct{})
	read(&s)
	p := &s
	read(&p)
}

func TestReadTruncated(t *testing.T) {
	const data = "0123456789abcdef"

	var b1 = make([]int32, 4)
	var b2 struct {
		A, B, C, D byte
		E          int32
		F          float64
	}

	for i := 0; i <= len(data); i++ {
		var errWant error
		switch i {
		case 0:
			errWant = io.EOF
		case len(data):
			errWant = nil
		default:
			errWant = io.ErrUnexpectedEOF
		}

		if err := Read(strings.NewReader(data[:i]), LittleEndian, &b1); err != errWant {
			t.Errorf("Read(%d) with slice: got %v, want %v", i, err, errWant)
		}
		if err := Read(strings.NewReader(data[:i]), LittleEndian, &b2); err != errWant {
			t.Errorf("Read(%d) with struct: got %v, want %v", i, err, errWant)
		}
	}
}

func testUint64SmallSliceLengthPanics() (panicked bool) {
	defer func() {
		panicked = recover() != nil
	}()
	b := [8]byte{1, 2, 3, 4, 5, 6, 7, 8}
	LittleEndian.Uint64(b[:4])
	return false
}

func testPutUint64SmallSliceLengthPanics() (panicked bool) {
	defer func() {
		panicked = recover() != nil
	}()
	b := [8]byte{}
	LittleEndian.PutUint64(b[:4], 0x0102030405060708)
	return false
}

func TestByteOrder(t *testing.T) {
	type byteOrder interface {
		ByteOrder
		AppendByteOrder
	}
	buf := make([]byte, 8)
	for _, order := range []byteOrder{LittleEndian, BigEndian} {
		const offset = 3
		for _, value := range []uint64{
			0x0000000000000000,
			0x0123456789abcdef,
			0xfedcba9876543210,
			0xffffffffffffffff,
			0xaaaaaaaaaaaaaaaa,
			math.Float64bits(math.Pi),
			math.Float64bits(math.E),
		} {
			want16 := uint16(value)
			order.PutUint16(buf[:2], want16)
			if got := order.Uint16(buf[:2]); got != want16 {
				t.Errorf("PutUint16: Uint16 = %v, want %v", got, want16)
			}
			buf = order.AppendUint16(buf[:offset], want16)
			if got := order.Uint16(buf[offset:]); got != want16 {
				t.Errorf("AppendUint16: Uint16 = %v, want %v", got, want16)
			}
			if len(buf) != offset+2 {
				t.Errorf("AppendUint16: len(buf) = %d, want %d", len(buf), offset+2)
			}

			want32 := uint32(value)
			order.PutUint32(buf[:4], want32)
			if got := order.Uint32(buf[:4]); got != want32 {
				t.Errorf("PutUint32: Uint32 = %v, want %v", got, want32)
			}
			buf = order.AppendUint32(buf[:offset], want32)
			if got := order.Uint32(buf[offset:]); got != want32 {
				t.Errorf("AppendUint32: Uint32 = %v, want %v", got, want32)
			}
			if len(buf) != offset+4 {
				t.Errorf("AppendUint32: len(buf) = %d, want %d", len(buf), offset+4)
			}

			want64 := uint64(value)
			order.PutUint64(buf[:8], want64)
			if got := order.Uint64(buf[:8]); got != want64 {
				t.Errorf("PutUint64: Uint64 = %v, want %v", got, want64)
			}
			buf = order.AppendUint64(buf[:offset], want64)
			if got := order.Uint64(buf[offset:]); got != want64 {
				t.Errorf("AppendUint64: Uint64 = %v, want %v", got, want64)
			}
			if len(buf) != offset+8 {
				t.Errorf("AppendUint64: len(buf) = %d, want %d", len(buf), offset+8)
			}
		}
	}
}

func TestEarlyBoundsChecks(t *testing.T) {
	if testUint64SmallSliceLengthPanics() != true {
		t.Errorf("binary.LittleEndian.Uint64 expected to panic for small slices, but didn't")
	}
	if testPutUint64SmallSliceLengthPanics() != true {
		t.Errorf("binary.LittleEndian.PutUint64 expected to panic for small slices, but didn't")
	}
}

func TestReadInvalidDestination(t *testing.T) {
	testReadInvalidDestination(t, BigEndian)
	testReadInvalidDestination(t, LittleEndian)
}

func testReadInvalidDestination(t *testing.T, order ByteOrder) {
	destinations := []any{
		int8(0),
		int16(0),
		int32(0),
		int64(0),

		uint8(0),
		uint16(0),
		uint32(0),
		uint64(0),

		bool(false),
	}

	for _, dst := range destinations {
		err := Read(bytes.NewReader([]byte{1, 2, 3, 4, 5, 6, 7, 8}), order, dst)
		want := fmt.Sprintf("binary.Read: invalid type %T", dst)
		if err == nil || err.Error() != want {
			t.Fatalf("for type %T: got %q; want %q", dst, err, want)
		}
	}
}

func TestNoFixedSize(t *testing.T) {
	type Person struct {
		Age    int
		Weight float64
		Height float64
	}

	person := Person{
		Age:    27,
		Weight: 67.3,
		Height: 177.8,
	}

	buf := new(bytes.Buffer)
	err := Write(buf, LittleEndian, &person)
	if err == nil {
		t.Fatal("binary.Write: unexpected success as size of type *binary.Person is not fixed")
	}
	errs := "binary.Write: some values are not fixed-sized in type *binary.Person"
	if err.Error() != errs {
		t.Fatalf("got %q, want %q", err, errs)
	}
}

type byteSliceReader struct {
	remain []byte
}

func (br *byteSliceReader) Read(p []byte) (int, error) {
	n := copy(p, br.remain)
	br.remain = br.remain[n:]
	return n, nil
}

func BenchmarkReadSlice1000Int32s(b *testing.B) {
	bsr := &byteSliceReader{}
	slice := make([]int32, 1000)
	buf := make([]byte, len(slice)*4)
	b.SetBytes(int64(len(buf)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bsr.remain = buf
		Read(bsr, BigEndian, slice)
	}
}

func BenchmarkReadStruct(b *testing.B) {
	bsr := &byteSliceReader{}
	var buf bytes.Buffer
	Write(&buf, BigEndian, &s)
	b.SetBytes(int64(dataSize(reflect.ValueOf(s))))
	t := s
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bsr.remain = buf.Bytes()
		Read(bsr, BigEndian, &t)
	}
	b.StopTimer()
	if b.N > 0 && !reflect.DeepEqual(s, t) {
		b.Fatalf("struct doesn't match:\ngot  %v;\nwant %v", t, s)
	}
}

func BenchmarkWriteStruct(b *testing.B) {
	b.SetBytes(int64(Size(&s)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Write(io.Discard, BigEndian, &s)
	}
}

func BenchmarkWriteSlice1000Structs(b *testing.B) {
	slice := make([]Struct, 1000)
	buf := new(bytes.Buffer)
	var w io.Writer = buf
	b.SetBytes(int64(Size(slice)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buf.Reset()
		Write(w, BigEndian, slice)
	}
	b.StopTimer()
}

func BenchmarkReadSlice1000Structs(b *testing.B) {
	bsr := &byteSliceReader{}
	slice := make([]Struct, 1000)
	buf := make([]byte, Size(slice))
	b.SetBytes(int64(len(buf)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bsr.remain = buf
		Read(bsr, BigEndian, slice)
	}
}

func BenchmarkReadInts(b *testing.B) {
	var ls Struct
	bsr := &byteSliceReader{}
	var r io.Reader = bsr
	b.SetBytes(2 * (1 + 2 + 4 + 8))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bsr.remain = big
		Read(r, BigEndian, &ls.Int8)
		Read(r, BigEndian, &ls.Int16)
		Read(r, BigEndian, &ls.Int32)
		Read(r, BigEndian, &ls.Int64)
		Read(r, BigEndian, &ls.Uint8)
		Read(r, BigEndian, &ls.Uint16)
		Read(r, BigEndian, &ls.Uint32)
		Read(r, BigEndian, &ls.Uint64)
	}
	b.StopTimer()
	want := s
	want.Float32 = 0
	want.Float64 = 0
	want.Complex64 = 0
	want.Complex128 = 0
	want.Array = [4]uint8{0, 0, 0, 0}
	want.Bool = false
	want.BoolArray = [4]bool{false, false, false, false}
	if b.N > 0 && !reflect.DeepEqual(ls, want) {
		b.Fatalf("struct doesn't match:\ngot  %v;\nwant %v", ls, want)
	}
}

func BenchmarkWriteInts(b *testing.B) {
	buf := new(bytes.Buffer)
	var w io.Writer = buf
	b.SetBytes(2 * (1 + 2 + 4 + 8))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buf.Reset()
		Write(w, BigEndian, s.Int8)
		Write(w, BigEndian, s.Int16)
		Write(w, BigEndian, s.Int32)
		Write(w, BigEndian, s.Int64)
		Write(w, BigEndian, s.Uint8)
		Write(w, BigEndian, s.Uint16)
		Write(w, BigEndian, s.Uint32)
		Write(w, BigEndian, s.Uint64)
	}
	b.StopTimer()
	if b.N > 0 && !bytes.Equal(buf.Bytes(), big[:30]) {
		b.Fatalf("first half doesn't match: %x %x", buf.Bytes(), big[:30])
	}
}

func BenchmarkWriteSlice1000Int32s(b *testing.B) {
	slice := make([]int32, 1000)
	buf := new(bytes.Buffer)
	var w io.Writer = buf
	b.SetBytes(4 * 1000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buf.Reset()
		Write(w, BigEndian, slice)
	}
	b.StopTimer()
}

func BenchmarkPutUint16(b *testing.B) {
	b.SetBytes(2)
	for i := 0; i < b.N; i++ {
		BigEndian.PutUint16(putbuf[:2], uint16(i))
	}
}

func BenchmarkAppendUint16(b *testing.B) {
	b.SetBytes(2)
	for i := 0; i < b.N; i++ {
		putbuf = BigEndian.AppendUint16(putbuf[:0], uint16(i))
	}
}

func BenchmarkPutUint32(b *testing.B) {
	b.SetBytes(4)
	for i := 0; i < b.N; i++ {
		BigEndian.PutUint32(putbuf[:4], uint32(i))
	}
}

func BenchmarkAppendUint32(b *testing.B) {
	b.SetBytes(4)
	for i := 0; i < b.N; i++ {
		putbuf = BigEndian.AppendUint32(putbuf[:0], uint32(i))
	}
}

func BenchmarkPutUint64(b *testing.B) {
	b.SetBytes(8)
	for i := 0; i < b.N; i++ {
		BigEndian.PutUint64(putbuf[:8], uint64(i))
	}
}

func BenchmarkAppendUint64(b *testing.B) {
	b.SetBytes(8)
	for i := 0; i < b.N; i++ {
		putbuf = BigEndian.AppendUint64(putbuf[:0], uint64(i))
	}
}

func BenchmarkLittleEndianPutUint16(b *testing.B) {
	b.SetBytes(2)
	for i := 0; i < b.N; i++ {
		LittleEndian.PutUint16(putbuf[:2], uint16(i))
	}
}

func BenchmarkLittleEndianAppendUint16(b *testing.B) {
	b.SetBytes(2)
	for i := 0; i < b.N; i++ {
		putbuf = LittleEndian.AppendUint16(putbuf[:0], uint16(i))
	}
}

func BenchmarkLittleEndianPutUint32(b *testing.B) {
	b.SetBytes(4)
	for i := 0; i < b.N; i++ {
		LittleEndian.PutUint32(putbuf[:4], uint32(i))
	}
}

func BenchmarkLittleEndianAppendUint32(b *testing.B) {
	b.SetBytes(4)
	for i := 0; i < b.N; i++ {
		putbuf = LittleEndian.AppendUint32(putbuf[:0], uint32(i))
	}
}

func BenchmarkLittleEndianPutUint64(b *testing.B) {
	b.SetBytes(8)
	for i := 0; i < b.N; i++ {
		LittleEndian.PutUint64(putbuf[:8], uint64(i))
	}
}

func BenchmarkLittleEndianAppendUint64(b *testing.B) {
	b.SetBytes(8)
	for i := 0; i < b.N; i++ {
		putbuf = LittleEndian.AppendUint64(putbuf[:0], uint64(i))
	}
}

func BenchmarkReadFloats(b *testing.B) {
	var ls Struct
	bsr := &byteSliceReader{}
	var r io.Reader = bsr
	b.SetBytes(4 + 8)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bsr.remain = big[30:]
		Read(r, BigEndian, &ls.Float32)
		Read(r, BigEndian, &ls.Float64)
	}
	b.StopTimer()
	want := s
	want.Int8 = 0
	want.Int16 = 0
	want.Int32 = 0
	want.Int64 = 0
	want.Uint8 = 0
	want.Uint16 = 0
	want.Uint32 = 0
	want.Uint64 = 0
	want.Complex64 = 0
	want.Complex128 = 0
	want.Array = [4]uint8{0, 0, 0, 0}
	want.Bool = false
	want.BoolArray = [4]bool{false, false, false, false}
	if b.N > 0 && !reflect.DeepEqual(ls, want) {
		b.Fatalf("struct doesn't match:\ngot  %v;\nwant %v", ls, want)
	}
}

func BenchmarkWriteFloats(b *testing.B) {
	buf := new(bytes.Buffer)
	var w io.Writer = buf
	b.SetBytes(4 + 8)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buf.Reset()
		Write(w, BigEndian, s.Float32)
		Write(w, BigEndian, s.Float64)
	}
	b.StopTimer()
	if b.N > 0 && !bytes.Equal(buf.Bytes(), big[30:30+4+8]) {
		b.Fatalf("first half doesn't match: %x %x", buf.Bytes(), big[30:30+4+8])
	}
}

func BenchmarkReadSlice1000Float32s(b *testing.B) {
	bsr := &byteSliceReader{}
	slice := make([]float32, 1000)
	buf := make([]byte, len(slice)*4)
	b.SetBytes(int64(len(buf)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bsr.remain = buf
		Read(bsr, BigEndian, slice)
	}
}

func BenchmarkWriteSlice1000Float32s(b *testing.B) {
	slice := make([]float32, 1000)
	buf := new(bytes.Buffer)
	var w io.Writer = buf
	b.SetBytes(4 * 1000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buf.Reset()
		Write(w, BigEndian, slice)
	}
	b.StopTimer()
}

func BenchmarkReadSlice1000Uint8s(b *testing.B) {
	bsr := &byteSliceReader{}
	slice := make([]uint8, 1000)
	buf := make([]byte, len(slice))
	b.SetBytes(int64(len(buf)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bsr.remain = buf
		Read(bsr, BigEndian, slice)
	}
}

func BenchmarkWriteSlice1000Uint8s(b *testing.B) {
	slice := make([]uint8, 1000)
	buf := new(bytes.Buffer)
	var w io.Writer = buf
	b.SetBytes(1000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buf.Reset()
		Write(w, BigEndian, slice)
	}
}

func TestNativeEndian(t *testing.T) {
	const val = 0x12345678
	i := uint32(val)
	s := unsafe.Slice((*byte)(unsafe.Pointer(&i)), unsafe.Sizeof(i))
	if v := NativeEndian.Uint32(s); v != val {
		t.Errorf("NativeEndian.Uint32 returned %#x, expected %#x", v, val)
	}
}
