// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests of the GobEncoder/GobDecoder support.

package gob

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"net"
	"strings"
	"testing"
	"time"
)

// Types that implement the GobEncoder/Decoder interfaces.

type ByteStruct struct {
	a byte // not an exported field
}

type StringStruct struct {
	s string // not an exported field
}

type ArrayStruct struct {
	a [8192]byte // not an exported field
}

type Gobber int

type ValueGobber string // encodes with a value, decodes with a pointer.

type BinaryGobber int

type BinaryValueGobber string

type TextGobber int

type TextValueGobber string

// The relevant methods

func (g *ByteStruct) GobEncode() ([]byte, error) {
	b := make([]byte, 3)
	b[0] = g.a
	b[1] = g.a + 1
	b[2] = g.a + 2
	return b, nil
}

func (g *ByteStruct) GobDecode(data []byte) error {
	if g == nil {
		return errors.New("NIL RECEIVER")
	}
	// Expect N sequential-valued bytes.
	if len(data) == 0 {
		return io.EOF
	}
	g.a = data[0]
	for i, c := range data {
		if c != g.a+byte(i) {
			return errors.New("invalid data sequence")
		}
	}
	return nil
}

func (g *StringStruct) GobEncode() ([]byte, error) {
	return []byte(g.s), nil
}

func (g *StringStruct) GobDecode(data []byte) error {
	// Expect N sequential-valued bytes.
	if len(data) == 0 {
		return io.EOF
	}
	a := data[0]
	for i, c := range data {
		if c != a+byte(i) {
			return errors.New("invalid data sequence")
		}
	}
	g.s = string(data)
	return nil
}

func (a *ArrayStruct) GobEncode() ([]byte, error) {
	return a.a[:], nil
}

func (a *ArrayStruct) GobDecode(data []byte) error {
	if len(data) != len(a.a) {
		return errors.New("wrong length in array decode")
	}
	copy(a.a[:], data)
	return nil
}

func (g *Gobber) GobEncode() ([]byte, error) {
	return []byte(fmt.Sprintf("VALUE=%d", *g)), nil
}

func (g *Gobber) GobDecode(data []byte) error {
	_, err := fmt.Sscanf(string(data), "VALUE=%d", (*int)(g))
	return err
}

func (g *BinaryGobber) MarshalBinary() ([]byte, error) {
	return []byte(fmt.Sprintf("VALUE=%d", *g)), nil
}

func (g *BinaryGobber) UnmarshalBinary(data []byte) error {
	_, err := fmt.Sscanf(string(data), "VALUE=%d", (*int)(g))
	return err
}

func (g *TextGobber) MarshalText() ([]byte, error) {
	return []byte(fmt.Sprintf("VALUE=%d", *g)), nil
}

func (g *TextGobber) UnmarshalText(data []byte) error {
	_, err := fmt.Sscanf(string(data), "VALUE=%d", (*int)(g))
	return err
}

func (v ValueGobber) GobEncode() ([]byte, error) {
	return []byte(fmt.Sprintf("VALUE=%s", v)), nil
}

func (v *ValueGobber) GobDecode(data []byte) error {
	_, err := fmt.Sscanf(string(data), "VALUE=%s", (*string)(v))
	return err
}

func (v BinaryValueGobber) MarshalBinary() ([]byte, error) {
	return []byte(fmt.Sprintf("VALUE=%s", v)), nil
}

func (v *BinaryValueGobber) UnmarshalBinary(data []byte) error {
	_, err := fmt.Sscanf(string(data), "VALUE=%s", (*string)(v))
	return err
}

func (v TextValueGobber) MarshalText() ([]byte, error) {
	return []byte(fmt.Sprintf("VALUE=%s", v)), nil
}

func (v *TextValueGobber) UnmarshalText(data []byte) error {
	_, err := fmt.Sscanf(string(data), "VALUE=%s", (*string)(v))
	return err
}

// Structs that include GobEncodable fields.

type GobTest0 struct {
	X int // guarantee we have  something in common with GobTest*
	G *ByteStruct
}

type GobTest1 struct {
	X int // guarantee we have  something in common with GobTest*
	G *StringStruct
}

type GobTest2 struct {
	X int    // guarantee we have  something in common with GobTest*
	G string // not a GobEncoder - should give us errors
}

type GobTest3 struct {
	X int // guarantee we have  something in common with GobTest*
	G *Gobber
	B *BinaryGobber
	T *TextGobber
}

type GobTest4 struct {
	X  int // guarantee we have  something in common with GobTest*
	V  ValueGobber
	BV BinaryValueGobber
	TV TextValueGobber
}

type GobTest5 struct {
	X  int // guarantee we have  something in common with GobTest*
	V  *ValueGobber
	BV *BinaryValueGobber
	TV *TextValueGobber
}

type GobTest6 struct {
	X  int // guarantee we have  something in common with GobTest*
	V  ValueGobber
	W  *ValueGobber
	BV BinaryValueGobber
	BW *BinaryValueGobber
	TV TextValueGobber
	TW *TextValueGobber
}

type GobTest7 struct {
	X  int // guarantee we have  something in common with GobTest*
	V  *ValueGobber
	W  ValueGobber
	BV *BinaryValueGobber
	BW BinaryValueGobber
	TV *TextValueGobber
	TW TextValueGobber
}

type GobTestIgnoreEncoder struct {
	X int // guarantee we have  something in common with GobTest*
}

type GobTestValueEncDec struct {
	X int          // guarantee we have  something in common with GobTest*
	G StringStruct // not a pointer.
}

type GobTestIndirectEncDec struct {
	X int             // guarantee we have  something in common with GobTest*
	G ***StringStruct // indirections to the receiver.
}

type GobTestArrayEncDec struct {
	X int         // guarantee we have  something in common with GobTest*
	A ArrayStruct // not a pointer.
}

type GobTestIndirectArrayEncDec struct {
	X int            // guarantee we have  something in common with GobTest*
	A ***ArrayStruct // indirections to a large receiver.
}

func TestGobEncoderField(t *testing.T) {
	b := new(bytes.Buffer)
	// First a field that's a structure.
	enc := NewEncoder(b)
	err := enc.Encode(GobTest0{17, &ByteStruct{'A'}})
	if err != nil {
		t.Fatal("encode error:", err)
	}
	dec := NewDecoder(b)
	x := new(GobTest0)
	err = dec.Decode(x)
	if err != nil {
		t.Fatal("decode error:", err)
	}
	if x.G.a != 'A' {
		t.Errorf("expected 'A' got %c", x.G.a)
	}
	// Now a field that's not a structure.
	b.Reset()
	gobber := Gobber(23)
	bgobber := BinaryGobber(24)
	tgobber := TextGobber(25)
	err = enc.Encode(GobTest3{17, &gobber, &bgobber, &tgobber})
	if err != nil {
		t.Fatal("encode error:", err)
	}
	y := new(GobTest3)
	err = dec.Decode(y)
	if err != nil {
		t.Fatal("decode error:", err)
	}
	if *y.G != 23 || *y.B != 24 || *y.T != 25 {
		t.Errorf("expected '23 got %d", *y.G)
	}
}

// Even though the field is a value, we can still take its address
// and should be able to call the methods.
func TestGobEncoderValueField(t *testing.T) {
	b := new(bytes.Buffer)
	// First a field that's a structure.
	enc := NewEncoder(b)
	err := enc.Encode(&GobTestValueEncDec{17, StringStruct{"HIJKL"}})
	if err != nil {
		t.Fatal("encode error:", err)
	}
	dec := NewDecoder(b)
	x := new(GobTestValueEncDec)
	err = dec.Decode(x)
	if err != nil {
		t.Fatal("decode error:", err)
	}
	if x.G.s != "HIJKL" {
		t.Errorf("expected `HIJKL` got %s", x.G.s)
	}
}

// GobEncode/Decode should work even if the value is
// more indirect than the receiver.
func TestGobEncoderIndirectField(t *testing.T) {
	b := new(bytes.Buffer)
	// First a field that's a structure.
	enc := NewEncoder(b)
	s := &StringStruct{"HIJKL"}
	sp := &s
	err := enc.Encode(GobTestIndirectEncDec{17, &sp})
	if err != nil {
		t.Fatal("encode error:", err)
	}
	dec := NewDecoder(b)
	x := new(GobTestIndirectEncDec)
	err = dec.Decode(x)
	if err != nil {
		t.Fatal("decode error:", err)
	}
	if (***x.G).s != "HIJKL" {
		t.Errorf("expected `HIJKL` got %s", (***x.G).s)
	}
}

// Test with a large field with methods.
func TestGobEncoderArrayField(t *testing.T) {
	b := new(bytes.Buffer)
	enc := NewEncoder(b)
	var a GobTestArrayEncDec
	a.X = 17
	for i := range a.A.a {
		a.A.a[i] = byte(i)
	}
	err := enc.Encode(&a)
	if err != nil {
		t.Fatal("encode error:", err)
	}
	dec := NewDecoder(b)
	x := new(GobTestArrayEncDec)
	err = dec.Decode(x)
	if err != nil {
		t.Fatal("decode error:", err)
	}
	for i, v := range x.A.a {
		if v != byte(i) {
			t.Errorf("expected %x got %x", byte(i), v)
			break
		}
	}
}

// Test an indirection to a large field with methods.
func TestGobEncoderIndirectArrayField(t *testing.T) {
	b := new(bytes.Buffer)
	enc := NewEncoder(b)
	var a GobTestIndirectArrayEncDec
	a.X = 17
	var array ArrayStruct
	ap := &array
	app := &ap
	a.A = &app
	for i := range array.a {
		array.a[i] = byte(i)
	}
	err := enc.Encode(a)
	if err != nil {
		t.Fatal("encode error:", err)
	}
	dec := NewDecoder(b)
	x := new(GobTestIndirectArrayEncDec)
	err = dec.Decode(x)
	if err != nil {
		t.Fatal("decode error:", err)
	}
	for i, v := range (***x.A).a {
		if v != byte(i) {
			t.Errorf("expected %x got %x", byte(i), v)
			break
		}
	}
}

// As long as the fields have the same name and implement the
// interface, we can cross-connect them. Not sure it's useful
// and may even be bad but it works and it's hard to prevent
// without exposing the contents of the object, which would
// defeat the purpose.
func TestGobEncoderFieldsOfDifferentType(t *testing.T) {
	// first, string in field to byte in field
	b := new(bytes.Buffer)
	enc := NewEncoder(b)
	err := enc.Encode(GobTest1{17, &StringStruct{"ABC"}})
	if err != nil {
		t.Fatal("encode error:", err)
	}
	dec := NewDecoder(b)
	x := new(GobTest0)
	err = dec.Decode(x)
	if err != nil {
		t.Fatal("decode error:", err)
	}
	if x.G.a != 'A' {
		t.Errorf("expected 'A' got %c", x.G.a)
	}
	// now the other direction, byte in field to string in field
	b.Reset()
	err = enc.Encode(GobTest0{17, &ByteStruct{'X'}})
	if err != nil {
		t.Fatal("encode error:", err)
	}
	y := new(GobTest1)
	err = dec.Decode(y)
	if err != nil {
		t.Fatal("decode error:", err)
	}
	if y.G.s != "XYZ" {
		t.Fatalf("expected `XYZ` got %q", y.G.s)
	}
}

// Test that we can encode a value and decode into a pointer.
func TestGobEncoderValueEncoder(t *testing.T) {
	// first, string in field to byte in field
	b := new(bytes.Buffer)
	enc := NewEncoder(b)
	err := enc.Encode(GobTest4{17, ValueGobber("hello"), BinaryValueGobber("Καλημέρα"), TextValueGobber("こんにちは")})
	if err != nil {
		t.Fatal("encode error:", err)
	}
	dec := NewDecoder(b)
	x := new(GobTest5)
	err = dec.Decode(x)
	if err != nil {
		t.Fatal("decode error:", err)
	}
	if *x.V != "hello" || *x.BV != "Καλημέρα" || *x.TV != "こんにちは" {
		t.Errorf("expected `hello` got %s", *x.V)
	}
}

// Test that we can use a value then a pointer type of a GobEncoder
// in the same encoded value. Bug 4647.
func TestGobEncoderValueThenPointer(t *testing.T) {
	v := ValueGobber("forty-two")
	w := ValueGobber("six-by-nine")
	bv := BinaryValueGobber("1nanocentury")
	bw := BinaryValueGobber("πseconds")
	tv := TextValueGobber("gravitationalacceleration")
	tw := TextValueGobber("π²ft/s²")

	// this was a bug: encoding a GobEncoder by value before a GobEncoder
	// pointer would cause duplicate type definitions to be sent.

	b := new(bytes.Buffer)
	enc := NewEncoder(b)
	if err := enc.Encode(GobTest6{42, v, &w, bv, &bw, tv, &tw}); err != nil {
		t.Fatal("encode error:", err)
	}
	dec := NewDecoder(b)
	x := new(GobTest6)
	if err := dec.Decode(x); err != nil {
		t.Fatal("decode error:", err)
	}

	if got, want := x.V, v; got != want {
		t.Errorf("v = %q, want %q", got, want)
	}
	if got, want := x.W, w; got == nil {
		t.Errorf("w = nil, want %q", want)
	} else if *got != want {
		t.Errorf("w = %q, want %q", *got, want)
	}

	if got, want := x.BV, bv; got != want {
		t.Errorf("bv = %q, want %q", got, want)
	}
	if got, want := x.BW, bw; got == nil {
		t.Errorf("bw = nil, want %q", want)
	} else if *got != want {
		t.Errorf("bw = %q, want %q", *got, want)
	}

	if got, want := x.TV, tv; got != want {
		t.Errorf("tv = %q, want %q", got, want)
	}
	if got, want := x.TW, tw; got == nil {
		t.Errorf("tw = nil, want %q", want)
	} else if *got != want {
		t.Errorf("tw = %q, want %q", *got, want)
	}
}

// Test that we can use a pointer then a value type of a GobEncoder
// in the same encoded value.
func TestGobEncoderPointerThenValue(t *testing.T) {
	v := ValueGobber("forty-two")
	w := ValueGobber("six-by-nine")
	bv := BinaryValueGobber("1nanocentury")
	bw := BinaryValueGobber("πseconds")
	tv := TextValueGobber("gravitationalacceleration")
	tw := TextValueGobber("π²ft/s²")

	b := new(bytes.Buffer)
	enc := NewEncoder(b)
	if err := enc.Encode(GobTest7{42, &v, w, &bv, bw, &tv, tw}); err != nil {
		t.Fatal("encode error:", err)
	}
	dec := NewDecoder(b)
	x := new(GobTest7)
	if err := dec.Decode(x); err != nil {
		t.Fatal("decode error:", err)
	}

	if got, want := x.V, v; got == nil {
		t.Errorf("v = nil, want %q", want)
	} else if *got != want {
		t.Errorf("v = %q, want %q", *got, want)
	}
	if got, want := x.W, w; got != want {
		t.Errorf("w = %q, want %q", got, want)
	}

	if got, want := x.BV, bv; got == nil {
		t.Errorf("bv = nil, want %q", want)
	} else if *got != want {
		t.Errorf("bv = %q, want %q", *got, want)
	}
	if got, want := x.BW, bw; got != want {
		t.Errorf("bw = %q, want %q", got, want)
	}

	if got, want := x.TV, tv; got == nil {
		t.Errorf("tv = nil, want %q", want)
	} else if *got != want {
		t.Errorf("tv = %q, want %q", *got, want)
	}
	if got, want := x.TW, tw; got != want {
		t.Errorf("tw = %q, want %q", got, want)
	}
}

func TestGobEncoderFieldTypeError(t *testing.T) {
	// GobEncoder to non-decoder: error
	b := new(bytes.Buffer)
	enc := NewEncoder(b)
	err := enc.Encode(GobTest1{17, &StringStruct{"ABC"}})
	if err != nil {
		t.Fatal("encode error:", err)
	}
	dec := NewDecoder(b)
	x := &GobTest2{}
	err = dec.Decode(x)
	if err == nil {
		t.Fatal("expected decode error for mismatched fields (encoder to non-decoder)")
	}
	if !strings.Contains(err.Error(), "type") {
		t.Fatal("expected type error; got", err)
	}
	// Non-encoder to GobDecoder: error
	b.Reset()
	err = enc.Encode(GobTest2{17, "ABC"})
	if err != nil {
		t.Fatal("encode error:", err)
	}
	y := &GobTest1{}
	err = dec.Decode(y)
	if err == nil {
		t.Fatal("expected decode error for mismatched fields (non-encoder to decoder)")
	}
	if !strings.Contains(err.Error(), "type") {
		t.Fatal("expected type error; got", err)
	}
}

// Even though ByteStruct is a struct, it's treated as a singleton at the top level.
func TestGobEncoderStructSingleton(t *testing.T) {
	b := new(bytes.Buffer)
	enc := NewEncoder(b)
	err := enc.Encode(&ByteStruct{'A'})
	if err != nil {
		t.Fatal("encode error:", err)
	}
	dec := NewDecoder(b)
	x := new(ByteStruct)
	err = dec.Decode(x)
	if err != nil {
		t.Fatal("decode error:", err)
	}
	if x.a != 'A' {
		t.Errorf("expected 'A' got %c", x.a)
	}
}

func TestGobEncoderNonStructSingleton(t *testing.T) {
	b := new(bytes.Buffer)
	enc := NewEncoder(b)
	var g Gobber = 1234
	err := enc.Encode(&g)
	if err != nil {
		t.Fatal("encode error:", err)
	}
	dec := NewDecoder(b)
	var x Gobber
	err = dec.Decode(&x)
	if err != nil {
		t.Fatal("decode error:", err)
	}
	if x != 1234 {
		t.Errorf("expected 1234 got %d", x)
	}
}

func TestGobEncoderIgnoreStructField(t *testing.T) {
	b := new(bytes.Buffer)
	// First a field that's a structure.
	enc := NewEncoder(b)
	err := enc.Encode(GobTest0{17, &ByteStruct{'A'}})
	if err != nil {
		t.Fatal("encode error:", err)
	}
	dec := NewDecoder(b)
	x := new(GobTestIgnoreEncoder)
	err = dec.Decode(x)
	if err != nil {
		t.Fatal("decode error:", err)
	}
	if x.X != 17 {
		t.Errorf("expected 17 got %c", x.X)
	}
}

func TestGobEncoderIgnoreNonStructField(t *testing.T) {
	b := new(bytes.Buffer)
	// First a field that's a structure.
	enc := NewEncoder(b)
	gobber := Gobber(23)
	bgobber := BinaryGobber(24)
	tgobber := TextGobber(25)
	err := enc.Encode(GobTest3{17, &gobber, &bgobber, &tgobber})
	if err != nil {
		t.Fatal("encode error:", err)
	}
	dec := NewDecoder(b)
	x := new(GobTestIgnoreEncoder)
	err = dec.Decode(x)
	if err != nil {
		t.Fatal("decode error:", err)
	}
	if x.X != 17 {
		t.Errorf("expected 17 got %c", x.X)
	}
}

func TestGobEncoderIgnoreNilEncoder(t *testing.T) {
	b := new(bytes.Buffer)
	// First a field that's a structure.
	enc := NewEncoder(b)
	err := enc.Encode(GobTest0{X: 18}) // G is nil
	if err != nil {
		t.Fatal("encode error:", err)
	}
	dec := NewDecoder(b)
	x := new(GobTest0)
	err = dec.Decode(x)
	if err != nil {
		t.Fatal("decode error:", err)
	}
	if x.X != 18 {
		t.Errorf("expected x.X = 18, got %v", x.X)
	}
	if x.G != nil {
		t.Errorf("expected x.G = nil, got %v", x.G)
	}
}

type gobDecoderBug0 struct {
	foo, bar string
}

func (br *gobDecoderBug0) String() string {
	return br.foo + "-" + br.bar
}

func (br *gobDecoderBug0) GobEncode() ([]byte, error) {
	return []byte(br.String()), nil
}

func (br *gobDecoderBug0) GobDecode(b []byte) error {
	br.foo = "foo"
	br.bar = "bar"
	return nil
}

// This was a bug: the receiver has a different indirection level
// than the variable.
func TestGobEncoderExtraIndirect(t *testing.T) {
	gdb := &gobDecoderBug0{"foo", "bar"}
	buf := new(bytes.Buffer)
	e := NewEncoder(buf)
	if err := e.Encode(gdb); err != nil {
		t.Fatalf("encode: %v", err)
	}
	d := NewDecoder(buf)
	var got *gobDecoderBug0
	if err := d.Decode(&got); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if got.foo != gdb.foo || got.bar != gdb.bar {
		t.Errorf("got = %q, want %q", got, gdb)
	}
}

// Another bug: this caused a crash with the new Go1 Time type.
// We throw in a gob-encoding array, to test another case of isZero,
// and a struct containing a nil interface, to test a third.
type isZeroBug struct {
	T time.Time
	S string
	I int
	A isZeroBugArray
	F isZeroBugInterface
}

type isZeroBugArray [2]uint8

// Receiver is value, not pointer, to test isZero of array.
func (a isZeroBugArray) GobEncode() (b []byte, e error) {
	b = append(b, a[:]...)
	return b, nil
}

func (a *isZeroBugArray) GobDecode(data []byte) error {
	if len(data) != len(a) {
		return io.EOF
	}
	a[0] = data[0]
	a[1] = data[1]
	return nil
}

type isZeroBugInterface struct {
	I any
}

func (i isZeroBugInterface) GobEncode() (b []byte, e error) {
	return []byte{}, nil
}

func (i *isZeroBugInterface) GobDecode(data []byte) error {
	return nil
}

func TestGobEncodeIsZero(t *testing.T) {
	x := isZeroBug{time.Unix(1e9, 0), "hello", -55, isZeroBugArray{1, 2}, isZeroBugInterface{}}
	b := new(bytes.Buffer)
	enc := NewEncoder(b)
	err := enc.Encode(x)
	if err != nil {
		t.Fatal("encode:", err)
	}
	var y isZeroBug
	dec := NewDecoder(b)
	err = dec.Decode(&y)
	if err != nil {
		t.Fatal("decode:", err)
	}
	if x != y {
		t.Fatalf("%v != %v", x, y)
	}
}

func TestGobEncodePtrError(t *testing.T) {
	var err error
	b := new(bytes.Buffer)
	enc := NewEncoder(b)
	err = enc.Encode(&err)
	if err != nil {
		t.Fatal("encode:", err)
	}
	dec := NewDecoder(b)
	err2 := fmt.Errorf("foo")
	err = dec.Decode(&err2)
	if err != nil {
		t.Fatal("decode:", err)
	}
	if err2 != nil {
		t.Fatalf("expected nil, got %v", err2)
	}
}

func TestNetIP(t *testing.T) {
	// Encoding of net.IP{1,2,3,4} in Go 1.1.
	enc := []byte{0x07, 0x0a, 0x00, 0x04, 0x01, 0x02, 0x03, 0x04}

	var ip net.IP
	err := NewDecoder(bytes.NewReader(enc)).Decode(&ip)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if ip.String() != "1.2.3.4" {
		t.Errorf("decoded to %v, want 1.2.3.4", ip.String())
	}
}
