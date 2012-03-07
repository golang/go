// Copyright 20011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests of the GobEncoder/GobDecoder support.

package gob

import (
	"bytes"
	"errors"
	"fmt"
	"io"
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

func (v ValueGobber) GobEncode() ([]byte, error) {
	return []byte(fmt.Sprintf("VALUE=%s", v)), nil
}

func (v *ValueGobber) GobDecode(data []byte) error {
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
}

type GobTest4 struct {
	X int // guarantee we have  something in common with GobTest*
	V ValueGobber
}

type GobTest5 struct {
	X int // guarantee we have  something in common with GobTest*
	V *ValueGobber
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
	err = enc.Encode(GobTest3{17, &gobber})
	if err != nil {
		t.Fatal("encode error:", err)
	}
	y := new(GobTest3)
	err = dec.Decode(y)
	if err != nil {
		t.Fatal("decode error:", err)
	}
	if *y.G != 23 {
		t.Errorf("expected '23 got %d", *y.G)
	}
}

// Even though the field is a value, we can still take its address
// and should be able to call the methods.
func TestGobEncoderValueField(t *testing.T) {
	b := new(bytes.Buffer)
	// First a field that's a structure.
	enc := NewEncoder(b)
	err := enc.Encode(GobTestValueEncDec{17, StringStruct{"HIJKL"}})
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
	err := enc.Encode(a)
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
// interface, we can cross-connect them.  Not sure it's useful
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
		t.Fatalf("expected `XYZ` got %c", y.G.s)
	}
}

// Test that we can encode a value and decode into a pointer.
func TestGobEncoderValueEncoder(t *testing.T) {
	// first, string in field to byte in field
	b := new(bytes.Buffer)
	enc := NewEncoder(b)
	err := enc.Encode(GobTest4{17, ValueGobber("hello")})
	if err != nil {
		t.Fatal("encode error:", err)
	}
	dec := NewDecoder(b)
	x := new(GobTest5)
	err = dec.Decode(x)
	if err != nil {
		t.Fatal("decode error:", err)
	}
	if *x.V != "hello" {
		t.Errorf("expected `hello` got %s", x.V)
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
	if strings.Index(err.Error(), "type") < 0 {
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
	if strings.Index(err.Error(), "type") < 0 {
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
	err := enc.Encode(Gobber(1234))
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
	err := enc.Encode(GobTest3{17, &gobber})
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
// We throw in a gob-encoding array, to test another case of isZero

type isZeroBug struct {
	T time.Time
	S string
	I int
	A isZeroBugArray
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

func TestGobEncodeIsZero(t *testing.T) {
	x := isZeroBug{time.Now(), "hello", -55, isZeroBugArray{1, 2}}
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
