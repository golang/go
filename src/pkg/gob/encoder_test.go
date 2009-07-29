// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gob

import (
	"bytes";
	"gob";
	"io";
	"os";
	"reflect";
	"strings";
	"testing";
	"unsafe";
)

type ET2 struct {
	x string;
}

type ET1 struct {
	a int;
	et2 *ET2;
	next *ET1;
}

// Like ET1 but with a different name for a field
type ET3 struct {
	a int;
	et2 *ET2;
	differentNext *ET1;
}

// Like ET1 but with a different type for a field
type ET4 struct {
	a int;
	et2 *ET1;
	next int;
}

func TestBasicEncoder(t *testing.T) {
	b := new(bytes.Buffer);
	enc := NewEncoder(b);
	et1 := new(ET1);
	et1.a = 7;
	et1.et2 = new(ET2);
	enc.Encode(et1);
	if enc.state.err != nil {
		t.Error("encoder fail:", enc.state.err)
	}

	// Decode the result by hand to verify;
	state := newDecodeState(b);
	// The output should be:
	// 0) The length, 38.
	length := decodeUint(state);
	if length != 38 {
		t.Fatal("0. expected length 38; got", length);
	}
	// 1) -7: the type id of ET1
	id1 := decodeInt(state);
	if id1 >= 0 {
		t.Fatal("expected ET1 negative id; got", id1);
	}
	// 2) The wireType for ET1
	wire1 := new(wireType);
	err := decode(b, tWireType, wire1);
	if err != nil {
		t.Fatal("error decoding ET1 type:", err);
	}
	info := getTypeInfo(reflect.Typeof(ET1{}));
	trueWire1 := &wireType{s: info.id.gobType().(*structType)};
	if !reflect.DeepEqual(wire1, trueWire1) {
		t.Fatalf("invalid wireType for ET1: expected %+v; got %+v\n", *trueWire1, *wire1);
	}
	// 3) The length, 21.
	length = decodeUint(state);
	if length != 21 {
		t.Fatal("3. expected length 21; got", length);
	}
	// 4) -8: the type id of ET2
	id2 := decodeInt(state);
	if id2 >= 0 {
		t.Fatal("expected ET2 negative id; got", id2);
	}
	// 5) The wireType for ET2
	wire2 := new(wireType);
	err = decode(b, tWireType, wire2);
	if err != nil {
		t.Fatal("error decoding ET2 type:", err);
	}
	info = getTypeInfo(reflect.Typeof(ET2{}));
	trueWire2 := &wireType{s: info.id.gobType().(*structType)};
	if !reflect.DeepEqual(wire2, trueWire2) {
		t.Fatalf("invalid wireType for ET2: expected %+v; got %+v\n", *trueWire2, *wire2);
	}
	// 6) The length, 6.
	length = decodeUint(state);
	if length != 6 {
		t.Fatal("6. expected length 6; got", length);
	}
	// 7) The type id for the et1 value
	newId1 := decodeInt(state);
	if newId1 != -id1 {
		t.Fatal("expected Et1 id", -id1, "got", newId1);
	}
	// 8) The value of et1
	newEt1 := new(ET1);
	et1Id := getTypeInfo(reflect.Typeof(*newEt1)).id;
	err = decode(b, et1Id, newEt1);
	if err != nil {
		t.Fatal("error decoding ET1 value:", err);
	}
	if !reflect.DeepEqual(et1, newEt1) {
		t.Fatalf("invalid data for et1: expected %+v; got %+v\n", *et1, *newEt1);
	}
	// 9) EOF
	if b.Len() != 0 {
		t.Error("not at eof;", b.Len(), "bytes left")
	}

	// Now do it again. This time we should see only the type id and value.
	b.Reset();
	enc.Encode(et1);
	if enc.state.err != nil {
		t.Error("2nd round: encoder fail:", enc.state.err)
	}
	// The length.
	length = decodeUint(state);
	if length != 6 {
		t.Fatal("6. expected length 6; got", length);
	}
	// 5a) The type id for the et1 value
	newId1 = decodeInt(state);
	if newId1 != -id1 {
		t.Fatal("2nd round: expected Et1 id", -id1, "got", newId1);
	}
	// 6a) The value of et1
	newEt1 = new(ET1);
	err = decode(b, et1Id, newEt1);
	if err != nil {
		t.Fatal("2nd round: error decoding ET1 value:", err);
	}
	if !reflect.DeepEqual(et1, newEt1) {
		t.Fatalf("2nd round: invalid data for et1: expected %+v; got %+v\n", *et1, *newEt1);
	}
	// 7a) EOF
	if b.Len() != 0 {
		t.Error("2nd round: not at eof;", b.Len(), "bytes left")
	}
}

func TestEncoderDecoder(t *testing.T) {
	b := new(bytes.Buffer);
	enc := NewEncoder(b);
	et1 := new(ET1);
	et1.a = 7;
	et1.et2 = new(ET2);
	enc.Encode(et1);
	if enc.state.err != nil {
		t.Error("encoder fail:", enc.state.err)
	}
	dec := NewDecoder(b);
	newEt1 := new(ET1);
	dec.Decode(newEt1);
	if dec.state.err != nil {
		t.Fatal("error decoding ET1:", dec.state.err);
	}

	if !reflect.DeepEqual(et1, newEt1) {
		t.Fatalf("invalid data for et1: expected %+v; got %+v\n", *et1, *newEt1);
	}
	if b.Len() != 0 {
		t.Error("not at eof;", b.Len(), "bytes left")
	}

	enc.Encode(et1);
	newEt1 = new(ET1);
	dec.Decode(newEt1);
	if dec.state.err != nil {
		t.Fatal("round 2: error decoding ET1:", dec.state.err);
	}
	if !reflect.DeepEqual(et1, newEt1) {
		t.Fatalf("round 2: invalid data for et1: expected %+v; got %+v\n", *et1, *newEt1);
	}
	if b.Len() != 0 {
		t.Error("round 2: not at eof;", b.Len(), "bytes left")
	}

	// Now test with a running encoder/decoder pair that we recognize a type mismatch.
	enc.Encode(et1);
	if enc.state.err != nil {
		t.Error("round 3: encoder fail:", enc.state.err)
	}
	newEt2 := new(ET2);
	dec.Decode(newEt2);
	if dec.state.err == nil {
		t.Fatal("round 3: expected `bad type' error decoding ET2");
	}
}

// Run one value through the encoder/decoder, but use the wrong type.
// Input is always an ET1; we compare it to whatever is under 'e'.
func badTypeCheck(e interface{}, shouldFail bool, msg string, t *testing.T) {
	b := new(bytes.Buffer);
	enc := NewEncoder(b);
	et1 := new(ET1);
	et1.a = 7;
	et1.et2 = new(ET2);
	enc.Encode(et1);
	if enc.state.err != nil {
		t.Error("encoder fail:", enc.state.err)
	}
	dec := NewDecoder(b);
	dec.Decode(e);
	if shouldFail && (dec.state.err == nil) {
		t.Error("expected error for", msg);
	}
	if !shouldFail && (dec.state.err != nil) {
		t.Error("unexpected error for", msg);
	}
}

// Test that we recognize a bad type the first time.
func TestWrongTypeDecoder(t *testing.T) {
	badTypeCheck(new(ET2), true, "no fields in common", t);
	badTypeCheck(new(ET3), false, "different name of field", t);
	badTypeCheck(new(ET4), true, "different type of field", t);
}

func corruptDataCheck(s string, err os.Error, t *testing.T) {
	b := bytes.NewBuffer(strings.Bytes(s));
	dec := NewDecoder(b);
	dec.Decode(new(ET2));
	if dec.state.err != err {
		t.Error("expected error", err, "got", dec.state.err);
	}
}

// Check that we survive bad data.
func TestBadData(t *testing.T) {
	corruptDataCheck("\x01\x01\x01", os.EOF, t);
	corruptDataCheck("\x7Fhi", io.ErrUnexpectedEOF, t);
	corruptDataCheck("\x03now is the time for all good men", errBadType, t);
}
