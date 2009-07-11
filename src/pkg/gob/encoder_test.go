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
	next *ET2;
}

// Like ET1 but with a different type for a self-referencing field
type ET5 struct {
	a int;
	et2 *ET2;
	next *ET1;
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
	state := new(DecState);
	state.r = b;
	// The output should be:
	// 1) -7: the type id of ET1
	id1 := DecodeInt(state);
	if id1 >= 0 {
		t.Fatal("expected ET1 negative id; got", id1);
	}
	// 2) The wireType for ET1
	wire1 := new(wireType);
	err := Decode(b, wire1);
	if err != nil {
		t.Fatal("error decoding ET1 type:", err);
	}
	info := getTypeInfo(reflect.Typeof(ET1{}));
	trueWire1 := &wireType{name:"ET1", s: info.typeId.gobType().(*structType)};
	if !reflect.DeepEqual(wire1, trueWire1) {
		t.Fatalf("invalid wireType for ET1: expected %+v; got %+v\n", *trueWire1, *wire1);
	}
	// 3) -8: the type id of ET2
	id2 := DecodeInt(state);
	if id2 >= 0 {
		t.Fatal("expected ET2 negative id; got", id2);
	}
	// 4) The wireType for ET2
	wire2 := new(wireType);
	err = Decode(b, wire2);
	if err != nil {
		t.Fatal("error decoding ET2 type:", err);
	}
	info = getTypeInfo(reflect.Typeof(ET2{}));
	trueWire2 := &wireType{name:"ET2", s: info.typeId.gobType().(*structType)};
	if !reflect.DeepEqual(wire2, trueWire2) {
		t.Fatalf("invalid wireType for ET2: expected %+v; got %+v\n", *trueWire2, *wire2);
	}
	// 5) The type id for the et1 value
	newId1 := DecodeInt(state);
	if newId1 != -id1 {
		t.Fatal("expected Et1 id", -id1, "got", newId1);
	}
	// 6) The value of et1
	newEt1 := new(ET1);
	err = Decode(b, newEt1);
	if err != nil {
		t.Fatal("error decoding ET1 value:", err);
	}
	if !reflect.DeepEqual(et1, newEt1) {
		t.Fatalf("invalid data for et1: expected %+v; got %+v\n", *et1, *newEt1);
	}
	// 7) EOF
	if b.Len() != 0 {
		t.Error("not at eof;", b.Len(), "bytes left")
	}

	// Now do it again. This time we should see only the type id and value.
	b.Reset();
	enc.Encode(et1);
	if enc.state.err != nil {
		t.Error("2nd round: encoder fail:", enc.state.err)
	}
	// 5a) The type id for the et1 value
	newId1 = DecodeInt(state);
	if newId1 != -id1 {
		t.Fatal("2nd round: expected Et1 id", -id1, "got", newId1);
	}
	// 6a) The value of et1
	newEt1 = new(ET1);
	err = Decode(b, newEt1);
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
		t.Fatalf("error decoding ET1:", dec.state.err);
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
		t.Fatalf("round 2: error decoding ET1:", dec.state.err);
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
		t.Fatalf("round 3: expected `bad type' error decoding ET2");
	}
}

// Run one value through the encoder/decoder, but use the wrong type.
func badTypeCheck(e interface{}, msg string, t *testing.T) {
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
	if dec.state.err == nil {
		t.Error("expected error for", msg);
	}
}

// Test that we recognize a bad type the first time.
func TestWrongTypeDecoder(t *testing.T) {
	badTypeCheck(new(ET2), "different number of fields", t);
	badTypeCheck(new(ET3), "different name of field", t);
	badTypeCheck(new(ET4), "different type of field", t);
	badTypeCheck(new(ET5), "different type of self-reference field", t);
}
