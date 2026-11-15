// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	stdjson "encoding/json"
	"reflect"
	"testing"
)

type pr4NumberAlias = stdjson.Number
type pr4DefinedNumber stdjson.Number
type pr4DefinedString string

type pr4QuotedObj int

func (pr4QuotedObj) MarshalJSON() ([]byte, error) {
	return []byte(`{"k":1}`), nil
}

type pr4QuotedStringObj string

func (pr4QuotedStringObj) MarshalJSON() ([]byte, error) {
	return []byte(`{"str_k":"custom"}`), nil
}

type pr4QuotedText string

func (pr4QuotedText) MarshalText() ([]byte, error) {
	return []byte("custom"), nil
}

type pr4PtrJSON int

func (*pr4PtrJSON) MarshalJSON() ([]byte, error) {
	return []byte(`"ptr"`), nil
}

type pr4StructPtrJSON struct {
	V pr4PtrJSON `json:"v,string"`
}

type pr4PtrText int

func (*pr4PtrText) MarshalText() ([]byte, error) {
	return []byte("ptrtext"), nil
}

type pr4StructPtrText struct {
	V pr4PtrText `json:"v,string"`
}

type pr4BothScalar int

func (pr4BothScalar) MarshalText() ([]byte, error) {
	return []byte("text"), nil
}

func (*pr4BothScalar) MarshalJSON() ([]byte, error) {
	return []byte(`"ptr"`), nil
}

type pr4StructBothScalar struct {
	V pr4BothScalar `json:"v,string"`
}

type pr4StructNilPtrs struct {
	Obj *pr4QuotedObj `json:"obj,string"`
	P   *int          `json:"p,string"`
}

func checkWalkInvalidNumber(t *testing.T, label string, v any) {
	t.Helper()
	checkWalkInvalidNumberImpl(t, label, v, false)
}

func checkWalkInvalidNumberExact(t *testing.T, label string, v any) {
	t.Helper()
	checkWalkInvalidNumberImpl(t, label, v, true)
}

func checkWalkInvalidNumberImpl(t *testing.T, label string, v any, matchText bool) {
	t.Helper()
	var s Walk
	got, gerr := s.Marshal(v)
	want, werr := stdjson.Marshal(v)
	if werr == nil {
		t.Fatalf("%s: test error: oracle unexpectedly succeeded: %s", label, want)
	}
	if gerr == nil {
		t.Fatalf("%s: walk unexpectedly succeeded (%s), want error matching oracle type %T (%v)", label, got, werr, werr)
	}
	if got != nil {
		t.Fatalf("%s: walk returned non-nil bytes %q on error", label, got)
	}
	if reflect.TypeOf(gerr) != reflect.TypeOf(werr) {
		t.Fatalf("%s: walk error type %T (%v), oracle error type %T (%v)", label, gerr, gerr, werr, werr)
	}
	if matchText && gerr.Error() != werr.Error() {
		t.Fatalf("%s: walk error %q, oracle error %q", label, gerr.Error(), werr.Error())
	}
	t.Logf("%s: oracle error: %v", label, werr)
}

func TestWalkPR4NumberSemantics(t *testing.T) {
	checkWalkParity(t, "root Number int", stdjson.Number("123"))
	checkWalkParity(t, "root Number minus zero", stdjson.Number("-0"))
	checkWalkParity(t, "root Number exp", stdjson.Number("1E+2"))
	checkWalkParity(t, "root Number float exp", stdjson.Number("0.5e-3"))

	checkWalkParity(t, "root Number empty", stdjson.Number(""))

	checkWalkParity(t, "struct field Number", struct {
		N stdjson.Number `json:"n"`
	}{N: "123"})
	checkWalkParity(t, "struct field Number empty", struct {
		N stdjson.Number `json:"n"`
	}{N: ""})

	seven := stdjson.Number("7")
	checkWalkParity(t, "struct ptr Number", struct {
		N *stdjson.Number `json:"n"`
	}{N: &seven})
	checkWalkParity(t, "struct nil ptr Number", struct {
		N *stdjson.Number `json:"n"`
	}{N: nil})

	checkWalkParity(t, "map[string]Number", map[string]stdjson.Number{"a": "7", "zero": ""})
	checkWalkParity(t, "map[string]any Number", map[string]any{"num": stdjson.Number("42")})
	checkWalkParity(t, "interface Number", any(stdjson.Number("2")))
	checkWalkParity(t, "slice Number", []stdjson.Number{"1", "", "2"})
	checkWalkParity(t, "nested composite Number", map[string][]stdjson.Number{"k": {"10", ""}})

	checkWalkParity(t, "alias Number field", struct {
		N pr4NumberAlias `json:"n"`
	}{N: "99"})
	checkWalkParity(t, "alias Number field empty", struct {
		N pr4NumberAlias `json:"n"`
	}{N: ""})

	checkWalkParity(t, "defined number field", struct {
		N pr4DefinedNumber `json:"n"`
	}{N: "123"})
	checkWalkParity(t, "defined number field empty", struct {
		N pr4DefinedNumber `json:"n"`
	}{N: ""})
	checkWalkParity(t, "defined string field", struct {
		S pr4DefinedString `json:"s"`
	}{S: "123"})
	checkWalkParity(t, "defined string field empty", struct {
		S pr4DefinedString `json:"s"`
	}{S: ""})
}

func TestWalkPR4NumberQuoted(t *testing.T) {
	checkWalkParity(t, "struct field Number,string", struct {
		N stdjson.Number `json:"n,string"`
	}{N: "123"})
	checkWalkParity(t, "struct field Number,string empty", struct {
		N stdjson.Number `json:"n,string"`
	}{N: ""})

	seven := stdjson.Number("7")
	checkWalkParity(t, "struct ptr Number,string", struct {
		N *stdjson.Number `json:"n,string"`
	}{N: &seven})
	empty := stdjson.Number("")
	checkWalkParity(t, "struct ptr Number,string empty", struct {
		N *stdjson.Number `json:"n,string"`
	}{N: &empty})
	checkWalkParity(t, "struct nil ptr Number,string", struct {
		N *stdjson.Number `json:"n,string"`
	}{N: nil})

	checkWalkParity(t, "alias Number field,string", struct {
		N pr4NumberAlias `json:"n,string"`
	}{N: "99"})
	checkWalkParity(t, "alias Number field,string empty", struct {
		N pr4NumberAlias `json:"n,string"`
	}{N: ""})

	checkWalkParity(t, "defined number field,string abc", struct {
		N pr4DefinedNumber `json:"n,string"`
	}{N: "abc"})
	checkWalkParity(t, "defined number field,string 123", struct {
		N pr4DefinedNumber `json:"n,string"`
	}{N: "123"})
	checkWalkParity(t, "defined number field,string empty", struct {
		N pr4DefinedNumber `json:"n,string"`
	}{N: ""})
	checkWalkParity(t, "defined string field,string abc", struct {
		S pr4DefinedString `json:"s,string"`
	}{S: "abc"})
	checkWalkParity(t, "defined string field,string 123", struct {
		S pr4DefinedString `json:"s,string"`
	}{S: "123"})
}

func TestWalkPR4NumberInvalid(t *testing.T) {
	invalids := []string{"-", "01", "1e", "1x", "+1", ".5", "1.", "NaN", " 1", "abc"}
	for _, bad := range invalids {
		num := stdjson.Number(bad)

		checkWalkInvalidNumberExact(t, "root invalid "+bad, num)
		checkWalkInvalidNumberExact(t, "any invalid "+bad, any(num))

		checkWalkInvalidNumber(t, "map value invalid "+bad, map[string]stdjson.Number{"bad": num})
		checkWalkInvalidNumber(t, "slice invalid "+bad, []stdjson.Number{num})
		checkWalkInvalidNumber(t, "struct plain invalid "+bad, struct {
			N stdjson.Number `json:"n"`
		}{N: num})

		checkWalkInvalidNumber(t, "struct tagged invalid "+bad, struct {
			N stdjson.Number `json:"n,string"`
		}{N: num})
		checkWalkInvalidNumber(t, "struct tagged ptr invalid "+bad, struct {
			N *stdjson.Number `json:"n,string"`
		}{N: &num})
	}
}

func TestWalkPR4QuotedMethodPrecedence(t *testing.T) {
	checkWalkParity(t, "named scalar MarshalJSON object", struct {
		V pr4QuotedObj `json:"v,string"`
	}{V: 42})

	checkWalkParity(t, "named string MarshalJSON object", struct {
		V pr4QuotedStringObj `json:"v,string"`
	}{V: "hello"})

	checkWalkParity(t, "named scalar MarshalText", struct {
		V pr4QuotedText `json:"v,string"`
	}{V: "plain"})

	checkWalkParity(t, "pointer-only MarshalJSON addressable", &pr4StructPtrJSON{V: 5})
	checkWalkParity(t, "pointer-only MarshalJSON non-addressable", pr4StructPtrJSON{V: 5})
	checkWalkParity(t, "pointer-only MarshalJSON in map", map[string]pr4StructPtrJSON{"k": {V: 5}})

	checkWalkParity(t, "pointer-only MarshalText addressable", &pr4StructPtrText{V: 5})
	checkWalkParity(t, "pointer-only MarshalText non-addressable", pr4StructPtrText{V: 5})

	checkWalkParity(t, "value-Text and pointer-JSON addressable", &pr4StructBothScalar{V: 1})
	checkWalkParity(t, "value-Text and pointer-JSON non-addressable", pr4StructBothScalar{V: 1})

	checkWalkParity(t, "nil pointers with and without methods", pr4StructNilPtrs{Obj: nil, P: nil})
	val := pr4QuotedObj(9)
	checkWalkParity(t, "non-nil pointer with methods", pr4StructNilPtrs{Obj: &val, P: nil})
	p := 123
	checkWalkParity(t, "non-nil pointer without methods", pr4StructNilPtrs{Obj: nil, P: &p})
}

type pr4IsZeroInt int

func (v pr4IsZeroInt) IsZero() bool { return v == 42 }

type pr4IsZeroString string

func (v pr4IsZeroString) IsZero() bool { return v == "zero" }

type pr4IsZeroBool bool

func (v pr4IsZeroBool) IsZero() bool { return bool(v) }

type pr4IsZeroPtrRecv int

func (v *pr4IsZeroPtrRecv) IsZero() bool { return *v == 7 }

type pr4IsZeroPlusJSON int

func (v pr4IsZeroPlusJSON) IsZero() bool { return v == 0 }

func (v pr4IsZeroPlusJSON) MarshalJSON() ([]byte, error) { return []byte(`"custom"`), nil }

func TestWalkPR4IsZeroStringOption(t *testing.T) {
	checkWalkParity(t, "iszero int,string", struct {
		V pr4IsZeroInt `json:"v,string"`
	}{V: 5})
	checkWalkParity(t, "iszero int,string custom-zero", struct {
		V pr4IsZeroInt `json:"v,string"`
	}{V: 42})
	checkWalkParity(t, "iszero string,string", struct {
		V pr4IsZeroString `json:"v,string"`
	}{V: "hi"})
	checkWalkParity(t, "iszero bool,string true", struct {
		V pr4IsZeroBool `json:"v,string"`
	}{V: true})
	checkWalkParity(t, "iszero bool,string false", struct {
		V pr4IsZeroBool `json:"v,string"`
	}{V: false})
}

func TestWalkPR4IsZeroOmitZero(t *testing.T) {
	checkWalkParity(t, "iszero int omitzero custom-zero omitted", struct {
		V pr4IsZeroInt `json:"v,omitzero"`
	}{V: 42})
	checkWalkParity(t, "iszero int omitzero keeps naive-zero", struct {
		V pr4IsZeroInt `json:"v,omitzero"`
	}{V: 0})
	checkWalkParity(t, "iszero string omitzero custom-zero omitted", struct {
		V pr4IsZeroString `json:"v,omitzero"`
	}{V: "zero"})
	checkWalkParity(t, "iszero string omitzero keeps empty", struct {
		V pr4IsZeroString `json:"v,omitzero"`
	}{V: ""})
	checkWalkParity(t, "iszero bool omitzero true omitted", struct {
		V pr4IsZeroBool `json:"v,omitzero"`
	}{V: true})
	checkWalkParity(t, "iszero bool omitzero keeps false", struct {
		V pr4IsZeroBool `json:"v,omitzero"`
	}{V: false})
	checkWalkParity(t, "iszero ptr-recv omitzero omitted", struct {
		V pr4IsZeroPtrRecv `json:"v,omitzero"`
	}{V: 7})
	checkWalkParity(t, "iszero ptr-recv omitzero kept", struct {
		V pr4IsZeroPtrRecv `json:"v,omitzero"`
	}{V: 0})
}

func TestWalkPR4IsZeroStringOmitZero(t *testing.T) {
	checkWalkParity(t, "iszero int string+omitzero custom-zero omitted", struct {
		V pr4IsZeroInt `json:"v,string,omitzero"`
	}{V: 42})
	checkWalkParity(t, "iszero int string+omitzero keeps naive-zero quoted", struct {
		V pr4IsZeroInt `json:"v,string,omitzero"`
	}{V: 0})
	checkWalkParity(t, "iszero string string+omitzero quoted", struct {
		V pr4IsZeroString `json:"v,string,omitzero"`
	}{V: "hi"})
	checkWalkParity(t, "iszero bool string+omitzero quoted", struct {
		V pr4IsZeroBool `json:"v,string,omitzero"`
	}{V: true})
}

func TestWalkPR4IsZeroMarshalerPrecedence(t *testing.T) {
	checkWalkParity(t, "iszero+marshalJSON keeps honour under string", struct {
		V pr4IsZeroPlusJSON `json:"v,string"`
	}{V: 5})
	checkWalkParity(t, "iszero+marshalJSON omitzero honour", struct {
		V pr4IsZeroPlusJSON `json:"v,omitzero"`
	}{V: 5})
}
