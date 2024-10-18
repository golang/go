// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	"bytes"
	"encoding"
	"fmt"
	"log"
	"math"
	"os"
	"reflect"
	"regexp"
	"runtime/debug"
	"runtime/metrics"
	"strconv"
	"testing"
	"time"
)

type OptionalsEmpty struct {
	Sr string `json:"sr"`
	So string `json:"so,omitempty"`
	Sw string `json:"-"`

	Ir int `json:"omitempty"` // actually named omitempty, not an option
	Io int `json:"io,omitempty"`

	Slr []string `json:"slr,random"`
	Slo []string `json:"slo,omitempty"`

	Mr map[string]any `json:"mr"`
	Mo map[string]any `json:",omitempty"`

	Fr float64 `json:"fr"`
	Fo float64 `json:"fo,omitempty"`

	Br bool `json:"br"`
	Bo bool `json:"bo,omitempty"`

	Ur uint `json:"ur"`
	Uo uint `json:"uo,omitempty"`

	Str struct{} `json:"str"`
	Sto struct{} `json:"sto,omitempty"`
}

func TestOmitEmpty(t *testing.T) {
	const want = `{
 "sr": "",
 "omitempty": 0,
 "slr": null,
 "mr": {},
 "fr": 0,
 "br": false,
 "ur": 0,
 "str": {},
 "sto": {}
}`
	var o OptionalsEmpty
	o.Sw = "something"
	o.Mr = map[string]any{}
	o.Mo = map[string]any{}

	got, err := MarshalIndent(&o, "", " ")
	if err != nil {
		t.Fatalf("MarshalIndent error: %v", err)
	}
	if got := string(got); got != want {
		t.Errorf("MarshalIndent:\n\tgot:  %s\n\twant: %s\n", indentNewlines(got), indentNewlines(want))
	}
}

type NonZeroStruct struct{}

func (nzs NonZeroStruct) IsZero() bool {
	return false
}

type NoPanicStruct struct {
	Int int `json:"int,omitzero"`
}

func (nps *NoPanicStruct) IsZero() bool {
	return nps.Int != 0
}

type OptionalsZero struct {
	Sr string `json:"sr"`
	So string `json:"so,omitzero"`
	Sw string `json:"-"`

	Ir int `json:"omitzero"` // actually named omitzero, not an option
	Io int `json:"io,omitzero"`

	Slr       []string `json:"slr,random"`
	Slo       []string `json:"slo,omitzero"`
	SloNonNil []string `json:"slononnil,omitzero"`

	Mr  map[string]any `json:"mr"`
	Mo  map[string]any `json:",omitzero"`
	Moo map[string]any `json:"moo,omitzero"`

	Fr   float64    `json:"fr"`
	Fo   float64    `json:"fo,omitzero"`
	Foo  float64    `json:"foo,omitzero"`
	Foo2 [2]float64 `json:"foo2,omitzero"`

	Br bool `json:"br"`
	Bo bool `json:"bo,omitzero"`

	Ur uint `json:"ur"`
	Uo uint `json:"uo,omitzero"`

	Str struct{} `json:"str"`
	Sto struct{} `json:"sto,omitzero"`

	Time      time.Time     `json:"time,omitzero"`
	TimeLocal time.Time     `json:"timelocal,omitzero"`
	Nzs       NonZeroStruct `json:"nzs,omitzero"`

	NilIsZeroer    isZeroer       `json:"niliszeroer,omitzero"`    // nil interface
	NonNilIsZeroer isZeroer       `json:"nonniliszeroer,omitzero"` // non-nil interface
	NoPanicStruct0 isZeroer       `json:"nps0,omitzero"`           // non-nil interface with nil pointer
	NoPanicStruct1 isZeroer       `json:"nps1,omitzero"`           // non-nil interface with non-nil pointer
	NoPanicStruct2 *NoPanicStruct `json:"nps2,omitzero"`           // nil pointer
	NoPanicStruct3 *NoPanicStruct `json:"nps3,omitzero"`           // non-nil pointer
	NoPanicStruct4 NoPanicStruct  `json:"nps4,omitzero"`           // concrete type
}

func TestOmitZero(t *testing.T) {
	const want = `{
 "sr": "",
 "omitzero": 0,
 "slr": null,
 "slononnil": [],
 "mr": {},
 "Mo": {},
 "fr": 0,
 "br": false,
 "ur": 0,
 "str": {},
 "nzs": {},
 "nps1": {},
 "nps3": {},
 "nps4": {}
}`
	var o OptionalsZero
	o.Sw = "something"
	o.SloNonNil = make([]string, 0)
	o.Mr = map[string]any{}
	o.Mo = map[string]any{}

	o.Foo = -0
	o.Foo2 = [2]float64{+0, -0}

	o.TimeLocal = time.Time{}.Local()

	o.NonNilIsZeroer = time.Time{}
	o.NoPanicStruct0 = (*NoPanicStruct)(nil)
	o.NoPanicStruct1 = &NoPanicStruct{}
	o.NoPanicStruct3 = &NoPanicStruct{}

	got, err := MarshalIndent(&o, "", " ")
	if err != nil {
		t.Fatalf("MarshalIndent error: %v", err)
	}
	if got := string(got); got != want {
		t.Errorf("MarshalIndent:\n\tgot:  %s\n\twant: %s\n", indentNewlines(got), indentNewlines(want))
	}
}

func TestOmitZeroMap(t *testing.T) {
	const want = `{
 "foo": {
  "sr": "",
  "omitzero": 0,
  "slr": null,
  "mr": null,
  "fr": 0,
  "br": false,
  "ur": 0,
  "str": {},
  "nzs": {},
  "nps4": {}
 }
}`
	m := map[string]OptionalsZero{"foo": {}}
	got, err := MarshalIndent(m, "", " ")
	if err != nil {
		t.Fatalf("MarshalIndent error: %v", err)
	}
	if got := string(got); got != want {
		fmt.Println(got)
		t.Errorf("MarshalIndent:\n\tgot:  %s\n\twant: %s\n", indentNewlines(got), indentNewlines(want))
	}
}

type OptionalsEmptyZero struct {
	Sr string `json:"sr"`
	So string `json:"so,omitempty,omitzero"`
	Sw string `json:"-"`

	Io int `json:"io,omitempty,omitzero"`

	Slr       []string `json:"slr,random"`
	Slo       []string `json:"slo,omitempty,omitzero"`
	SloNonNil []string `json:"slononnil,omitempty,omitzero"`

	Mr map[string]any `json:"mr"`
	Mo map[string]any `json:",omitempty,omitzero"`

	Fr float64 `json:"fr"`
	Fo float64 `json:"fo,omitempty,omitzero"`

	Br bool `json:"br"`
	Bo bool `json:"bo,omitempty,omitzero"`

	Ur uint `json:"ur"`
	Uo uint `json:"uo,omitempty,omitzero"`

	Str struct{} `json:"str"`
	Sto struct{} `json:"sto,omitempty,omitzero"`

	Time time.Time     `json:"time,omitempty,omitzero"`
	Nzs  NonZeroStruct `json:"nzs,omitempty,omitzero"`
}

func TestOmitEmptyZero(t *testing.T) {
	const want = `{
 "sr": "",
 "slr": null,
 "mr": {},
 "fr": 0,
 "br": false,
 "ur": 0,
 "str": {},
 "nzs": {}
}`
	var o OptionalsEmptyZero
	o.Sw = "something"
	o.SloNonNil = make([]string, 0)
	o.Mr = map[string]any{}
	o.Mo = map[string]any{}

	got, err := MarshalIndent(&o, "", " ")
	if err != nil {
		t.Fatalf("MarshalIndent error: %v", err)
	}
	if got := string(got); got != want {
		t.Errorf("MarshalIndent:\n\tgot:  %s\n\twant: %s\n", indentNewlines(got), indentNewlines(want))
	}
}

type StringTag struct {
	BoolStr    bool    `json:",string"`
	IntStr     int64   `json:",string"`
	UintptrStr uintptr `json:",string"`
	StrStr     string  `json:",string"`
	NumberStr  Number  `json:",string"`
}

func TestRoundtripStringTag(t *testing.T) {
	tests := []struct {
		CaseName
		in   StringTag
		want string // empty to just test that we roundtrip
	}{{
		CaseName: Name("AllTypes"),
		in: StringTag{
			BoolStr:    true,
			IntStr:     42,
			UintptrStr: 44,
			StrStr:     "xzbit",
			NumberStr:  "46",
		},
		want: `{
	"BoolStr": "true",
	"IntStr": "42",
	"UintptrStr": "44",
	"StrStr": "\"xzbit\"",
	"NumberStr": "46"
}`,
	}, {
		// See golang.org/issues/38173.
		CaseName: Name("StringDoubleEscapes"),
		in: StringTag{
			StrStr:    "\b\f\n\r\t\"\\",
			NumberStr: "0", // just to satisfy the roundtrip
		},
		want: `{
	"BoolStr": "false",
	"IntStr": "0",
	"UintptrStr": "0",
	"StrStr": "\"\\b\\f\\n\\r\\t\\\"\\\\\"",
	"NumberStr": "0"
}`,
	}}
	for _, tt := range tests {
		t.Run(tt.Name, func(t *testing.T) {
			got, err := MarshalIndent(&tt.in, "", "\t")
			if err != nil {
				t.Fatalf("%s: MarshalIndent error: %v", tt.Where, err)
			}
			if got := string(got); got != tt.want {
				t.Fatalf("%s: MarshalIndent:\n\tgot:  %s\n\twant: %s", tt.Where, stripWhitespace(got), stripWhitespace(tt.want))
			}

			// Verify that it round-trips.
			var s2 StringTag
			if err := Unmarshal(got, &s2); err != nil {
				t.Fatalf("%s: Decode error: %v", tt.Where, err)
			}
			if !reflect.DeepEqual(s2, tt.in) {
				t.Fatalf("%s: Decode:\n\tinput: %s\n\tgot:  %#v\n\twant: %#v", tt.Where, indentNewlines(string(got)), s2, tt.in)
			}
		})
	}
}

// byte slices are special even if they're renamed types.
type renamedByte byte
type renamedByteSlice []byte
type renamedRenamedByteSlice []renamedByte

func TestEncodeRenamedByteSlice(t *testing.T) {
	s := renamedByteSlice("abc")
	got, err := Marshal(s)
	if err != nil {
		t.Fatalf("Marshal error: %v", err)
	}
	want := `"YWJj"`
	if string(got) != want {
		t.Errorf("Marshal:\n\tgot:  %s\n\twant: %s", got, want)
	}
	r := renamedRenamedByteSlice("abc")
	got, err = Marshal(r)
	if err != nil {
		t.Fatalf("Marshal error: %v", err)
	}
	if string(got) != want {
		t.Errorf("Marshal:\n\tgot:  %s\n\twant: %s", got, want)
	}
}

type SamePointerNoCycle struct {
	Ptr1, Ptr2 *SamePointerNoCycle
}

var samePointerNoCycle = &SamePointerNoCycle{}

type PointerCycle struct {
	Ptr *PointerCycle
}

var pointerCycle = &PointerCycle{}

type PointerCycleIndirect struct {
	Ptrs []any
}

type RecursiveSlice []RecursiveSlice

var (
	pointerCycleIndirect = &PointerCycleIndirect{}
	mapCycle             = make(map[string]any)
	sliceCycle           = []any{nil}
	sliceNoCycle         = []any{nil, nil}
	recursiveSliceCycle  = []RecursiveSlice{nil}
)

func init() {
	ptr := &SamePointerNoCycle{}
	samePointerNoCycle.Ptr1 = ptr
	samePointerNoCycle.Ptr2 = ptr

	pointerCycle.Ptr = pointerCycle
	pointerCycleIndirect.Ptrs = []any{pointerCycleIndirect}

	mapCycle["x"] = mapCycle
	sliceCycle[0] = sliceCycle
	sliceNoCycle[1] = sliceNoCycle[:1]
	for i := startDetectingCyclesAfter; i > 0; i-- {
		sliceNoCycle = []any{sliceNoCycle}
	}
	recursiveSliceCycle[0] = recursiveSliceCycle
}

func TestSamePointerNoCycle(t *testing.T) {
	if _, err := Marshal(samePointerNoCycle); err != nil {
		t.Fatalf("Marshal error: %v", err)
	}
}

func TestSliceNoCycle(t *testing.T) {
	if _, err := Marshal(sliceNoCycle); err != nil {
		t.Fatalf("Marshal error: %v", err)
	}
}

func TestUnsupportedValues(t *testing.T) {
	tests := []struct {
		CaseName
		in any
	}{
		{Name(""), math.NaN()},
		{Name(""), math.Inf(-1)},
		{Name(""), math.Inf(1)},
		{Name(""), pointerCycle},
		{Name(""), pointerCycleIndirect},
		{Name(""), mapCycle},
		{Name(""), sliceCycle},
		{Name(""), recursiveSliceCycle},
	}
	for _, tt := range tests {
		t.Run(tt.Name, func(t *testing.T) {
			if _, err := Marshal(tt.in); err != nil {
				if _, ok := err.(*UnsupportedValueError); !ok {
					t.Errorf("%s: Marshal error:\n\tgot:  %T\n\twant: %T", tt.Where, err, new(UnsupportedValueError))
				}
			} else {
				t.Errorf("%s: Marshal error: got nil, want non-nil", tt.Where)
			}
		})
	}
}

// Issue 43207
func TestMarshalTextFloatMap(t *testing.T) {
	m := map[textfloat]string{
		textfloat(math.NaN()): "1",
		textfloat(math.NaN()): "1",
	}
	got, err := Marshal(m)
	if err != nil {
		t.Errorf("Marshal error: %v", err)
	}
	want := `{"TF:NaN":"1","TF:NaN":"1"}`
	if string(got) != want {
		t.Errorf("Marshal:\n\tgot:  %s\n\twant: %s", got, want)
	}
}

// Ref has Marshaler and Unmarshaler methods with pointer receiver.
type Ref int

func (*Ref) MarshalJSON() ([]byte, error) {
	return []byte(`"ref"`), nil
}

func (r *Ref) UnmarshalJSON([]byte) error {
	*r = 12
	return nil
}

// Val has Marshaler methods with value receiver.
type Val int

func (Val) MarshalJSON() ([]byte, error) {
	return []byte(`"val"`), nil
}

// RefText has Marshaler and Unmarshaler methods with pointer receiver.
type RefText int

func (*RefText) MarshalText() ([]byte, error) {
	return []byte(`"ref"`), nil
}

func (r *RefText) UnmarshalText([]byte) error {
	*r = 13
	return nil
}

// ValText has Marshaler methods with value receiver.
type ValText int

func (ValText) MarshalText() ([]byte, error) {
	return []byte(`"val"`), nil
}

func TestRefValMarshal(t *testing.T) {
	var s = struct {
		R0 Ref
		R1 *Ref
		R2 RefText
		R3 *RefText
		V0 Val
		V1 *Val
		V2 ValText
		V3 *ValText
	}{
		R0: 12,
		R1: new(Ref),
		R2: 14,
		R3: new(RefText),
		V0: 13,
		V1: new(Val),
		V2: 15,
		V3: new(ValText),
	}
	const want = `{"R0":"ref","R1":"ref","R2":"\"ref\"","R3":"\"ref\"","V0":"val","V1":"val","V2":"\"val\"","V3":"\"val\""}`
	b, err := Marshal(&s)
	if err != nil {
		t.Fatalf("Marshal error: %v", err)
	}
	if got := string(b); got != want {
		t.Errorf("Marshal:\n\tgot:  %s\n\twant: %s", got, want)
	}
}

// C implements Marshaler and returns unescaped JSON.
type C int

func (C) MarshalJSON() ([]byte, error) {
	return []byte(`"<&>"`), nil
}

// CText implements Marshaler and returns unescaped text.
type CText int

func (CText) MarshalText() ([]byte, error) {
	return []byte(`"<&>"`), nil
}

func TestMarshalerEscaping(t *testing.T) {
	var c C
	want := `"\u003c\u0026\u003e"`
	b, err := Marshal(c)
	if err != nil {
		t.Fatalf("Marshal error: %v", err)
	}
	if got := string(b); got != want {
		t.Errorf("Marshal:\n\tgot:  %s\n\twant: %s", got, want)
	}

	var ct CText
	want = `"\"\u003c\u0026\u003e\""`
	b, err = Marshal(ct)
	if err != nil {
		t.Fatalf("Marshal error: %v", err)
	}
	if got := string(b); got != want {
		t.Errorf("Marshal:\n\tgot:  %s\n\twant: %s", got, want)
	}
}

func TestAnonymousFields(t *testing.T) {
	tests := []struct {
		CaseName
		makeInput func() any // Function to create input value
		want      string     // Expected JSON output
	}{{
		// Both S1 and S2 have a field named X. From the perspective of S,
		// it is ambiguous which one X refers to.
		// This should not serialize either field.
		CaseName: Name("AmbiguousField"),
		makeInput: func() any {
			type (
				S1 struct{ x, X int }
				S2 struct{ x, X int }
				S  struct {
					S1
					S2
				}
			)
			return S{S1{1, 2}, S2{3, 4}}
		},
		want: `{}`,
	}, {
		CaseName: Name("DominantField"),
		// Both S1 and S2 have a field named X, but since S has an X field as
		// well, it takes precedence over S1.X and S2.X.
		makeInput: func() any {
			type (
				S1 struct{ x, X int }
				S2 struct{ x, X int }
				S  struct {
					S1
					S2
					x, X int
				}
			)
			return S{S1{1, 2}, S2{3, 4}, 5, 6}
		},
		want: `{"X":6}`,
	}, {
		// Unexported embedded field of non-struct type should not be serialized.
		CaseName: Name("UnexportedEmbeddedInt"),
		makeInput: func() any {
			type (
				myInt int
				S     struct{ myInt }
			)
			return S{5}
		},
		want: `{}`,
	}, {
		// Exported embedded field of non-struct type should be serialized.
		CaseName: Name("ExportedEmbeddedInt"),
		makeInput: func() any {
			type (
				MyInt int
				S     struct{ MyInt }
			)
			return S{5}
		},
		want: `{"MyInt":5}`,
	}, {
		// Unexported embedded field of pointer to non-struct type
		// should not be serialized.
		CaseName: Name("UnexportedEmbeddedIntPointer"),
		makeInput: func() any {
			type (
				myInt int
				S     struct{ *myInt }
			)
			s := S{new(myInt)}
			*s.myInt = 5
			return s
		},
		want: `{}`,
	}, {
		// Exported embedded field of pointer to non-struct type
		// should be serialized.
		CaseName: Name("ExportedEmbeddedIntPointer"),
		makeInput: func() any {
			type (
				MyInt int
				S     struct{ *MyInt }
			)
			s := S{new(MyInt)}
			*s.MyInt = 5
			return s
		},
		want: `{"MyInt":5}`,
	}, {
		// Exported fields of embedded structs should have their
		// exported fields be serialized regardless of whether the struct types
		// themselves are exported.
		CaseName: Name("EmbeddedStruct"),
		makeInput: func() any {
			type (
				s1 struct{ x, X int }
				S2 struct{ y, Y int }
				S  struct {
					s1
					S2
				}
			)
			return S{s1{1, 2}, S2{3, 4}}
		},
		want: `{"X":2,"Y":4}`,
	}, {
		// Exported fields of pointers to embedded structs should have their
		// exported fields be serialized regardless of whether the struct types
		// themselves are exported.
		CaseName: Name("EmbeddedStructPointer"),
		makeInput: func() any {
			type (
				s1 struct{ x, X int }
				S2 struct{ y, Y int }
				S  struct {
					*s1
					*S2
				}
			)
			return S{&s1{1, 2}, &S2{3, 4}}
		},
		want: `{"X":2,"Y":4}`,
	}, {
		// Exported fields on embedded unexported structs at multiple levels
		// of nesting should still be serialized.
		CaseName: Name("NestedStructAndInts"),
		makeInput: func() any {
			type (
				MyInt1 int
				MyInt2 int
				myInt  int
				s2     struct {
					MyInt2
					myInt
				}
				s1 struct {
					MyInt1
					myInt
					s2
				}
				S struct {
					s1
					myInt
				}
			)
			return S{s1{1, 2, s2{3, 4}}, 6}
		},
		want: `{"MyInt1":1,"MyInt2":3}`,
	}, {
		// If an anonymous struct pointer field is nil, we should ignore
		// the embedded fields behind it. Not properly doing so may
		// result in the wrong output or reflect panics.
		CaseName: Name("EmbeddedFieldBehindNilPointer"),
		makeInput: func() any {
			type (
				S2 struct{ Field string }
				S  struct{ *S2 }
			)
			return S{}
		},
		want: `{}`,
	}}

	for _, tt := range tests {
		t.Run(tt.Name, func(t *testing.T) {
			b, err := Marshal(tt.makeInput())
			if err != nil {
				t.Fatalf("%s: Marshal error: %v", tt.Where, err)
			}
			if string(b) != tt.want {
				t.Fatalf("%s: Marshal:\n\tgot:  %s\n\twant: %s", tt.Where, b, tt.want)
			}
		})
	}
}

type BugA struct {
	S string
}

type BugB struct {
	BugA
	S string
}

type BugC struct {
	S string
}

// Legal Go: We never use the repeated embedded field (S).
type BugX struct {
	A int
	BugA
	BugB
}

// golang.org/issue/16042.
// Even if a nil interface value is passed in, as long as
// it implements Marshaler, it should be marshaled.
type nilJSONMarshaler string

func (nm *nilJSONMarshaler) MarshalJSON() ([]byte, error) {
	if nm == nil {
		return Marshal("0zenil0")
	}
	return Marshal("zenil:" + string(*nm))
}

// golang.org/issue/34235.
// Even if a nil interface value is passed in, as long as
// it implements encoding.TextMarshaler, it should be marshaled.
type nilTextMarshaler string

func (nm *nilTextMarshaler) MarshalText() ([]byte, error) {
	if nm == nil {
		return []byte("0zenil0"), nil
	}
	return []byte("zenil:" + string(*nm)), nil
}

// See golang.org/issue/16042 and golang.org/issue/34235.
func TestNilMarshal(t *testing.T) {
	tests := []struct {
		CaseName
		in   any
		want string
	}{
		{Name(""), nil, `null`},
		{Name(""), new(float64), `0`},
		{Name(""), []any(nil), `null`},
		{Name(""), []string(nil), `null`},
		{Name(""), map[string]string(nil), `null`},
		{Name(""), []byte(nil), `null`},
		{Name(""), struct{ M string }{"gopher"}, `{"M":"gopher"}`},
		{Name(""), struct{ M Marshaler }{}, `{"M":null}`},
		{Name(""), struct{ M Marshaler }{(*nilJSONMarshaler)(nil)}, `{"M":"0zenil0"}`},
		{Name(""), struct{ M any }{(*nilJSONMarshaler)(nil)}, `{"M":null}`},
		{Name(""), struct{ M encoding.TextMarshaler }{}, `{"M":null}`},
		{Name(""), struct{ M encoding.TextMarshaler }{(*nilTextMarshaler)(nil)}, `{"M":"0zenil0"}`},
		{Name(""), struct{ M any }{(*nilTextMarshaler)(nil)}, `{"M":null}`},
	}
	for _, tt := range tests {
		t.Run(tt.Name, func(t *testing.T) {
			switch got, err := Marshal(tt.in); {
			case err != nil:
				t.Fatalf("%s: Marshal error: %v", tt.Where, err)
			case string(got) != tt.want:
				t.Fatalf("%s: Marshal:\n\tgot:  %s\n\twant: %s", tt.Where, got, tt.want)
			}
		})
	}
}

// Issue 5245.
func TestEmbeddedBug(t *testing.T) {
	v := BugB{
		BugA{"A"},
		"B",
	}
	b, err := Marshal(v)
	if err != nil {
		t.Fatal("Marshal error:", err)
	}
	want := `{"S":"B"}`
	got := string(b)
	if got != want {
		t.Fatalf("Marshal:\n\tgot:  %s\n\twant: %s", got, want)
	}
	// Now check that the duplicate field, S, does not appear.
	x := BugX{
		A: 23,
	}
	b, err = Marshal(x)
	if err != nil {
		t.Fatal("Marshal error:", err)
	}
	want = `{"A":23}`
	got = string(b)
	if got != want {
		t.Fatalf("Marshal:\n\tgot:  %s\n\twant: %s", got, want)
	}
}

type BugD struct { // Same as BugA after tagging.
	XXX string `json:"S"`
}

// BugD's tagged S field should dominate BugA's.
type BugY struct {
	BugA
	BugD
}

// Test that a field with a tag dominates untagged fields.
func TestTaggedFieldDominates(t *testing.T) {
	v := BugY{
		BugA{"BugA"},
		BugD{"BugD"},
	}
	b, err := Marshal(v)
	if err != nil {
		t.Fatal("Marshal error:", err)
	}
	want := `{"S":"BugD"}`
	got := string(b)
	if got != want {
		t.Fatalf("Marshal:\n\tgot:  %s\n\twant: %s", got, want)
	}
}

// There are no tags here, so S should not appear.
type BugZ struct {
	BugA
	BugC
	BugY // Contains a tagged S field through BugD; should not dominate.
}

func TestDuplicatedFieldDisappears(t *testing.T) {
	v := BugZ{
		BugA{"BugA"},
		BugC{"BugC"},
		BugY{
			BugA{"nested BugA"},
			BugD{"nested BugD"},
		},
	}
	b, err := Marshal(v)
	if err != nil {
		t.Fatal("Marshal error:", err)
	}
	want := `{}`
	got := string(b)
	if got != want {
		t.Fatalf("Marshal:\n\tgot:  %s\n\twant: %s", got, want)
	}
}

func TestIssue10281(t *testing.T) {
	type Foo struct {
		N Number
	}
	x := Foo{Number(`invalid`)}

	if _, err := Marshal(&x); err == nil {
		t.Fatalf("Marshal error: got nil, want non-nil")
	}
}

func TestMarshalErrorAndReuseEncodeState(t *testing.T) {
	// Disable the GC temporarily to prevent encodeState's in Pool being cleaned away during the test.
	percent := debug.SetGCPercent(-1)
	defer debug.SetGCPercent(percent)

	// Trigger an error in Marshal with cyclic data.
	type Dummy struct {
		Name string
		Next *Dummy
	}
	dummy := Dummy{Name: "Dummy"}
	dummy.Next = &dummy
	if _, err := Marshal(dummy); err == nil {
		t.Errorf("Marshal error: got nil, want non-nil")
	}

	type Data struct {
		A string
		I int
	}
	want := Data{A: "a", I: 1}
	b, err := Marshal(want)
	if err != nil {
		t.Errorf("Marshal error: %v", err)
	}

	var got Data
	if err := Unmarshal(b, &got); err != nil {
		t.Errorf("Unmarshal error: %v", err)
	}
	if got != want {
		t.Errorf("Unmarshal:\n\tgot:  %v\n\twant: %v", got, want)
	}
}

func TestHTMLEscape(t *testing.T) {
	var b, want bytes.Buffer
	m := `{"M":"<html>foo &` + "\xe2\x80\xa8 \xe2\x80\xa9" + `</html>"}`
	want.Write([]byte(`{"M":"\u003chtml\u003efoo \u0026\u2028 \u2029\u003c/html\u003e"}`))
	HTMLEscape(&b, []byte(m))
	if !bytes.Equal(b.Bytes(), want.Bytes()) {
		t.Errorf("HTMLEscape:\n\tgot:  %s\n\twant: %s", b.Bytes(), want.Bytes())
	}
}

// golang.org/issue/8582
func TestEncodePointerString(t *testing.T) {
	type stringPointer struct {
		N *int64 `json:"n,string"`
	}
	var n int64 = 42
	b, err := Marshal(stringPointer{N: &n})
	if err != nil {
		t.Fatalf("Marshal error: %v", err)
	}
	if got, want := string(b), `{"n":"42"}`; got != want {
		t.Fatalf("Marshal:\n\tgot:  %s\n\twant: %s", got, want)
	}
	var back stringPointer
	switch err = Unmarshal(b, &back); {
	case err != nil:
		t.Fatalf("Unmarshal error: %v", err)
	case back.N == nil:
		t.Fatalf("Unmarshal: back.N = nil, want non-nil")
	case *back.N != 42:
		t.Fatalf("Unmarshal: *back.N = %d, want 42", *back.N)
	}
}

var encodeStringTests = []struct {
	in  string
	out string
}{
	{"\x00", `"\u0000"`},
	{"\x01", `"\u0001"`},
	{"\x02", `"\u0002"`},
	{"\x03", `"\u0003"`},
	{"\x04", `"\u0004"`},
	{"\x05", `"\u0005"`},
	{"\x06", `"\u0006"`},
	{"\x07", `"\u0007"`},
	{"\x08", `"\b"`},
	{"\x09", `"\t"`},
	{"\x0a", `"\n"`},
	{"\x0b", `"\u000b"`},
	{"\x0c", `"\f"`},
	{"\x0d", `"\r"`},
	{"\x0e", `"\u000e"`},
	{"\x0f", `"\u000f"`},
	{"\x10", `"\u0010"`},
	{"\x11", `"\u0011"`},
	{"\x12", `"\u0012"`},
	{"\x13", `"\u0013"`},
	{"\x14", `"\u0014"`},
	{"\x15", `"\u0015"`},
	{"\x16", `"\u0016"`},
	{"\x17", `"\u0017"`},
	{"\x18", `"\u0018"`},
	{"\x19", `"\u0019"`},
	{"\x1a", `"\u001a"`},
	{"\x1b", `"\u001b"`},
	{"\x1c", `"\u001c"`},
	{"\x1d", `"\u001d"`},
	{"\x1e", `"\u001e"`},
	{"\x1f", `"\u001f"`},
}

func TestEncodeString(t *testing.T) {
	for _, tt := range encodeStringTests {
		b, err := Marshal(tt.in)
		if err != nil {
			t.Errorf("Marshal(%q) error: %v", tt.in, err)
			continue
		}
		out := string(b)
		if out != tt.out {
			t.Errorf("Marshal(%q) = %#q, want %#q", tt.in, out, tt.out)
		}
	}
}

type jsonbyte byte

func (b jsonbyte) MarshalJSON() ([]byte, error) { return tenc(`{"JB":%d}`, b) }

type textbyte byte

func (b textbyte) MarshalText() ([]byte, error) { return tenc(`TB:%d`, b) }

type jsonint int

func (i jsonint) MarshalJSON() ([]byte, error) { return tenc(`{"JI":%d}`, i) }

type textint int

func (i textint) MarshalText() ([]byte, error) { return tenc(`TI:%d`, i) }

func tenc(format string, a ...any) ([]byte, error) {
	var buf bytes.Buffer
	fmt.Fprintf(&buf, format, a...)
	return buf.Bytes(), nil
}

type textfloat float64

func (f textfloat) MarshalText() ([]byte, error) { return tenc(`TF:%0.2f`, f) }

// Issue 13783
func TestEncodeBytekind(t *testing.T) {
	tests := []struct {
		CaseName
		in   any
		want string
	}{
		{Name(""), byte(7), "7"},
		{Name(""), jsonbyte(7), `{"JB":7}`},
		{Name(""), textbyte(4), `"TB:4"`},
		{Name(""), jsonint(5), `{"JI":5}`},
		{Name(""), textint(1), `"TI:1"`},
		{Name(""), []byte{0, 1}, `"AAE="`},
		{Name(""), []jsonbyte{0, 1}, `[{"JB":0},{"JB":1}]`},
		{Name(""), [][]jsonbyte{{0, 1}, {3}}, `[[{"JB":0},{"JB":1}],[{"JB":3}]]`},
		{Name(""), []textbyte{2, 3}, `["TB:2","TB:3"]`},
		{Name(""), []jsonint{5, 4}, `[{"JI":5},{"JI":4}]`},
		{Name(""), []textint{9, 3}, `["TI:9","TI:3"]`},
		{Name(""), []int{9, 3}, `[9,3]`},
		{Name(""), []textfloat{12, 3}, `["TF:12.00","TF:3.00"]`},
	}
	for _, tt := range tests {
		t.Run(tt.Name, func(t *testing.T) {
			b, err := Marshal(tt.in)
			if err != nil {
				t.Errorf("%s: Marshal error: %v", tt.Where, err)
			}
			got, want := string(b), tt.want
			if got != want {
				t.Errorf("%s: Marshal:\n\tgot:  %s\n\twant: %s", tt.Where, got, want)
			}
		})
	}
}

func TestTextMarshalerMapKeysAreSorted(t *testing.T) {
	got, err := Marshal(map[unmarshalerText]int{
		{"x", "y"}: 1,
		{"y", "x"}: 2,
		{"a", "z"}: 3,
		{"z", "a"}: 4,
	})
	if err != nil {
		t.Fatalf("Marshal error: %v", err)
	}
	const want = `{"a:z":3,"x:y":1,"y:x":2,"z:a":4}`
	if string(got) != want {
		t.Errorf("Marshal:\n\tgot:  %s\n\twant: %s", got, want)
	}
}

// https://golang.org/issue/33675
func TestNilMarshalerTextMapKey(t *testing.T) {
	got, err := Marshal(map[*unmarshalerText]int{
		(*unmarshalerText)(nil): 1,
		{"A", "B"}:              2,
	})
	if err != nil {
		t.Fatalf("Marshal error: %v", err)
	}
	const want = `{"":1,"A:B":2}`
	if string(got) != want {
		t.Errorf("Marshal:\n\tgot:  %s\n\twant: %s", got, want)
	}
}

var re = regexp.MustCompile

// syntactic checks on form of marshaled floating point numbers.
var badFloatREs = []*regexp.Regexp{
	re(`p`),                     // no binary exponential notation
	re(`^\+`),                   // no leading + sign
	re(`^-?0[^.]`),              // no unnecessary leading zeros
	re(`^-?\.`),                 // leading zero required before decimal point
	re(`\.(e|$)`),               // no trailing decimal
	re(`\.[0-9]+0(e|$)`),        // no trailing zero in fraction
	re(`^-?(0|[0-9]{2,})\..*e`), // exponential notation must have normalized mantissa
	re(`e[0-9]`),                // positive exponent must be signed
	re(`e[+-]0`),                // exponent must not have leading zeros
	re(`e-[1-6]$`),              // not tiny enough for exponential notation
	re(`e+(.|1.|20)$`),          // not big enough for exponential notation
	re(`^-?0\.0000000`),         // too tiny, should use exponential notation
	re(`^-?[0-9]{22}`),          // too big, should use exponential notation
	re(`[1-9][0-9]{16}[1-9]`),   // too many significant digits in integer
	re(`[1-9][0-9.]{17}[1-9]`),  // too many significant digits in decimal
	// below here for float32 only
	re(`[1-9][0-9]{8}[1-9]`),  // too many significant digits in integer
	re(`[1-9][0-9.]{9}[1-9]`), // too many significant digits in decimal
}

func TestMarshalFloat(t *testing.T) {
	t.Parallel()
	nfail := 0
	test := func(f float64, bits int) {
		vf := any(f)
		if bits == 32 {
			f = float64(float32(f)) // round
			vf = float32(f)
		}
		bout, err := Marshal(vf)
		if err != nil {
			t.Errorf("Marshal(%T(%g)) error: %v", vf, vf, err)
			nfail++
			return
		}
		out := string(bout)

		// result must convert back to the same float
		g, err := strconv.ParseFloat(out, bits)
		if err != nil {
			t.Errorf("ParseFloat(%q) error: %v", out, err)
			nfail++
			return
		}
		if f != g || fmt.Sprint(f) != fmt.Sprint(g) { // fmt.Sprint handles ±0
			t.Errorf("ParseFloat(%q):\n\tgot:  %g\n\twant: %g", out, float32(g), vf)
			nfail++
			return
		}

		bad := badFloatREs
		if bits == 64 {
			bad = bad[:len(bad)-2]
		}
		for _, re := range bad {
			if re.MatchString(out) {
				t.Errorf("Marshal(%T(%g)) = %q; must not match /%s/", vf, vf, out, re)
				nfail++
				return
			}
		}
	}

	var (
		bigger  = math.Inf(+1)
		smaller = math.Inf(-1)
	)

	var digits = "1.2345678901234567890123"
	for i := len(digits); i >= 2; i-- {
		if testing.Short() && i < len(digits)-4 {
			break
		}
		for exp := -30; exp <= 30; exp++ {
			for _, sign := range "+-" {
				for bits := 32; bits <= 64; bits += 32 {
					s := fmt.Sprintf("%c%se%d", sign, digits[:i], exp)
					f, err := strconv.ParseFloat(s, bits)
					if err != nil {
						log.Fatal(err)
					}
					next := math.Nextafter
					if bits == 32 {
						next = func(g, h float64) float64 {
							return float64(math.Nextafter32(float32(g), float32(h)))
						}
					}
					test(f, bits)
					test(next(f, bigger), bits)
					test(next(f, smaller), bits)
					if nfail > 50 {
						t.Fatalf("stopping test early")
					}
				}
			}
		}
	}
	test(0, 64)
	test(math.Copysign(0, -1), 64)
	test(0, 32)
	test(math.Copysign(0, -1), 32)
}

func TestMarshalRawMessageValue(t *testing.T) {
	type (
		T1 struct {
			M RawMessage `json:",omitempty"`
		}
		T2 struct {
			M *RawMessage `json:",omitempty"`
		}
	)

	var (
		rawNil   = RawMessage(nil)
		rawEmpty = RawMessage([]byte{})
		rawText  = RawMessage([]byte(`"foo"`))
	)

	tests := []struct {
		CaseName
		in   any
		want string
		ok   bool
	}{
		// Test with nil RawMessage.
		{Name(""), rawNil, "null", true},
		{Name(""), &rawNil, "null", true},
		{Name(""), []any{rawNil}, "[null]", true},
		{Name(""), &[]any{rawNil}, "[null]", true},
		{Name(""), []any{&rawNil}, "[null]", true},
		{Name(""), &[]any{&rawNil}, "[null]", true},
		{Name(""), struct{ M RawMessage }{rawNil}, `{"M":null}`, true},
		{Name(""), &struct{ M RawMessage }{rawNil}, `{"M":null}`, true},
		{Name(""), struct{ M *RawMessage }{&rawNil}, `{"M":null}`, true},
		{Name(""), &struct{ M *RawMessage }{&rawNil}, `{"M":null}`, true},
		{Name(""), map[string]any{"M": rawNil}, `{"M":null}`, true},
		{Name(""), &map[string]any{"M": rawNil}, `{"M":null}`, true},
		{Name(""), map[string]any{"M": &rawNil}, `{"M":null}`, true},
		{Name(""), &map[string]any{"M": &rawNil}, `{"M":null}`, true},
		{Name(""), T1{rawNil}, "{}", true},
		{Name(""), T2{&rawNil}, `{"M":null}`, true},
		{Name(""), &T1{rawNil}, "{}", true},
		{Name(""), &T2{&rawNil}, `{"M":null}`, true},

		// Test with empty, but non-nil, RawMessage.
		{Name(""), rawEmpty, "", false},
		{Name(""), &rawEmpty, "", false},
		{Name(""), []any{rawEmpty}, "", false},
		{Name(""), &[]any{rawEmpty}, "", false},
		{Name(""), []any{&rawEmpty}, "", false},
		{Name(""), &[]any{&rawEmpty}, "", false},
		{Name(""), struct{ X RawMessage }{rawEmpty}, "", false},
		{Name(""), &struct{ X RawMessage }{rawEmpty}, "", false},
		{Name(""), struct{ X *RawMessage }{&rawEmpty}, "", false},
		{Name(""), &struct{ X *RawMessage }{&rawEmpty}, "", false},
		{Name(""), map[string]any{"nil": rawEmpty}, "", false},
		{Name(""), &map[string]any{"nil": rawEmpty}, "", false},
		{Name(""), map[string]any{"nil": &rawEmpty}, "", false},
		{Name(""), &map[string]any{"nil": &rawEmpty}, "", false},
		{Name(""), T1{rawEmpty}, "{}", true},
		{Name(""), T2{&rawEmpty}, "", false},
		{Name(""), &T1{rawEmpty}, "{}", true},
		{Name(""), &T2{&rawEmpty}, "", false},

		// Test with RawMessage with some text.
		//
		// The tests below marked with Issue6458 used to generate "ImZvbyI=" instead "foo".
		// This behavior was intentionally changed in Go 1.8.
		// See https://golang.org/issues/14493#issuecomment-255857318
		{Name(""), rawText, `"foo"`, true}, // Issue6458
		{Name(""), &rawText, `"foo"`, true},
		{Name(""), []any{rawText}, `["foo"]`, true},  // Issue6458
		{Name(""), &[]any{rawText}, `["foo"]`, true}, // Issue6458
		{Name(""), []any{&rawText}, `["foo"]`, true},
		{Name(""), &[]any{&rawText}, `["foo"]`, true},
		{Name(""), struct{ M RawMessage }{rawText}, `{"M":"foo"}`, true}, // Issue6458
		{Name(""), &struct{ M RawMessage }{rawText}, `{"M":"foo"}`, true},
		{Name(""), struct{ M *RawMessage }{&rawText}, `{"M":"foo"}`, true},
		{Name(""), &struct{ M *RawMessage }{&rawText}, `{"M":"foo"}`, true},
		{Name(""), map[string]any{"M": rawText}, `{"M":"foo"}`, true},  // Issue6458
		{Name(""), &map[string]any{"M": rawText}, `{"M":"foo"}`, true}, // Issue6458
		{Name(""), map[string]any{"M": &rawText}, `{"M":"foo"}`, true},
		{Name(""), &map[string]any{"M": &rawText}, `{"M":"foo"}`, true},
		{Name(""), T1{rawText}, `{"M":"foo"}`, true}, // Issue6458
		{Name(""), T2{&rawText}, `{"M":"foo"}`, true},
		{Name(""), &T1{rawText}, `{"M":"foo"}`, true},
		{Name(""), &T2{&rawText}, `{"M":"foo"}`, true},
	}

	for _, tt := range tests {
		t.Run(tt.Name, func(t *testing.T) {
			b, err := Marshal(tt.in)
			if ok := (err == nil); ok != tt.ok {
				if err != nil {
					t.Errorf("%s: Marshal error: %v", tt.Where, err)
				} else {
					t.Errorf("%s: Marshal error: got nil, want non-nil", tt.Where)
				}
			}
			if got := string(b); got != tt.want {
				t.Errorf("%s: Marshal:\n\tinput: %#v\n\tgot:  %s\n\twant: %s", tt.Where, tt.in, got, tt.want)
			}
		})
	}
}

type marshalPanic struct{}

func (marshalPanic) MarshalJSON() ([]byte, error) { panic(0xdead) }

func TestMarshalPanic(t *testing.T) {
	defer func() {
		if got := recover(); !reflect.DeepEqual(got, 0xdead) {
			t.Errorf("panic() = (%T)(%v), want 0xdead", got, got)
		}
	}()
	Marshal(&marshalPanic{})
	t.Error("Marshal should have panicked")
}

func TestMarshalUncommonFieldNames(t *testing.T) {
	v := struct {
		A0, À, Aβ int
	}{}
	b, err := Marshal(v)
	if err != nil {
		t.Fatal("Marshal error:", err)
	}
	want := `{"A0":0,"À":0,"Aβ":0}`
	got := string(b)
	if got != want {
		t.Fatalf("Marshal:\n\tgot:  %s\n\twant: %s", got, want)
	}
}

func TestMarshalerError(t *testing.T) {
	s := "test variable"
	st := reflect.TypeOf(s)
	const errText = "json: test error"

	tests := []struct {
		CaseName
		err  *MarshalerError
		want string
	}{{
		Name(""),
		&MarshalerError{st, fmt.Errorf(errText), ""},
		"json: error calling MarshalJSON for type " + st.String() + ": " + errText,
	}, {
		Name(""),
		&MarshalerError{st, fmt.Errorf(errText), "TestMarshalerError"},
		"json: error calling TestMarshalerError for type " + st.String() + ": " + errText,
	}}

	for _, tt := range tests {
		t.Run(tt.Name, func(t *testing.T) {
			got := tt.err.Error()
			if got != tt.want {
				t.Errorf("%s: Error:\n\tgot:  %s\n\twant: %s", tt.Where, got, tt.want)
			}
		})
	}
}

type marshaledValue string

func (v marshaledValue) MarshalJSON() ([]byte, error) {
	return []byte(v), nil
}

func TestIssue63379(t *testing.T) {
	for _, v := range []string{
		"[]<",
		"[]>",
		"[]&",
		"[]\u2028",
		"[]\u2029",
		"{}<",
		"{}>",
		"{}&",
		"{}\u2028",
		"{}\u2029",
	} {
		_, err := Marshal(marshaledValue(v))
		if err == nil {
			t.Errorf("expected error for %q", v)
		}
	}
}

type structWithMarshalJSON struct{ v int }

func (s *structWithMarshalJSON) MarshalJSON() ([]byte, error) {
	return []byte(fmt.Sprintf(`"marshalled(%d)"`, s.v)), nil
}

var _ = Marshaler(&structWithMarshalJSON{})

type embedder struct {
	V interface{}
}

type structWithMarshalText struct{ v int }

func (s *structWithMarshalText) MarshalText() ([]byte, error) {
	return []byte(fmt.Sprintf("marshalled(%d)", s.v)), nil
}

var _ = encoding.TextMarshaler(&structWithMarshalText{})

func TestMarshalJSONWithPointerMarshalers(t *testing.T) {
	for _, test := range []struct {
		name                     string
		jsoninconsistentmarshal  bool
		v                        interface{}
		expected                 string
		expectedOldBehaviorCount uint64
		expectedError            string
	}{
		// MarshalJSON
		{name: "a value with MarshalJSON", v: structWithMarshalJSON{v: 1}, expected: `"marshalled(1)"`},
		{name: "pointer to a value with MarshalJSON", v: &structWithMarshalJSON{v: 1}, expected: `"marshalled(1)"`},
		{
			name:     "a map with a value with MarshalJSON",
			v:        map[string]interface{}{"v": structWithMarshalJSON{v: 1}},
			expected: `{"v":"marshalled(1)"}`,
		},
		{
			name:     "a map with a pointer to a value with MarshalJSON",
			v:        map[string]interface{}{"v": &structWithMarshalJSON{v: 1}},
			expected: `{"v":"marshalled(1)"}`,
		},
		{
			name:     "a slice of maps with a value with MarshalJSON",
			v:        []map[string]interface{}{{"v": structWithMarshalJSON{v: 1}}},
			expected: `[{"v":"marshalled(1)"}]`,
		},
		{
			name:     "a slice of maps with a pointer to a value with MarshalJSON",
			v:        []map[string]interface{}{{"v": &structWithMarshalJSON{v: 1}}},
			expected: `[{"v":"marshalled(1)"}]`,
		},
		{
			name:     "a struct with a value with MarshalJSON",
			v:        embedder{V: structWithMarshalJSON{v: 1}},
			expected: `{"V":"marshalled(1)"}`,
		},
		{
			name:     "a slice of structs with a value with MarshalJSON",
			v:        []embedder{{V: structWithMarshalJSON{v: 1}}},
			expected: `[{"V":"marshalled(1)"}]`,
		},
		{
			name:                     "a value with MarshalJSON (only addressable)",
			jsoninconsistentmarshal:  true,
			v:                        structWithMarshalJSON{v: 1},
			expected:                 `{}`,
			expectedOldBehaviorCount: 1,
		},
		{
			name:                    "pointer to a value with MarshalJSON (only addressable)",
			jsoninconsistentmarshal: true,
			v:                       &structWithMarshalJSON{v: 1},
			expected:                `"marshalled(1)"`,
		},
		{
			name:                     "a map with a value with MarshalJSON (only addressable)",
			jsoninconsistentmarshal:  true,
			v:                        map[string]interface{}{"v": structWithMarshalJSON{v: 1}},
			expected:                 `{"v":{}}`,
			expectedOldBehaviorCount: 1,
		},
		{
			name:                    "a map with a pointer to a value with MarshalJSON (only addressable)",
			jsoninconsistentmarshal: true,
			v:                       map[string]interface{}{"v": &structWithMarshalJSON{v: 1}},
			expected:                `{"v":"marshalled(1)"}`,
		},
		{
			name:                     "a slice of maps with a value with MarshalJSON (only addressable)",
			jsoninconsistentmarshal:  true,
			v:                        []map[string]interface{}{{"v": structWithMarshalJSON{v: 1}}},
			expected:                 `[{"v":{}}]`,
			expectedOldBehaviorCount: 1,
		},
		{
			name:                    "a slice of maps with a pointer to a value with MarshalJSON (only addressable)",
			jsoninconsistentmarshal: true,
			v:                       []map[string]interface{}{{"v": &structWithMarshalJSON{v: 1}}},
			expected:                `[{"v":"marshalled(1)"}]`,
		},
		{
			name:                     "a struct with a value with MarshalJSON (only addressable)",
			jsoninconsistentmarshal:  true,
			v:                        embedder{V: structWithMarshalJSON{v: 1}},
			expected:                 `{"V":{}}`,
			expectedOldBehaviorCount: 1,
		},
		{
			name:                     "a slice of structs with a value with MarshalJSON (only addressable)",
			jsoninconsistentmarshal:  true,
			v:                        []embedder{{V: structWithMarshalJSON{v: 1}}},
			expected:                 `[{"V":{}}]`,
			expectedOldBehaviorCount: 1,
		},
		{
			name:                     "a slice of structs with a value with MarshalJSON with two elements (only addressable)",
			jsoninconsistentmarshal:  true,
			v:                        []embedder{{V: structWithMarshalJSON{v: 1}}, {V: structWithMarshalJSON{v: 2}}},
			expected:                 `[{"V":{}},{"V":{}}]`,
			expectedOldBehaviorCount: 2,
		},
		// MarshalText
		{name: "a value with MarshalText", v: structWithMarshalText{v: 1}, expected: `"marshalled(1)"`},
		{name: "pointer to a value with MarshalText", v: &structWithMarshalText{v: 1}, expected: `"marshalled(1)"`},
		{name: "a map with a value with MarshalText", v: map[string]interface{}{"v": structWithMarshalText{v: 1}}, expected: `{"v":"marshalled(1)"}`},
		{
			name:     "a map with a pointer to a value with MarshalText",
			v:        map[string]interface{}{"v": &structWithMarshalText{v: 1}},
			expected: `{"v":"marshalled(1)"}`,
		},
		{
			name:     "a slice of maps with a value with MarshalText",
			v:        []map[string]interface{}{{"v": structWithMarshalText{v: 1}}},
			expected: `[{"v":"marshalled(1)"}]`,
		},
		{
			name:     "a slice of maps with a pointer to a value with MarshalText",
			v:        []map[string]interface{}{{"v": &structWithMarshalText{v: 1}}},
			expected: `[{"v":"marshalled(1)"}]`,
		},
		{
			name:     "a struct with a value with MarshalText",
			v:        embedder{V: structWithMarshalText{v: 1}},
			expected: `{"V":"marshalled(1)"}`,
		},
		{
			name:     "a slice of structs with a value with MarshalText",
			v:        []embedder{{V: structWithMarshalText{v: 1}}},
			expected: `[{"V":"marshalled(1)"}]`,
		},
		{
			name:                     "a value with MarshalText (only addressable)",
			jsoninconsistentmarshal:  true,
			v:                        structWithMarshalText{v: 1},
			expected:                 `{}`,
			expectedOldBehaviorCount: 1,
		},
		{
			name:                    "pointer to a value with MarshalText (only addressable)",
			jsoninconsistentmarshal: true,
			v:                       &structWithMarshalText{v: 1},
			expected:                `"marshalled(1)"`,
		},
		{
			name:                     "a map with a value with MarshalText (only addressable)",
			jsoninconsistentmarshal:  true,
			v:                        map[string]interface{}{"v": structWithMarshalText{v: 1}},
			expected:                 `{"v":{}}`,
			expectedOldBehaviorCount: 1,
		},
		{
			name:                    "a map with a pointer to a value with MarshalText (only addressable)",
			jsoninconsistentmarshal: true,
			v:                       map[string]interface{}{"v": &structWithMarshalText{v: 1}},
			expected:                `{"v":"marshalled(1)"}`,
		},
		{
			name:                     "a slice of maps with a value with MarshalText (only addressable)",
			jsoninconsistentmarshal:  true,
			v:                        []map[string]interface{}{{"v": structWithMarshalText{v: 1}}},
			expected:                 `[{"v":{}}]`,
			expectedOldBehaviorCount: 1,
		},
		{
			name:                    "a slice of maps with a pointer to a value with MarshalText (only addressable)",
			jsoninconsistentmarshal: true,
			v:                       []map[string]interface{}{{"v": &structWithMarshalText{v: 1}}},
			expected:                `[{"v":"marshalled(1)"}]`,
		},
		{
			name:                     "a struct with a value with MarshalText (only addressable)",
			jsoninconsistentmarshal:  true,
			v:                        embedder{V: structWithMarshalText{v: 1}},
			expected:                 `{"V":{}}`,
			expectedOldBehaviorCount: 1,
		},
		{
			name:                     "a slice of structs with a value with MarshalText (only addressable)",
			jsoninconsistentmarshal:  true,
			v:                        []embedder{{V: structWithMarshalText{v: 1}}},
			expected:                 `[{"V":{}}]`,
			expectedOldBehaviorCount: 1,
		},
	} {
		test := test
		t.Run(test.name, func(t *testing.T) {
			const metricName = "/godebug/non-default-behavior/jsoninconsistentmarshal:events"
			sample := make([]metrics.Sample, 1)
			sample[0].Name = metricName
			metrics.Read(sample)
			metricOldValue := sample[0].Value.Uint64()

			if test.jsoninconsistentmarshal {
				os.Setenv("GODEBUG", "jsoninconsistentmarshal=1")
				defer os.Unsetenv("GODEBUG")
			}
			result, err := Marshal(test.v)

			metrics.Read(sample)
			metricNewValue := sample[0].Value.Uint64()
			oldBehaviorCount := metricNewValue - metricOldValue

			if oldBehaviorCount != test.expectedOldBehaviorCount {
				t.Errorf("The old behavior count is %d, want %d", oldBehaviorCount, test.expectedOldBehaviorCount)
			}

			if err != nil {
				if test.expectedError != "" {
					if err.Error() != test.expectedError {
						t.Errorf("Unexpected Marshal error: %s, expected: %s", err.Error(), test.expectedError)
					}
					return
				}
				t.Fatalf("Unexpected Marshal error: %v", err)
			}

			if string(result) != test.expected {
				t.Errorf("Marshal:\n\tgot:  %s\n\twant: %s", result, test.expected)
			}
		})
	}
}
