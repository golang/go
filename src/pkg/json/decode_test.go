// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	"bytes"
	"os"
	"reflect"
	"strings"
	"testing"
)

type T struct {
	X string
	Y int
}

type tx struct {
	x int
}

var txType = reflect.Typeof((*tx)(nil)).(*reflect.PtrType).Elem().(*reflect.StructType)

// A type that can unmarshal itself.

type unmarshaler struct {
	T bool
}

func (u *unmarshaler) UnmarshalJSON(b []byte) os.Error {
	*u = unmarshaler{true} // All we need to see that UnmarshalJson is called.
	return nil
}

var (
	um0, um1 unmarshaler // target2 of unmarshaling
	ump      = &um1
	umtrue   = unmarshaler{true}
)


type unmarshalTest struct {
	in  string
	ptr interface{}
	out interface{}
	err os.Error
}

var unmarshalTests = []unmarshalTest{
	// basic types
	{`true`, new(bool), true, nil},
	{`1`, new(int), 1, nil},
	{`1.2`, new(float64), 1.2, nil},
	{`-5`, new(int16), int16(-5), nil},
	{`"a\u1234"`, new(string), "a\u1234", nil},
	{`"http:\/\/"`, new(string), "http://", nil},
	{`"g-clef: \uD834\uDD1E"`, new(string), "g-clef: \U0001D11E", nil},
	{`"invalid: \uD834x\uDD1E"`, new(string), "invalid: \uFFFDx\uFFFD", nil},
	{"null", new(interface{}), nil, nil},
	{`{"X": [1,2,3], "Y": 4}`, new(T), T{Y: 4}, &UnmarshalTypeError{"array", reflect.Typeof("")}},
	{`{"x": 1}`, new(tx), tx{}, &UnmarshalFieldError{"x", txType, txType.Field(0)}},

	// syntax errors
	{`{"X": "foo", "Y"}`, nil, nil, SyntaxError("invalid character '}' after object key")},

	// composite tests
	{allValueIndent, new(All), allValue, nil},
	{allValueCompact, new(All), allValue, nil},
	{allValueIndent, new(*All), &allValue, nil},
	{allValueCompact, new(*All), &allValue, nil},
	{pallValueIndent, new(All), pallValue, nil},
	{pallValueCompact, new(All), pallValue, nil},
	{pallValueIndent, new(*All), &pallValue, nil},
	{pallValueCompact, new(*All), &pallValue, nil},

	// unmarshal interface test
	{`{"T":false}`, &um0, umtrue, nil}, // use "false" so test will fail if custom unmarshaler is not called
	{`{"T":false}`, &ump, &umtrue, nil},
}

func TestMarshal(t *testing.T) {
	b, err := Marshal(allValue)
	if err != nil {
		t.Fatalf("Marshal allValue: %v", err)
	}
	if string(b) != allValueCompact {
		t.Errorf("Marshal allValueCompact")
		diff(t, b, []byte(allValueCompact))
		return
	}

	b, err = Marshal(pallValue)
	if err != nil {
		t.Fatalf("Marshal pallValue: %v", err)
	}
	if string(b) != pallValueCompact {
		t.Errorf("Marshal pallValueCompact")
		diff(t, b, []byte(pallValueCompact))
		return
	}
}

func TestMarshalBadUTF8(t *testing.T) {
	s := "hello\xffworld"
	b, err := Marshal(s)
	if err == nil {
		t.Fatal("Marshal bad UTF8: no error")
	}
	if len(b) != 0 {
		t.Fatal("Marshal returned data")
	}
	if _, ok := err.(*InvalidUTF8Error); !ok {
		t.Fatalf("Marshal did not return InvalidUTF8Error: %T %v", err, err)
	}
}

func TestUnmarshal(t *testing.T) {
	var scan scanner
	for i, tt := range unmarshalTests {
		in := []byte(tt.in)
		if err := checkValid(in, &scan); err != nil {
			if !reflect.DeepEqual(err, tt.err) {
				t.Errorf("#%d: checkValid: %v", i, err)
				continue
			}
		}
		if tt.ptr == nil {
			continue
		}
		// v = new(right-type)
		v := reflect.NewValue(tt.ptr).(*reflect.PtrValue)
		v.PointTo(reflect.MakeZero(v.Type().(*reflect.PtrType).Elem()))
		if err := Unmarshal([]byte(in), v.Interface()); !reflect.DeepEqual(err, tt.err) {
			t.Errorf("#%d: %v want %v", i, err, tt.err)
			continue
		}
		if !reflect.DeepEqual(v.Elem().Interface(), tt.out) {
			t.Errorf("#%d: mismatch\nhave: %#+v\nwant: %#+v", i, v.Elem().Interface(), tt.out)
			data, _ := Marshal(v.Elem().Interface())
			println(string(data))
			data, _ = Marshal(tt.out)
			println(string(data))
			return
			continue
		}
	}
}

func TestUnmarshalMarshal(t *testing.T) {
	var v interface{}
	if err := Unmarshal(jsonBig, &v); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	b, err := Marshal(v)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if bytes.Compare(jsonBig, b) != 0 {
		t.Errorf("Marshal jsonBig")
		diff(t, b, jsonBig)
		return
	}
}

type Xint struct {
	X int
}

func TestUnmarshalInterface(t *testing.T) {
	var xint Xint
	var i interface{} = &xint
	if err := Unmarshal([]byte(`{"X":1}`), &i); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if xint.X != 1 {
		t.Fatalf("Did not write to xint")
	}
}

func TestUnmarshalPtrPtr(t *testing.T) {
	var xint Xint
	pxint := &xint
	if err := Unmarshal([]byte(`{"X":1}`), &pxint); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if xint.X != 1 {
		t.Fatalf("Did not write to xint")
	}
}

func TestHTMLEscape(t *testing.T) {
	b, err := MarshalForHTML("foobarbaz<>&quux")
	if err != nil {
		t.Fatalf("MarshalForHTML error: %v", err)
	}
	if !bytes.Equal(b, []byte(`"foobarbaz\u003c\u003e\u0026quux"`)) {
		t.Fatalf("Unexpected encoding of \"<>&\": %s", b)
	}
}

func noSpace(c int) int {
	if isSpace(c) {
		return -1
	}
	return c
}

type All struct {
	Bool    bool
	Int     int
	Int8    int8
	Int16   int16
	Int32   int32
	Int64   int64
	Uint    uint
	Uint8   uint8
	Uint16  uint16
	Uint32  uint32
	Uint64  uint64
	Uintptr uintptr
	Float32 float32
	Float64 float64

	Foo string "bar"

	PBool    *bool
	PInt     *int
	PInt8    *int8
	PInt16   *int16
	PInt32   *int32
	PInt64   *int64
	PUint    *uint
	PUint8   *uint8
	PUint16  *uint16
	PUint32  *uint32
	PUint64  *uint64
	PUintptr *uintptr
	PFloat32 *float32
	PFloat64 *float64

	String  string
	PString *string

	Map   map[string]Small
	MapP  map[string]*Small
	PMap  *map[string]Small
	PMapP *map[string]*Small

	EmptyMap map[string]Small
	NilMap   map[string]Small

	Slice   []Small
	SliceP  []*Small
	PSlice  *[]Small
	PSliceP *[]*Small

	EmptySlice []Small
	NilSlice   []Small

	StringSlice []string
	ByteSlice   []byte

	Small   Small
	PSmall  *Small
	PPSmall **Small

	Interface  interface{}
	PInterface *interface{}

	unexported int
}

type Small struct {
	Tag string
}

var allValue = All{
	Bool:    true,
	Int:     2,
	Int8:    3,
	Int16:   4,
	Int32:   5,
	Int64:   6,
	Uint:    7,
	Uint8:   8,
	Uint16:  9,
	Uint32:  10,
	Uint64:  11,
	Uintptr: 12,
	Float32: 14.1,
	Float64: 15.1,
	Foo:     "foo",
	String:  "16",
	Map: map[string]Small{
		"17": {Tag: "tag17"},
		"18": {Tag: "tag18"},
	},
	MapP: map[string]*Small{
		"19": &Small{Tag: "tag19"},
		"20": nil,
	},
	EmptyMap:    map[string]Small{},
	Slice:       []Small{{Tag: "tag20"}, {Tag: "tag21"}},
	SliceP:      []*Small{&Small{Tag: "tag22"}, nil, &Small{Tag: "tag23"}},
	EmptySlice:  []Small{},
	StringSlice: []string{"str24", "str25", "str26"},
	ByteSlice:   []byte{27, 28, 29},
	Small:       Small{Tag: "tag30"},
	PSmall:      &Small{Tag: "tag31"},
	Interface:   5.2,
}

var pallValue = All{
	PBool:      &allValue.Bool,
	PInt:       &allValue.Int,
	PInt8:      &allValue.Int8,
	PInt16:     &allValue.Int16,
	PInt32:     &allValue.Int32,
	PInt64:     &allValue.Int64,
	PUint:      &allValue.Uint,
	PUint8:     &allValue.Uint8,
	PUint16:    &allValue.Uint16,
	PUint32:    &allValue.Uint32,
	PUint64:    &allValue.Uint64,
	PUintptr:   &allValue.Uintptr,
	PFloat32:   &allValue.Float32,
	PFloat64:   &allValue.Float64,
	PString:    &allValue.String,
	PMap:       &allValue.Map,
	PMapP:      &allValue.MapP,
	PSlice:     &allValue.Slice,
	PSliceP:    &allValue.SliceP,
	PPSmall:    &allValue.PSmall,
	PInterface: &allValue.Interface,
}

var allValueIndent = `{
	"Bool": true,
	"Int": 2,
	"Int8": 3,
	"Int16": 4,
	"Int32": 5,
	"Int64": 6,
	"Uint": 7,
	"Uint8": 8,
	"Uint16": 9,
	"Uint32": 10,
	"Uint64": 11,
	"Uintptr": 12,
	"Float32": 14.1,
	"Float64": 15.1,
	"bar": "foo",
	"PBool": null,
	"PInt": null,
	"PInt8": null,
	"PInt16": null,
	"PInt32": null,
	"PInt64": null,
	"PUint": null,
	"PUint8": null,
	"PUint16": null,
	"PUint32": null,
	"PUint64": null,
	"PUintptr": null,
	"PFloat32": null,
	"PFloat64": null,
	"String": "16",
	"PString": null,
	"Map": {
		"17": {
			"Tag": "tag17"
		},
		"18": {
			"Tag": "tag18"
		}
	},
	"MapP": {
		"19": {
			"Tag": "tag19"
		},
		"20": null
	},
	"PMap": null,
	"PMapP": null,
	"EmptyMap": {},
	"NilMap": null,
	"Slice": [
		{
			"Tag": "tag20"
		},
		{
			"Tag": "tag21"
		}
	],
	"SliceP": [
		{
			"Tag": "tag22"
		},
		null,
		{
			"Tag": "tag23"
		}
	],
	"PSlice": null,
	"PSliceP": null,
	"EmptySlice": [],
	"NilSlice": [],
	"StringSlice": [
		"str24",
		"str25",
		"str26"
	],
	"ByteSlice": [
		27,
		28,
		29
	],
	"Small": {
		"Tag": "tag30"
	},
	"PSmall": {
		"Tag": "tag31"
	},
	"PPSmall": null,
	"Interface": 5.2,
	"PInterface": null
}`

var allValueCompact = strings.Map(noSpace, allValueIndent)

var pallValueIndent = `{
	"Bool": false,
	"Int": 0,
	"Int8": 0,
	"Int16": 0,
	"Int32": 0,
	"Int64": 0,
	"Uint": 0,
	"Uint8": 0,
	"Uint16": 0,
	"Uint32": 0,
	"Uint64": 0,
	"Uintptr": 0,
	"Float32": 0,
	"Float64": 0,
	"bar": "",
	"PBool": true,
	"PInt": 2,
	"PInt8": 3,
	"PInt16": 4,
	"PInt32": 5,
	"PInt64": 6,
	"PUint": 7,
	"PUint8": 8,
	"PUint16": 9,
	"PUint32": 10,
	"PUint64": 11,
	"PUintptr": 12,
	"PFloat32": 14.1,
	"PFloat64": 15.1,
	"String": "",
	"PString": "16",
	"Map": null,
	"MapP": null,
	"PMap": {
		"17": {
			"Tag": "tag17"
		},
		"18": {
			"Tag": "tag18"
		}
	},
	"PMapP": {
		"19": {
			"Tag": "tag19"
		},
		"20": null
	},
	"EmptyMap": null,
	"NilMap": null,
	"Slice": [],
	"SliceP": [],
	"PSlice": [
		{
			"Tag": "tag20"
		},
		{
			"Tag": "tag21"
		}
	],
	"PSliceP": [
		{
			"Tag": "tag22"
		},
		null,
		{
			"Tag": "tag23"
		}
	],
	"EmptySlice": [],
	"NilSlice": [],
	"StringSlice": [],
	"ByteSlice": [],
	"Small": {
		"Tag": ""
	},
	"PSmall": null,
	"PPSmall": {
		"Tag": "tag31"
	},
	"Interface": null,
	"PInterface": 5.2
}`

var pallValueCompact = strings.Map(noSpace, pallValueIndent)
