// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	"bytes"
	"reflect"
	"strings"
	"testing"
)

type unmarshalTest struct {
	in  string
	ptr interface{}
	out interface{}
}

var unmarshalTests = []unmarshalTest{
	// basic types
	unmarshalTest{`true`, new(bool), true},
	unmarshalTest{`1`, new(int), 1},
	unmarshalTest{`1.2`, new(float), 1.2},
	unmarshalTest{`-5`, new(int16), int16(-5)},
	unmarshalTest{`"a\u1234"`, new(string), "a\u1234"},
	unmarshalTest{`"g-clef: \uD834\uDD1E"`, new(string), "g-clef: \U0001D11E"},
	unmarshalTest{`"invalid: \uD834x\uDD1E"`, new(string), "invalid: \uFFFDx\uFFFD"},
	unmarshalTest{"null", new(interface{}), nil},

	// composite tests
	unmarshalTest{allValueIndent, new(All), allValue},
	unmarshalTest{allValueCompact, new(All), allValue},
	unmarshalTest{allValueIndent, new(*All), &allValue},
	unmarshalTest{allValueCompact, new(*All), &allValue},
	unmarshalTest{pallValueIndent, new(All), pallValue},
	unmarshalTest{pallValueCompact, new(All), pallValue},
	unmarshalTest{pallValueIndent, new(*All), &pallValue},
	unmarshalTest{pallValueCompact, new(*All), &pallValue},
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

func TestUnmarshal(t *testing.T) {
	var scan scanner
	for i, tt := range unmarshalTests {
		in := []byte(tt.in)
		if err := checkValid(in, &scan); err != nil {
			t.Errorf("#%d: checkValid: %v", i, err)
			continue
		}
		// v = new(right-type)
		v := reflect.NewValue(tt.ptr).(*reflect.PtrValue)
		v.PointTo(reflect.MakeZero(v.Type().(*reflect.PtrType).Elem()))
		if err := Unmarshal([]byte(in), v.Interface()); err != nil {
			t.Errorf("#%d: %v", i, err)
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
	Float   float
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
	PFloat   *float
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
	Float:   13.1,
	Float32: 14.1,
	Float64: 15.1,
	Foo:     "foo",
	String:  "16",
	Map: map[string]Small{
		"17": Small{Tag: "tag17"},
		"18": Small{Tag: "tag18"},
	},
	MapP: map[string]*Small{
		"19": &Small{Tag: "tag19"},
		"20": nil,
	},
	EmptyMap:    map[string]Small{},
	Slice:       []Small{Small{Tag: "tag20"}, Small{Tag: "tag21"}},
	SliceP:      []*Small{&Small{Tag: "tag22"}, nil, &Small{Tag: "tag23"}},
	EmptySlice:  []Small{},
	StringSlice: []string{"str24", "str25", "str26"},
	ByteSlice:   []byte{27, 28, 29},
	Small:       Small{Tag: "tag30"},
	PSmall:      &Small{Tag: "tag31"},
	Interface:   float64(5.2),
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
	PFloat:     &allValue.Float,
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
	"bool": true,
	"int": 2,
	"int8": 3,
	"int16": 4,
	"int32": 5,
	"int64": 6,
	"uint": 7,
	"uint8": 8,
	"uint16": 9,
	"uint32": 10,
	"uint64": 11,
	"uintptr": 12,
	"float": 13.1,
	"float32": 14.1,
	"float64": 15.1,
	"bar": "foo",
	"pbool": null,
	"pint": null,
	"pint8": null,
	"pint16": null,
	"pint32": null,
	"pint64": null,
	"puint": null,
	"puint8": null,
	"puint16": null,
	"puint32": null,
	"puint64": null,
	"puintptr": null,
	"pfloat": null,
	"pfloat32": null,
	"pfloat64": null,
	"string": "16",
	"pstring": null,
	"map": {
		"17": {
			"tag": "tag17"
		},
		"18": {
			"tag": "tag18"
		}
	},
	"mapp": {
		"19": {
			"tag": "tag19"
		},
		"20": null
	},
	"pmap": null,
	"pmapp": null,
	"emptymap": {},
	"nilmap": null,
	"slice": [
		{
			"tag": "tag20"
		},
		{
			"tag": "tag21"
		}
	],
	"slicep": [
		{
			"tag": "tag22"
		},
		null,
		{
			"tag": "tag23"
		}
	],
	"pslice": null,
	"pslicep": null,
	"emptyslice": [],
	"nilslice": [],
	"stringslice": [
		"str24",
		"str25",
		"str26"
	],
	"byteslice": [
		27,
		28,
		29
	],
	"small": {
		"tag": "tag30"
	},
	"psmall": {
		"tag": "tag31"
	},
	"ppsmall": null,
	"interface": 5.2,
	"pinterface": null
}`

var allValueCompact = strings.Map(noSpace, allValueIndent)

var pallValueIndent = `{
	"bool": false,
	"int": 0,
	"int8": 0,
	"int16": 0,
	"int32": 0,
	"int64": 0,
	"uint": 0,
	"uint8": 0,
	"uint16": 0,
	"uint32": 0,
	"uint64": 0,
	"uintptr": 0,
	"float": 0,
	"float32": 0,
	"float64": 0,
	"bar": "",
	"pbool": true,
	"pint": 2,
	"pint8": 3,
	"pint16": 4,
	"pint32": 5,
	"pint64": 6,
	"puint": 7,
	"puint8": 8,
	"puint16": 9,
	"puint32": 10,
	"puint64": 11,
	"puintptr": 12,
	"pfloat": 13.1,
	"pfloat32": 14.1,
	"pfloat64": 15.1,
	"string": "",
	"pstring": "16",
	"map": null,
	"mapp": null,
	"pmap": {
		"17": {
			"tag": "tag17"
		},
		"18": {
			"tag": "tag18"
		}
	},
	"pmapp": {
		"19": {
			"tag": "tag19"
		},
		"20": null
	},
	"emptymap": null,
	"nilmap": null,
	"slice": [],
	"slicep": [],
	"pslice": [
		{
			"tag": "tag20"
		},
		{
			"tag": "tag21"
		}
	],
	"pslicep": [
		{
			"tag": "tag22"
		},
		null,
		{
			"tag": "tag23"
		}
	],
	"emptyslice": [],
	"nilslice": [],
	"stringslice": [],
	"byteslice": [],
	"small": {
		"tag": ""
	},
	"psmall": null,
	"ppsmall": {
		"tag": "tag31"
	},
	"interface": null,
	"pinterface": 5.2
}`

var pallValueCompact = strings.Map(noSpace, pallValueIndent)
