// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	"bytes"
	"reflect"
	"strconv"
	"testing"
)

type myStruct struct {
	T            bool
	F            bool
	S            string
	I8           int8
	I16          int16
	I32          int32
	I64          int64
	U8           uint8
	U16          uint16
	U32          uint32
	U64          uint64
	I            int
	U            uint
	Fl           float
	Fl32         float32
	Fl64         float64
	A            []string
	My           *myStruct
	Map          map[string][]int
	MapStruct    map[string]myStruct
	MapPtrStruct map[string]*myStruct
}

const encoded = `{"t":true,"f":false,"s":"abc","i8":1,"i16":2,"i32":3,"i64":4,` +
	` "u8":5,"u16":6,"u32":7,"u64":8,` +
	` "i":-9,"u":10,"bogusfield":"should be ignored",` +
	` "fl":11.5,"fl32":12.25,"fl64":13.75,` +
	` "a":["x","y","z"],"my":{"s":"subguy"},` +
	`"map":{"k1":[1,2,3],"k2":[],"k3":[3,4]},` +
	`"mapstruct":{"m1":{"u8":8}},` +
	`"mapptrstruct":{"m1":{"u8":8}}}`

var decodedMap = map[string][]int{
	"k1": []int{1, 2, 3},
	"k2": []int{},
	"k3": []int{3, 4},
}

var decodedMapStruct = map[string]myStruct{
	"m1": myStruct{U8: 8},
}

var decodedMapPtrStruct = map[string]*myStruct{
	"m1": &myStruct{U8: 8},
}

func check(t *testing.T, ok bool, name string, v interface{}) {
	if !ok {
		t.Errorf("%s = %v (BAD)", name, v)
	} else {
		t.Logf("%s = %v (good)", name, v)
	}
}

const whiteSpaceEncoded = " \t{\n\"s\"\r:\"string\"\v}"

func TestUnmarshalWhitespace(t *testing.T) {
	var m myStruct
	ok, errtok := Unmarshal(whiteSpaceEncoded, &m)
	if !ok {
		t.Fatalf("Unmarshal failed near %s", errtok)
	}
	check(t, m.S == "string", "string", m.S)
}

func TestUnmarshal(t *testing.T) {
	var m myStruct
	m.F = true
	ok, errtok := Unmarshal(encoded, &m)
	if !ok {
		t.Fatalf("Unmarshal failed near %s", errtok)
	}
	check(t, m.T == true, "t", m.T)
	check(t, m.F == false, "f", m.F)
	check(t, m.S == "abc", "s", m.S)
	check(t, m.I8 == 1, "i8", m.I8)
	check(t, m.I16 == 2, "i16", m.I16)
	check(t, m.I32 == 3, "i32", m.I32)
	check(t, m.I64 == 4, "i64", m.I64)
	check(t, m.U8 == 5, "u8", m.U8)
	check(t, m.U16 == 6, "u16", m.U16)
	check(t, m.U32 == 7, "u32", m.U32)
	check(t, m.U64 == 8, "u64", m.U64)
	check(t, m.I == -9, "i", m.I)
	check(t, m.U == 10, "u", m.U)
	check(t, m.Fl == 11.5, "fl", m.Fl)
	check(t, m.Fl32 == 12.25, "fl32", m.Fl32)
	check(t, m.Fl64 == 13.75, "fl64", m.Fl64)
	check(t, m.A != nil, "a", m.A)
	if m.A != nil {
		check(t, m.A[0] == "x", "a[0]", m.A[0])
		check(t, m.A[1] == "y", "a[1]", m.A[1])
		check(t, m.A[2] == "z", "a[2]", m.A[2])
	}
	check(t, m.My != nil, "my", m.My)
	if m.My != nil {
		check(t, m.My.S == "subguy", "my.s", m.My.S)
	}
	check(t, reflect.DeepEqual(m.Map, decodedMap), "map", m.Map)
	check(t, reflect.DeepEqual(m.MapStruct, decodedMapStruct), "mapstruct", m.MapStruct)
	check(t, reflect.DeepEqual(m.MapPtrStruct, decodedMapPtrStruct), "mapptrstruct", m.MapPtrStruct)
}

type Issue147Text struct {
	Text string
}

type Issue147 struct {
	Test []Issue147Text
}

const issue147Input = `{"test": [{"text":"0"},{"text":"1"},{"text":"2"},
{"text":"3"},{"text":"4"},{"text":"5"},
{"text":"6"},{"text":"7"},{"text":"8"},
{"text":"9"},{"text":"10"},{"text":"11"},
{"text":"12"},{"text":"13"},{"text":"14"},
{"text":"15"},{"text":"16"},{"text":"17"},
{"text":"18"},{"text":"19"},{"text":"20"},
{"text":"21"},{"text":"22"},{"text":"23"},
{"text":"24"},{"text":"25"},{"text":"26"},
{"text":"27"},{"text":"28"},{"text":"29"}]}`

func TestIssue147(t *testing.T) {
	var timeline Issue147
	Unmarshal(issue147Input, &timeline)

	if len(timeline.Test) != 30 {
		t.Errorf("wrong length: got %d want 30", len(timeline.Test))
	}

	for i, e := range timeline.Test {
		if e.Text != strconv.Itoa(i) {
			t.Errorf("index: %d got: %s want: %d", i, e.Text, i)
		}
	}
}

type Issue114 struct {
	Text string
}

const issue114Input = `[{"text" : "0"}, {"text" : "1"}, {"text" : "2"}, {"text" : "3"}]`

func TestIssue114(t *testing.T) {
	var items []Issue114
	Unmarshal(issue114Input, &items)

	if len(items) != 4 {
		t.Errorf("wrong length: got %d want 4", len(items))
	}

	for i, e := range items {
		if e.Text != strconv.Itoa(i) {
			t.Errorf("index: %d got: %s want: %d", i, e.Text, i)
		}
	}
}

type marshalTest struct {
	val interface{}
	out string
}

var marshalTests = []marshalTest{
	// basic string
	marshalTest{nil, "null"},
	marshalTest{true, "true"},
	marshalTest{false, "false"},
	marshalTest{123, "123"},
	marshalTest{0.1, "0.1"},
	marshalTest{1e-10, "1e-10"},
	marshalTest{"teststring", `"teststring"`},
	marshalTest{[4]int{1, 2, 3, 4}, "[1,2,3,4]"},
	marshalTest{[]int{1, 2, 3, 4}, "[1,2,3,4]"},
	marshalTest{[]interface{}{nil}, "[null]"},
	marshalTest{[][]int{[]int{1, 2}, []int{3, 4}}, "[[1,2],[3,4]]"},
	marshalTest{map[string]string{"one": "one"}, `{"one":"one"}`},
	marshalTest{map[string]int{"one": 1}, `{"one":1}`},
	marshalTest{map[string]interface{}{"null": nil}, `{"null":null}`},
	marshalTest{struct{}{}, "{}"},
	marshalTest{struct{ a int }{1}, `{"a":1}`},
	marshalTest{struct{ a interface{} }{nil}, `{"a":null}`},
	marshalTest{struct {
		a int
		b string
	}{1, "hello"},
		`{"a":1,"b":"hello"}`,
	},
	marshalTest{map[string][]int{"3": []int{1, 2, 3}}, `{"3":[1,2,3]}`},
}

func TestMarshal(t *testing.T) {
	for _, tt := range marshalTests {
		var buf bytes.Buffer

		err := Marshal(&buf, tt.val)
		if err != nil {
			t.Fatalf("Marshal(%T): %s", tt.val, err)
		}

		s := buf.String()
		if s != tt.out {
			t.Errorf("Marshal(%T) = %q, want %q\n", tt.val, tt.out, s)
		}
	}
}

type marshalErrorTest struct {
	val   interface{}
	error string
}

type MTE string

var marshalErrorTests = []marshalErrorTest{
	marshalErrorTest{map[chan int]string{make(chan int): "one"}, "json cannot encode value of type map[chan int] string"},
	marshalErrorTest{map[string]*MTE{"hi": nil}, "json cannot encode value of type *json.MTE"},
}

func TestMarshalError(t *testing.T) {
	for _, tt := range marshalErrorTests {
		var buf bytes.Buffer

		err := Marshal(&buf, tt.val)

		if err == nil {
			t.Fatalf("Marshal(%T): no error, want error %s", tt.val, tt.error)
		}

		if err.String() != tt.error {
			t.Fatalf("Marshal(%T) = error %s, want error %s", tt.val, err, tt.error)
		}

	}
}
