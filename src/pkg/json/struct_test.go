// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	"reflect";
	"strconv";
	"testing";
)

type myStruct struct {
	T		bool;
	F		bool;
	S		string;
	I8		int8;
	I16		int16;
	I32		int32;
	I64		int64;
	U8		uint8;
	U16		uint16;
	U32		uint32;
	U64		uint64;
	I		int;
	U		uint;
	Fl		float;
	Fl32		float32;
	Fl64		float64;
	A		[]string;
	My		*myStruct;
	Map		map[string][]int;
	MapStruct	map[string]myStruct;
	MapPtrStruct	map[string]*myStruct;
}

const encoded = `{"t":true,"f":false,"s":"abc","i8":1,"i16":2,"i32":3,"i64":4,`
	` "u8":5,"u16":6,"u32":7,"u64":8,`
	` "i":-9,"u":10,"bogusfield":"should be ignored",`
	` "fl":11.5,"fl32":12.25,"fl64":13.75,`
	` "a":["x","y","z"],"my":{"s":"subguy"},`
	`"map":{"k1":[1,2,3],"k2":[],"k3":[3,4]},`
	`"mapstruct":{"m1":{"u8":8}},`
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

func TestUnmarshal(t *testing.T) {
	var m myStruct;
	m.F = true;
	ok, errtok := Unmarshal(encoded, &m);
	if !ok {
		t.Fatalf("Unmarshal failed near %s", errtok)
	}
	check(t, m.T == true, "t", m.T);
	check(t, m.F == false, "f", m.F);
	check(t, m.S == "abc", "s", m.S);
	check(t, m.I8 == 1, "i8", m.I8);
	check(t, m.I16 == 2, "i16", m.I16);
	check(t, m.I32 == 3, "i32", m.I32);
	check(t, m.I64 == 4, "i64", m.I64);
	check(t, m.U8 == 5, "u8", m.U8);
	check(t, m.U16 == 6, "u16", m.U16);
	check(t, m.U32 == 7, "u32", m.U32);
	check(t, m.U64 == 8, "u64", m.U64);
	check(t, m.I == -9, "i", m.I);
	check(t, m.U == 10, "u", m.U);
	check(t, m.Fl == 11.5, "fl", m.Fl);
	check(t, m.Fl32 == 12.25, "fl32", m.Fl32);
	check(t, m.Fl64 == 13.75, "fl64", m.Fl64);
	check(t, m.A != nil, "a", m.A);
	if m.A != nil {
		check(t, m.A[0] == "x", "a[0]", m.A[0]);
		check(t, m.A[1] == "y", "a[1]", m.A[1]);
		check(t, m.A[2] == "z", "a[2]", m.A[2]);
	}
	check(t, m.My != nil, "my", m.My);
	if m.My != nil {
		check(t, m.My.S == "subguy", "my.s", m.My.S)
	}
	check(t, reflect.DeepEqual(m.Map, decodedMap), "map", m.Map);
	check(t, reflect.DeepEqual(m.MapStruct, decodedMapStruct), "mapstruct", m.MapStruct);
	check(t, reflect.DeepEqual(m.MapPtrStruct, decodedMapPtrStruct), "mapptrstruct", m.MapPtrStruct);
}

type Issue147Text struct {
	Text string;
}

type Issue147 struct {
	Test []Issue147Text;
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
	var timeline Issue147;
	Unmarshal(issue147Input, &timeline);

	for i, e := range timeline.Test {
		if e.Text != strconv.Itoa(i) {
			t.Errorf("index: %d got: %s want: %d", i, e.Text, i)
		}
	}
}
