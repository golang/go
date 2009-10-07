// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	"testing";
)

type _MyStruct struct {
	T	bool;
	F	bool;
	S	string;
	I8	int8;
	I16	int16;
	I32	int32;
	I64	int64;
	U8	uint8;
	U16	uint16;
	U32	uint32;
	U64	uint64;
	I	int;
	U	uint;
	Fl	float;
	Fl32	float32;
	Fl64	float64;
	A	[]string;
	My	*_MyStruct;
}

const _Encoded = `{"t":true,"f":false,"s":"abc","i8":1,"i16":2,"i32":3,"i64":4,`
	` "u8":5,"u16":6,"u32":7,"u64":8,`
	` "i":-9,"u":10,"bogusfield":"should be ignored",`
	` "fl":11.5,"fl32":12.25,"fl64":13.75,`
	` "a":["x","y","z"],"my":{"s":"subguy"}}`


func _Check(t *testing.T, ok bool, name string, v interface{}) {
	if !ok {
		t.Errorf("%s = %v (BAD)", name, v);
	} else {
		t.Logf("%s = %v (good)", name, v);
	}
}

func TestUnmarshal(t *testing.T) {
	var m _MyStruct;
	m.F = true;
	ok, errtok := Unmarshal(_Encoded, &m);
	if !ok {
		t.Fatalf("Unmarshal failed near %s", errtok);
	}
	_Check(t, m.T == true, "t", m.T);
	_Check(t, m.F == false, "f", m.F);
	_Check(t, m.S == "abc", "s", m.S);
	_Check(t, m.I8 == 1, "i8", m.I8);
	_Check(t, m.I16 == 2, "i16", m.I16);
	_Check(t, m.I32 == 3, "i32", m.I32);
	_Check(t, m.I64 == 4, "i64", m.I64);
	_Check(t, m.U8 == 5, "u8", m.U8);
	_Check(t, m.U16 == 6, "u16", m.U16);
	_Check(t, m.U32 == 7, "u32", m.U32);
	_Check(t, m.U64 == 8, "u64", m.U64);
	_Check(t, m.I == -9, "i", m.I);
	_Check(t, m.U == 10, "u", m.U);
	_Check(t, m.Fl == 11.5, "fl", m.Fl);
	_Check(t, m.Fl32 == 12.25, "fl32", m.Fl32);
	_Check(t, m.Fl64 == 13.75, "fl64", m.Fl64);
	_Check(t, m.A != nil, "a", m.A);
	if m.A != nil {
		_Check(t, m.A[0] == "x", "a[0]", m.A[0]);
		_Check(t, m.A[1] == "y", "a[1]", m.A[1]);
		_Check(t, m.A[2] == "z", "a[2]", m.A[2]);
	}
	_Check(t, m.My != nil, "my", m.My);
	if m.My != nil {
		_Check(t, m.My.S == "subguy", "my.s", m.My.S);
	}
}
