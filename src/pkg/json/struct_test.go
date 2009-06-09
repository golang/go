// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	"json";
	"testing";
)

type _MyStruct struct {
	t bool;
	f bool;
	s string;
	i8 int8;
	i16 int16;
	i32 int32;
	i64 int64;
	u8 uint8;
	u16 uint16;
	u32 uint32;
	u64 uint64;
	i int;
	u uint;
	fl float;
	fl32 float32;
	fl64 float64;
	a *[]string;	// TODO(rsc): Should be able to use []string.
	my *_MyStruct;
};

const _Encoded =
	`{"t":true,"f":false,"s":"abc","i8":1,"i16":2,"i32":3,"i64":4,`
	` "u8":5,"u16":6,"u32":7,"u64":8,`
	` "i":-9,"u":10,"bogusfield":"should be ignored",`
	` "fl":11.5,"fl32":12.25,"fl64":13.75,`
	` "a":["x","y","z"],"my":{"s":"subguy"}}`;


func _Check(t *testing.T, ok bool, name string, v interface{}) {
	if !ok {
		t.Errorf("%s = %v (BAD)", name, v);
	} else {
		t.Logf("%s = %v (good)", name, v);
	}
}

func TestUnmarshal(t *testing.T) {
	var m _MyStruct;
	m.f = true;
	ok, errtok := Unmarshal(_Encoded, &m);
	if !ok {
		t.Fatalf("Unmarshal failed near %s", errtok);
	}
	_Check(t, m.t==true, "t", m.t);
	_Check(t, m.f==false, "f", m.f);
	_Check(t, m.s=="abc", "s", m.s);
	_Check(t, m.i8==1, "i8", m.i8);
	_Check(t, m.i16==2, "i16", m.i16);
	_Check(t, m.i32==3, "i32", m.i32);
	_Check(t, m.i64==4, "i64", m.i64);
	_Check(t, m.u8==5, "u8", m.u8);
	_Check(t, m.u16==6, "u16", m.u16);
	_Check(t, m.u32==7, "u32", m.u32);
	_Check(t, m.u64==8, "u64", m.u64);
	_Check(t, m.i==-9, "i", m.i);
	_Check(t, m.u==10, "u", m.u);
	_Check(t, m.fl==11.5, "fl", m.fl);
	_Check(t, m.fl32==12.25, "fl32", m.fl32);
	_Check(t, m.fl64==13.75, "fl64", m.fl64);
	_Check(t, m.a!=nil, "a", m.a);
	if m.a != nil {
		_Check(t, m.a[0]=="x", "a[0]", m.a[0]);
		_Check(t, m.a[1]=="y", "a[1]", m.a[1]);
		_Check(t, m.a[2]=="z", "a[2]", m.a[2]);
	}
	_Check(t, m.my!=nil, "my", m.my);
	if m.my != nil {
		_Check(t, m.my.s=="subguy", "my.s", m.my.s);
	}
}
