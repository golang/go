// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type	myint		int;
type	mystring	string;
type	I0		interface {};

func
f()
{
	var ia, ib I0;
	var i myint;
	var s mystring;

	if ia != ib { panicln("1"); }

	i = 1;
	ia = i;
	ib = i;
	if ia != ib { panicln("2"); }
	if ia == nil { panicln("3"); }

	i = 2;
	ia = i;
	if ia == ib { panicln("4"); }

	ia = nil;
	if ia == ib { panicln("5"); }

	ib = nil;
	if ia != ib { panicln("6"); }

	if ia != nil { panicln("7"); }

	s = "abc";
	ia = s;
	ib = nil;
	if ia == ib { panicln("8"); }

	s = "def";
	ib = s;
	if ia == ib { panicln("9"); }

	s = "abc";
	ib = s;
	if ia != ib { panicln("a"); }
}

func
main()
{
	var ia [20]I0;
	var b bool;
	var s string;
	var i8 int8;
	var i16 int16;
	var i32 int32;
	var i64 int64;
	var u8 uint8;
	var u16 uint16;
	var u32 uint32;
	var u64 uint64;

	f();

	ia[0] = "xxx";
	ia[1] = 12345;
	ia[2] = true;

	s = "now is";	ia[3] = s;
	b = false;	ia[4] = b;

	i8 = 29;	ia[5] = i8;
	i16 = 994;	ia[6] = i16;
	i32 = 3434;	ia[7] = i32;
	i64 = 1234567;	ia[8] = i64;

	u8 = 12;	ia[9] = u8;
	u16 = 799;	ia[10] = u16;
	u32 = 4455;	ia[11] = u32;
	u64 = 765432;	ia[12] = u64;

	s = ia[0];	if s != "xxx" { panicln(0,s); }
	i32 = int32(ia[1].(int));
			if i32 != 12345 { panicln(1,i32); }
	b = ia[2];	if b != true { panicln(2,b); }

	s = ia[3];	if s != "now is" { panicln(3,s); }
	b = ia[4];	if b != false { panicln(4,b); }

	i8 = ia[5];	if i8 != 29 { panicln(5,i8); }
	i16 = ia[6];	if i16 != 994 { panicln(6,i16); }
	i32 = ia[7];	if i32 != 3434 { panicln(7,i32); }
	i64 = ia[8];	if i64 != 1234567 { panicln(8,i64); }

	u8 = ia[9];	if u8 != 12 { panicln(5,u8); }
	u16 = ia[10];	if u16 != 799 { panicln(6,u16); }
	u32 = ia[11];	if u32 != 4455 { panicln(7,u32); }
	u64 = ia[12];	if u64 != 765432 { panicln(8,u64); }
}
