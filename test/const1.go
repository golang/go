// errchk $G -e $F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type I interface {}
const (
	// assume all types behave similarly to int8/uint8
	Int8 int8 = 101;
	Minus1 int8 = -1;
	Uint8 uint8 = 102;
	Const = 103;

	Float32 float32 = 104.5;
	Float float = 105.5;
	ConstFloat = 106.5;
	Big float64 = 1e300;

	String = "abc";
	Bool = true;
)

var (
	a1 = Int8 * 100;	// ERROR "overflows"
	a2 = Int8 * -1;	// OK
	a3 = Int8 * 1000;	// ERROR "overflows"
	a4 = Int8 * int8(1000);	// ERROR "overflows"
	a5 = int8(Int8 * 1000);	// ERROR "overflows"
	a6 = int8(Int8 * int8(1000));	// ERROR "overflows"
	a7 = Int8 - 2*Int8 - 2*Int8;	// ERROR "overflows"
	a8 = Int8 * Const / 100;	// ERROR "overflows"
	a9 = Int8 * (Const / 100);	// OK

	b1 = Uint8 * Uint8;	// ERROR "overflows"
	b2 = Uint8 * -1;	// ERROR "overflows"
	b3 = Uint8 - Uint8;	// OK
	b4 = Uint8 - Uint8 - Uint8;	// ERROR "overflows"
	b5 = uint8(^0);	// ERROR "overflows"
	b6 = ^uint8(0);	// ERROR "overflows"
	b7 = uint8(Minus1);	// ERROR "overflows"
	b8 = uint8(int8(-1));	// ERROR "overflows"
	b8a = uint8(-1);	// ERROR "overflows"
	b9 byte = (1<<10) >> 8;	// OK
	b10 byte = (1<<10);	// ERROR "overflows"
	b11 byte = (byte(1)<<10) >> 8;	// ERROR "overflows"
	b12 byte = 1000;	// ERROR "overflows"
	b13 byte = byte(1000);	// ERROR "overflows"
	b14 byte = byte(100) * byte(100);	// ERROR "overflows"
	b15 byte = byte(100) * 100;	// ERROR "overflows"
	b16 byte = byte(0) * 1000;	// ERROR "overflows"
	b16a byte = 0 * 1000;	// OK
	b17 byte = byte(0) * byte(1000);	// ERROR "overflows"
	b18 byte = Uint8/0;	// ERROR "division by zero"

	c1 float64 = Big;
	c2 float64 = Big*Big;	// ERROR "overflows"
	c3 float64 = float64(Big)*Big;	// ERROR "overflows"
	c4 = Big*Big;	// ERROR "overflows"
	c5 = Big/0;	// ERROR "division by zero"
)

func f(int);

func main() {
	f(Int8);	// ERROR "convert"
	f(Minus1);	// ERROR "convert"
	f(Uint8);	// ERROR "convert"
	f(Const);	// OK
	f(Float32);	// ERROR "convert"
	f(Float);	// ERROR "convert"
	f(ConstFloat);	// ERROR "truncate"
	f(ConstFloat - 0.5);	// OK
	f(Big);	// ERROR "convert"
	f(String);	// ERROR "convert"
	f(Bool);	// ERROR "convert"
}
