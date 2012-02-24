// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test interfaces on basic types.

package main

type myint int
type mystring string
type I0 interface{}

func f() {
	var ia, ib I0
	var i myint
	var s mystring

	if ia != ib {
		panic("1")
	}

	i = 1
	ia = i
	ib = i
	if ia != ib {
		panic("2")
	}
	if ia == nil {
		panic("3")
	}

	i = 2
	ia = i
	if ia == ib {
		panic("4")
	}

	ia = nil
	if ia == ib {
		panic("5")
	}

	ib = nil
	if ia != ib {
		panic("6")
	}

	if ia != nil {
		panic("7")
	}

	s = "abc"
	ia = s
	ib = nil
	if ia == ib {
		panic("8")
	}

	s = "def"
	ib = s
	if ia == ib {
		panic("9")
	}

	s = "abc"
	ib = s
	if ia != ib {
		panic("a")
	}
}

func main() {
	var ia [20]I0
	var b bool
	var s string
	var i8 int8
	var i16 int16
	var i32 int32
	var i64 int64
	var u8 uint8
	var u16 uint16
	var u32 uint32
	var u64 uint64

	f()

	ia[0] = "xxx"
	ia[1] = 12345
	ia[2] = true

	s = "now is"
	ia[3] = s
	b = false
	ia[4] = b

	i8 = 29
	ia[5] = i8
	i16 = 994
	ia[6] = i16
	i32 = 3434
	ia[7] = i32
	i64 = 1234567
	ia[8] = i64

	u8 = 12
	ia[9] = u8
	u16 = 799
	ia[10] = u16
	u32 = 4455
	ia[11] = u32
	u64 = 765432
	ia[12] = u64

	s = ia[0].(string)
	if s != "xxx" {
		println(0, s)
		panic("fail")
	}
	i32 = int32(ia[1].(int))
	if i32 != 12345 {
		println(1, i32)
		panic("fail")
	}
	b = ia[2].(bool)
	if b != true {
		println(2, b)
		panic("fail")
	}

	s = ia[3].(string)
	if s != "now is" {
		println(3, s)
		panic("fail")
	}
	b = ia[4].(bool)
	if b != false {
		println(4, b)
		panic("fail")
	}

	i8 = ia[5].(int8)
	if i8 != 29 {
		println(5, i8)
		panic("fail")
	}
	i16 = ia[6].(int16)
	if i16 != 994 {
		println(6, i16)
		panic("fail")
	}
	i32 = ia[7].(int32)
	if i32 != 3434 {
		println(7, i32)
		panic("fail")
	}
	i64 = ia[8].(int64)
	if i64 != 1234567 {
		println(8, i64)
		panic("fail")
	}

	u8 = ia[9].(uint8)
	if u8 != 12 {
		println(5, u8)
		panic("fail")
	}
	u16 = ia[10].(uint16)
	if u16 != 799 {
		println(6, u16)
		panic("fail")
	}
	u32 = ia[11].(uint32)
	if u32 != 4455 {
		println(7, u32)
		panic("fail")
	}
	u64 = ia[12].(uint64)
	if u64 != 765432 {
		println(8, u64)
		panic("fail")
	}
}
