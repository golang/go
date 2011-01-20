// $G $F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var nbad int

func assert(cond bool, msg string) {
	if !cond {
		if nbad == 0 {
			print("BUG")
		}
		nbad++
		print(" ", msg)
	}
}

func equal(a, b float32) bool {
	return a == b
}


func main() {
	// bool
	var t bool = true
	var f bool = false
	assert(t == !f, "bool")

	// int8
	var i00 int8 = 0
	var i01 int8 = 1
	var i02 int8 = -1
	var i03 int8 = 127
	var i04 int8 = -127
	var i05 int8 = -128
	var i06 int8 = +127
	assert(i01 == i00+1, "i01")
	assert(i02 == -i01, "i02")
	assert(i03 == -i04, "i03")
	assert(-(i05+1) == i06, "i05")

	// int16
	var i10 int16 = 0
	var i11 int16 = 1
	var i12 int16 = -1
	var i13 int16 = 32767
	var i14 int16 = -32767
	var i15 int16 = -32768
	var i16 int16 = +32767
	assert(i11 == i10+1, "i11")
	assert(i12 == -i11, "i12")
	assert(i13 == -i14, "i13")
	assert(-(i15+1) == i16, "i15")

	// int32
	var i20 int32 = 0
	var i21 int32 = 1
	var i22 int32 = -1
	var i23 int32 = 2147483647
	var i24 int32 = -2147483647
	var i25 int32 = -2147483648
	var i26 int32 = +2147483647
	assert(i21 == i20+1, "i21")
	assert(i22 == -i21, "i22")
	assert(i23 == -i24, "i23")
	assert(-(i25+1) == i26, "i25")
	assert(i23 == (1<<31)-1, "i23 size")

	// int64
	var i30 int64 = 0
	var i31 int64 = 1
	var i32 int64 = -1
	var i33 int64 = 9223372036854775807
	var i34 int64 = -9223372036854775807
	var i35 int64 = -9223372036854775808
	var i36 int64 = +9223372036854775807
	assert(i31 == i30+1, "i31")
	assert(i32 == -i31, "i32")
	assert(i33 == -i34, "i33")
	assert(-(i35+1) == i36, "i35")
	assert(i33 == (1<<63)-1, "i33 size")

	// uint8
	var u00 uint8 = 0
	var u01 uint8 = 1
	var u02 uint8 = 255
	var u03 uint8 = +255
	assert(u01 == u00+1, "u01")
	assert(u02 == u03, "u02")
	assert(u03 == (1<<8)-1, "u03 size")

	// uint16
	var u10 uint16 = 0
	var u11 uint16 = 1
	var u12 uint16 = 65535
	var u13 uint16 = +65535
	assert(u11 == u10+1, "u11")
	assert(u12 == u13, "u12")

	// uint32
	var u20 uint32 = 0
	var u21 uint32 = 1
	var u22 uint32 = 4294967295
	var u23 uint32 = +4294967295
	assert(u21 == u20+1, "u21")
	assert(u22 == u23, "u22")

	// uint64
	var u30 uint64 = 0
	var u31 uint64 = 1
	var u32 uint64 = 18446744073709551615
	var u33 uint64 = +18446744073709551615
	_, _, _, _ = u30, u31, u32, u33

	// float
	var f00 float32 = 3.14159
	var f01 float32 = -3.14159
	var f02 float32 = +3.14159
	var f03 float32 = 0.0
	var f04 float32 = .0
	var f05 float32 = 0.
	var f06 float32 = -0.0
	var f07 float32 = 1e10
	var f08 float32 = -1e10
	var f09 float32 = 1e-10
	var f10 float32 = 1e+10
	var f11 float32 = 1.e-10
	var f12 float32 = 1.e+10
	var f13 float32 = .1e-10
	var f14 float32 = .1e+10
	var f15 float32 = 1.1e-10
	var f16 float32 = 1.1e+10
	assert(f01 == -f00, "f01")
	assert(f02 == -f01, "f02")
	assert(f03 == f04, "f03")
	assert(f04 == f05, "f04")
	assert(f05 == f06, "f05")
	assert(f07 == -f08, "f07")
	assert(equal(f09, 1/f10), "f09")
	assert(f11 == f09, "f11")
	assert(f12 == f10, "f12")
	assert(equal(f13, f09/10.0), "f13")
	assert(equal(f14, f12/10.0), "f14")
	assert(equal(f15, f16/1e20), "f15")

	// character
	var c0 uint8 = 'a'
	var c1 uint8 = 'ä'
	var c2 uint8 = '\a'
	var c3 uint8 = '\b'
	var c4 uint8 = '\f'
	var c5 uint8 = '\n'
	var c6 uint8 = '\r'
	var c7 uint8 = '\t'
	var c8 uint8 = '\v'
	// var c9 uint8 = '本' // correctly caught as error
	var c9 uint16 = '本'
	assert(c0 == 0x61, "c0")
	assert(c1 == 0xe4, "c1")
	assert(c2 == 0x07, "c2")
	assert(c3 == 0x08, "c3")
	assert(c4 == 0x0c, "c4")
	assert(c5 == 0x0a, "c4")
	assert(c6 == 0x0d, "c6")
	assert(c7 == 0x09, "c7")
	assert(c8 == 0x0b, "c8")
	assert(c9 == 0x672c, "c9")

	var c00 uint8 = '\000'
	var c01 uint8 = '\007'
	var c02 uint8 = '\177'
	var c03 uint8 = '\377'
	assert(c00 == 0, "c00")
	assert(c01 == 7, "c01")
	assert(c02 == 127, "c02")
	assert(c03 == 255, "c03")

	var cx0 uint8 = '\x00'
	var cx1 uint8 = '\x0f'
	var cx2 uint8 = '\xff'
	assert(cx0 == 0, "cx0")
	assert(cx1 == 15, "cx1")
	assert(cx2 == 255, "cx2")

	var cu0 uint16 = '\u1234'
	var cu1 uint32 = '\U00101234'
	assert(cu0 == 0x1234, "cu0")
	assert(cu1 == 0x101234, "cu1")

	// string
	var s0 string = ""
	var s1 string = "hellô"
	assert(s1[0] == 'h', "s1-0")
	assert(s1[4] == 0xc3, "s1-4")
	assert(s1[5] == 0xb4, "s1-5")
	var s2 string = "\a\b\f\n\r\t\v"
	_, _ = s0, s2

	var s00 string = "\000"
	var s01 string = "\007"
	var s02 string = "\377"
	assert(s00[0] == 0, "s00")
	assert(s01[0] == 7, "s01")
	assert(s02[0] == 255, "s02")

	var x00 string = "\x00"
	var x01 string = "\x0f"
	var x02 string = "\xff"
	assert(x00[0] == 0, "x00")
	assert(x01[0] == 15, "x01")
	assert(x02[0] == 255, "x02")

	// these are all the same string
	var sj0 string = "日本語"
	var sj1 string = "\u65e5\u672c\u8a9e"
	var sj2 string = "\U000065e5\U0000672c\U00008a9e"
	var sj3 string = "\xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e"
	assert(sj0 == sj1, "sj1")
	assert(sj0 == sj2, "sj2")
	assert(sj0 == sj3, "sj3")

	if nbad > 0 {
		println()
	}
}
