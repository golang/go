// errorcheck -0 -d=escapedebug=1

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test the data word used for interface conversions
// that might otherwise allocate.

package dataword

var sink interface{}

func string1() {
	sink = "abc" // ERROR "using global for interface value"
}

func string2() {
	v := "abc"
	sink = v // ERROR "using global for interface value"
}

func string3() {
	sink = "" // ERROR "using global for interface value"
}

func string4() {
	v := ""
	sink = v // ERROR "using global for interface value"
}

func string5() {
	var a any = "abc" // ERROR "using global for interface value"
	_ = a
}

func string6() {
	var a any
	v := "abc"
	a = v // ERROR "using global for interface value"
	_ = a
}

// string7 can be inlined.
func string7(v string) {
	sink = v
}

func string8() {
	v0 := "abc"
	v := v0
	string7(v) // ERROR "using global for interface value"
}

func string9() {
	v0 := "abc"
	v := v0
	f := func() {
		string7(v)
	}
	f() // ERROR "using global for interface value"
}

func string10() {
	v0 := "abc"
	v := v0
	f := func() {
		f2 := func() {
			string7(v)
		}
		f2()
	}
	f() // ERROR "using global for interface value"
}

func string11() {
	v0 := "abc"
	v := v0
	defer func() {
		string7(v) // ERROR "using global for interface value"
	}()
}

func integer1() {
	sink = 42 // ERROR "using global for interface value"
}

func integer2() {
	v := 42
	sink = v // ERROR "using global for interface value"
}

func integer3() {
	sink = 0 // ERROR "using global for interface value"
}

func integer4a() {
	v := 0
	sink = v // ERROR "using global for interface value"
}

func integer4b() {
	v := uint8(0)
	sink = v // ERROR "using global for single-byte interface value"
}

func integer5() {
	var a any = 42 // ERROR "using global for interface value"
	_ = a
}

func integer6() {
	var a any
	v := 42
	a = v // ERROR "using global for interface value"
	_ = a
}

func integer7(v int) {
	sink = v
}

type M interface{ M() }

type MyInt int

func (m MyInt) M() {}

func escapes(m M) {
	sink = m
}

func named1a() {
	sink = MyInt(42) // ERROR "using global for interface value"
}

func named1b() {
	escapes(MyInt(42)) // ERROR "using global for interface value"
}

func named2a() {
	v := MyInt(0)
	sink = v // ERROR "using global for interface value"
}

func named2b() {
	v := MyInt(42)
	escapes(v) // ERROR "using global for interface value"
}

func named2c() {
	v := 42
	sink = MyInt(v) // ERROR "using global for interface value"
}

func named2d() {
	v := 42
	escapes(MyInt(v)) // ERROR "using global for interface value"
}
func named3a() {
	sink = MyInt(42) // ERROR "using global for interface value"
}

func named3b() {
	escapes(MyInt(0)) // ERROR "using global for interface value"
}

func named4a() {
	v := MyInt(0)
	sink = v // ERROR "using global for interface value"
}

func named4b() {
	v := MyInt(0)
	escapes(v) // ERROR "using global for interface value"
}

func named4c() {
	v := 0
	sink = MyInt(v) // ERROR "using global for interface value"
}

func named4d() {
	v := 0
	escapes(MyInt(v)) // ERROR "using global for interface value"
}

func named5() {
	var a any = MyInt(42) // ERROR "using global for interface value"
	_ = a
}

func named6() {
	var a any
	v := MyInt(42)
	a = v // ERROR "using global for interface value"
	_ = a
}

func named7a(v MyInt) {
	sink = v
}

func named7b(v MyInt) {
	escapes(v)
}

type S struct{ a, b int64 }

func struct1() {
	sink = S{1, 1}
}

func struct2() {
	v := S{1, 1}
	sink = v
}

func struct3() {
	sink = S{}
}

func struct4() {
	v := S{}
	sink = v
}

func struct5() {
	var a any = S{1, 1} // ERROR "using stack temporary for interface value"
	_ = a
}

func struct6() {
	var a any
	v := S{1, 1}
	a = v // ERROR "using stack temporary for interface value"
	_ = a
}

func struct7(v S) {
	sink = v
}

func emptyStruct1() {
	sink = struct{}{} // ERROR "using global for zero-sized interface value"
}

func emptyStruct2() {
	v := struct{}{}
	sink = v // ERROR "using global for zero-sized interface value"
}

func emptyStruct3(v struct{}) { // ERROR "using global for zero-sized interface value"
	sink = v
}

// Some light emulation of conditional debug printing (such as in #53465).

func Printf(format string, args ...any) {
	for _, arg := range args {
		sink = arg
	}
}

var enabled = true

func debugf(format string, args ...interface{}) {
	if enabled {
		Printf(format, args...)
	}
}

//go:noinline
func debugf2(format string, args ...interface{}) {
	if enabled {
		Printf(format, args...)
	}
}

func f1() {
	v := 1000
	debugf("hello %d", v) // ERROR "using global for interface value"
}

func f2() {
	v := 1000
	debugf2("hello %d", v) // ERROR "using global for interface value"
}

//go:noinline
func f3(i int) {
	debugf("hello %d", i)
}

func f4() {
	f3(1000)
}
