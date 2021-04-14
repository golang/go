// errorcheck

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify overflow is detected when using numeric constants.
// Does not compile.

package main

import "unsafe"

type I interface{}

const (
	// assume all types behave similarly to int8/uint8
	Int8   int8  = 101
	Minus1 int8  = -1
	Uint8  uint8 = 102
	Const        = 103

	Float32    float32 = 104.5
	Float64    float64 = 105.5
	ConstFloat         = 106.5
	Big        float64 = 1e300

	String = "abc"
	Bool   = true
)

var (
	a1 = Int8 * 100              // ERROR "overflow"
	a2 = Int8 * -1               // OK
	a3 = Int8 * 1000             // ERROR "overflow"
	a4 = Int8 * int8(1000)       // ERROR "overflow"
	a5 = int8(Int8 * 1000)       // ERROR "overflow"
	a6 = int8(Int8 * int8(1000)) // ERROR "overflow"
	a7 = Int8 - 2*Int8 - 2*Int8  // ERROR "overflow"
	a8 = Int8 * Const / 100      // ERROR "overflow"
	a9 = Int8 * (Const / 100)    // OK

	b1        = Uint8 * Uint8         // ERROR "overflow"
	b2        = Uint8 * -1            // ERROR "overflow"
	b3        = Uint8 - Uint8         // OK
	b4        = Uint8 - Uint8 - Uint8 // ERROR "overflow"
	b5        = uint8(^0)             // ERROR "overflow"
	b5a       = int64(^0)             // OK
	b6        = ^uint8(0)             // OK
	b6a       = ^int64(0)             // OK
	b7        = uint8(Minus1)         // ERROR "overflow"
	b8        = uint8(int8(-1))       // ERROR "overflow"
	b8a       = uint8(-1)             // ERROR "overflow"
	b9   byte = (1 << 10) >> 8        // OK
	b10  byte = (1 << 10)             // ERROR "overflow"
	b11  byte = (byte(1) << 10) >> 8  // ERROR "overflow"
	b12  byte = 1000                  // ERROR "overflow"
	b13  byte = byte(1000)            // ERROR "overflow"
	b14  byte = byte(100) * byte(100) // ERROR "overflow"
	b15  byte = byte(100) * 100       // ERROR "overflow"
	b16  byte = byte(0) * 1000        // ERROR "overflow"
	b16a byte = 0 * 1000              // OK
	b17  byte = byte(0) * byte(1000)  // ERROR "overflow"
	b18  byte = Uint8 / 0             // ERROR "division by zero"

	c1 float64 = Big
	c2 float64 = Big * Big          // ERROR "overflow"
	c3 float64 = float64(Big) * Big // ERROR "overflow"
	c4         = Big * Big          // ERROR "overflow"
	c5         = Big / 0            // ERROR "division by zero"
	c6         = 1000 % 1e3         // ERROR "invalid operation|expected integer type"
)

func f(int)

func main() {
	f(Int8)             // ERROR "convert|wrong type|cannot"
	f(Minus1)           // ERROR "convert|wrong type|cannot"
	f(Uint8)            // ERROR "convert|wrong type|cannot"
	f(Const)            // OK
	f(Float32)          // ERROR "convert|wrong type|cannot"
	f(Float64)          // ERROR "convert|wrong type|cannot"
	f(ConstFloat)       // ERROR "truncate"
	f(ConstFloat - 0.5) // OK
	f(Big)              // ERROR "convert|wrong type|cannot"
	f(String)           // ERROR "convert|wrong type|cannot|incompatible"
	f(Bool)             // ERROR "convert|wrong type|cannot|incompatible"
}

const ptr = nil // ERROR "const.*nil"
const _ = string([]byte(nil)) // ERROR "is not a? ?constant"
const _ = uintptr(unsafe.Pointer((*int)(nil))) // ERROR "is not a? ?constant"
const _ = unsafe.Pointer((*int)(nil)) // ERROR "cannot be nil|invalid constant type|is not a constant"
const _ = (*int)(nil) // ERROR "cannot be nil|invalid constant type|is not a constant"
