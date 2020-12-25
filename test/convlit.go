// errorcheck

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that illegal assignments with both explicit and implicit conversions of literals are detected.
// Does not compile.

package main

import "unsafe"

// explicit conversion of constants
var x1 = string(1)
var x2 string = string(1)
var x3 = int(1.5)     // ERROR "convert|truncate"
var x4 int = int(1.5) // ERROR "convert|truncate"
var x5 = "a" + string(1)
var x6 = int(1e100)      // ERROR "overflow"
var x7 = float32(1e1000) // ERROR "overflow"

// unsafe.Pointer can only convert to/from uintptr
var _ = string(unsafe.Pointer(uintptr(65)))  // ERROR "convert|conversion"
var _ = float64(unsafe.Pointer(uintptr(65))) // ERROR "convert|conversion"
var _ = int(unsafe.Pointer(uintptr(65)))     // ERROR "convert|conversion"

// implicit conversions merit scrutiny
var s string
var bad1 string = 1  // ERROR "conver|incompatible|invalid|cannot"
var bad2 = s + 1     // ERROR "conver|incompatible|invalid|cannot"
var bad3 = s + 'a'   // ERROR "conver|incompatible|invalid|cannot"
var bad4 = "a" + 1   // ERROR "literals|incompatible|convert|invalid"
var bad5 = "a" + 'a' // ERROR "literals|incompatible|convert|invalid"

var bad6 int = 1.5       // ERROR "convert|truncate"
var bad7 int = 1e100     // ERROR "overflow"
var bad8 float32 = 1e200 // ERROR "overflow"

// but these implicit conversions are okay
var good1 string = "a"
var good2 int = 1.0
var good3 int = 1e9
var good4 float64 = 1e20

// explicit conversion of string is okay
var _ = []rune("abc")
var _ = []byte("abc")

// implicit is not
var _ []int = "abc"  // ERROR "cannot use|incompatible|invalid"
var _ []byte = "abc" // ERROR "cannot use|incompatible|invalid"

// named string is okay
type Tstring string

var ss Tstring = "abc"
var _ = []rune(ss)
var _ = []byte(ss)

// implicit is still not
var _ []rune = ss // ERROR "cannot use|incompatible|invalid"
var _ []byte = ss // ERROR "cannot use|incompatible|invalid"

// named slice is now ok
type Trune []rune
type Tbyte []byte

var _ = Trune("abc") // ok
var _ = Tbyte("abc") // ok

// implicit is still not
var _ Trune = "abc" // ERROR "cannot use|incompatible|invalid"
var _ Tbyte = "abc" // ERROR "cannot use|incompatible|invalid"
