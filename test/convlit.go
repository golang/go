// errchk $G -e $D/$F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// explicit conversion of constants is work in progress.
// the ERRORs in this block are debatable, but they're what
// the language spec says for now.
var x1 = string(1)
var x2 string = string(1)
var x3 = int(1.5)     // ERROR "convert|truncate"
var x4 int = int(1.5) // ERROR "convert|truncate"
var x5 = "a" + string(1)
var x6 = int(1e100)      // ERROR "overflow"
var x7 = float32(1e1000) // ERROR "overflow"

// implicit conversions merit scrutiny
var s string
var bad1 string = 1  // ERROR "conver|incompatible|invalid|cannot"
var bad2 = s + 1     // ERROR "conver|incompatible|invalid"
var bad3 = s + 'a'   // ERROR "conver|incompatible|invalid"
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
var _ = []int("abc")
var _ = []byte("abc")

// implicit is not
var _ []int = "abc"  // ERROR "cannot use|incompatible|invalid"
var _ []byte = "abc" // ERROR "cannot use|incompatible|invalid"

// named string is okay
type Tstring string

var ss Tstring = "abc"
var _ = []int(ss)
var _ = []byte(ss)

// implicit is still not
var _ []int = ss  // ERROR "cannot use|incompatible|invalid"
var _ []byte = ss // ERROR "cannot use|incompatible|invalid"

// named slice is not
type Tint []int
type Tbyte []byte

var _ = Tint("abc")  // ERROR "convert|incompatible|invalid"
var _ = Tbyte("abc") // ERROR "convert|incompatible|invalid"

// implicit is still not
var _ Tint = "abc"  // ERROR "cannot use|incompatible|invalid"
var _ Tbyte = "abc" // ERROR "cannot use|incompatible|invalid"
