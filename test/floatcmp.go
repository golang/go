// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test floating-point comparison involving NaN.

package main

import "math"

type floatTest struct {
	name string
	expr bool
	want bool
}

var nan float64 = math.NaN()
var f float64 = 1

var tests = []floatTest{
	floatTest{"nan == nan", nan == nan, false},
	floatTest{"nan != nan", nan != nan, true},
	floatTest{"nan < nan", nan < nan, false},
	floatTest{"nan > nan", nan > nan, false},
	floatTest{"nan <= nan", nan <= nan, false},
	floatTest{"nan >= nan", nan >= nan, false},
	floatTest{"f == nan", f == nan, false},
	floatTest{"f != nan", f != nan, true},
	floatTest{"f < nan", f < nan, false},
	floatTest{"f > nan", f > nan, false},
	floatTest{"f <= nan", f <= nan, false},
	floatTest{"f >= nan", f >= nan, false},
	floatTest{"nan == f", nan == f, false},
	floatTest{"nan != f", nan != f, true},
	floatTest{"nan < f", nan < f, false},
	floatTest{"nan > f", nan > f, false},
	floatTest{"nan <= f", nan <= f, false},
	floatTest{"nan >= f", nan >= f, false},
	floatTest{"!(nan == nan)", !(nan == nan), true},
	floatTest{"!(nan != nan)", !(nan != nan), false},
	floatTest{"!(nan < nan)", !(nan < nan), true},
	floatTest{"!(nan > nan)", !(nan > nan), true},
	floatTest{"!(nan <= nan)", !(nan <= nan), true},
	floatTest{"!(nan >= nan)", !(nan >= nan), true},
	floatTest{"!(f == nan)", !(f == nan), true},
	floatTest{"!(f != nan)", !(f != nan), false},
	floatTest{"!(f < nan)", !(f < nan), true},
	floatTest{"!(f > nan)", !(f > nan), true},
	floatTest{"!(f <= nan)", !(f <= nan), true},
	floatTest{"!(f >= nan)", !(f >= nan), true},
	floatTest{"!(nan == f)", !(nan == f), true},
	floatTest{"!(nan != f)", !(nan != f), false},
	floatTest{"!(nan < f)", !(nan < f), true},
	floatTest{"!(nan > f)", !(nan > f), true},
	floatTest{"!(nan <= f)", !(nan <= f), true},
	floatTest{"!(nan >= f)", !(nan >= f), true},
	floatTest{"!!(nan == nan)", !!(nan == nan), false},
	floatTest{"!!(nan != nan)", !!(nan != nan), true},
	floatTest{"!!(nan < nan)", !!(nan < nan), false},
	floatTest{"!!(nan > nan)", !!(nan > nan), false},
	floatTest{"!!(nan <= nan)", !!(nan <= nan), false},
	floatTest{"!!(nan >= nan)", !!(nan >= nan), false},
	floatTest{"!!(f == nan)", !!(f == nan), false},
	floatTest{"!!(f != nan)", !!(f != nan), true},
	floatTest{"!!(f < nan)", !!(f < nan), false},
	floatTest{"!!(f > nan)", !!(f > nan), false},
	floatTest{"!!(f <= nan)", !!(f <= nan), false},
	floatTest{"!!(f >= nan)", !!(f >= nan), false},
	floatTest{"!!(nan == f)", !!(nan == f), false},
	floatTest{"!!(nan != f)", !!(nan != f), true},
	floatTest{"!!(nan < f)", !!(nan < f), false},
	floatTest{"!!(nan > f)", !!(nan > f), false},
	floatTest{"!!(nan <= f)", !!(nan <= f), false},
	floatTest{"!!(nan >= f)", !!(nan >= f), false},
}

func main() {
	bad := false
	for _, t := range tests {
		if t.expr != t.want {
			if !bad {
				bad = true
				println("BUG: floatcmp")
			}
			println(t.name, "=", t.expr, "want", t.want)
		}
	}
	if bad {
		panic("floatcmp failed")
	}
}
