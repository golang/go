// errchk $G -e $D/$F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check methods derived from embedded interface and *interface values.

package main

import "os"

const Value = 1e12

type Inter interface { M() int64 }

type T int64
func (t T) M() int64 { return int64(t) }
var t = T(Value)
var pt = &t
var ti Inter = t
var pti = &ti

type S struct { Inter }
var s = S{ ti }
var ps = &s

type SP struct { *Inter }	// ERROR "interface"

var i Inter
var pi = &i

var ok = true

func check(s string, v int64) {
	if v != Value {
		println(s, v)
		ok = false
	}
}

func main() {
	check("t.M()", t.M())
	check("pt.M()", pt.M())
	check("ti.M()", ti.M())
	check("pti.M()", pti.M())	// ERROR "method"
	check("s.M()", s.M())
	check("ps.M()", ps.M())

	i = t
	check("i = t; i.M()", i.M())
	check("i = t; pi.M()", pi.M())	// ERROR "method"

	i = pt
	check("i = pt; i.M()", i.M())
	check("i = pt; pi.M()", pi.M())	// ERROR "method"

	i = s
	check("i = s; i.M()", i.M())
	check("i = s; pi.M()", pi.M())	// ERROR "method"

	i = ps
	check("i = ps; i.M()", i.M())
	check("i = ps; pi.M()", pi.M())	// ERROR "method"

	if !ok {
		println("BUG: interface10")
		os.Exit(1)
	}
}
