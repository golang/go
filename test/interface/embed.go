// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test methods derived from embedded interface values.

package main

import "os"

const Value = 1e12

type Inter interface { M() int64 }

type T int64
func (t T) M() int64 { return int64(t) }
var t = T(Value)
var pt = &t
var ti Inter = t

type S struct { Inter }
var s = S{ ti }
var ps = &s

var i Inter

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
	check("s.M()", s.M())
	check("ps.M()", ps.M())

	i = t
	check("i = t; i.M()", i.M())

	i = pt
	check("i = pt; i.M()", i.M())

	i = s
	check("i = s; i.M()", i.M())

	i = ps
	check("i = ps; i.M()", i.M())

	if !ok {
		println("BUG: interface10")
		os.Exit(1)
	}
}
