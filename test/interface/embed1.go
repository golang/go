// $G $D/embed0.go && $G $D/$F.go && $L $F.$A && ./$A.out

// NOTE: This test is not run by 'run.go' and so not run by all.bash.
// To run this test you must use the ./run shell script.

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that embedded interface types can have local methods.

package main

import "./embed0"

type T int
func (t T) m() {}

type I interface { m() }
type J interface { I }

type PI interface { p.I }
type PJ interface { p.J }

func main() {
	var i I
	var j J
	var t T
	i = t
	j = t
	_ = i
	_ = j
	i = j
	_ = i
	j = i
	_ = j
	var pi PI
	var pj PJ
	var pt p.T
	pi = pt
	pj = pt
	_ = pi
	_ = pj
	pi = pj
	_ = pi
	pj = pi
	_ = pj
}
