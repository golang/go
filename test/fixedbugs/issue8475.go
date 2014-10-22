// build

// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 8745: comma-ok assignments should produce untyped bool as 2nd result.

package main

type mybool bool

func main() {
	var ok mybool
	_ = ok

	var i interface{}
	_, ok = i.(int)

	var m map[int]int
	_, ok = m[0]

	var c chan int
	_, ok = <-c
}
