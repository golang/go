// run

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Checks that conversion of CMP(x,-y) -> CMN(x,y) is only applied in correct context.

package main

type decimal struct {
	d  [8]byte // digits, big-endian representation
	dp int     // decimal point
}

var powtab = []int{1, 3, 6, 9, 13, 16, 19, 23, 26}

//go:noinline
func foo(d *decimal) int {
	exp := int(d.d[1])
	if d.dp < 0 || d.dp == 0 && d.d[0] < '5' {
		var n int
		if -d.dp >= len(powtab) {
			n = 27
		} else {
			n = powtab[-d.dp] // incorrect CMP -> CMN substitution causes indexing panic.
		}
		exp += n
	}
	return exp
}

func main() {
	var d decimal
	d.d[0] = '1'
	if foo(&d) != 1 {
		println("FAILURE (though not the one this test was written to catch)")
	}
}
