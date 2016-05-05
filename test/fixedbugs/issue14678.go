// run

// Copyright 2016 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	m := make(map[int]bool)
	i := interface{}(1)
	var v int

	// Ensure map is updated properly
	_, m[1] = i.(int)
	v, m[2] = i.(int)

	if v != 1 {
		panic("fail: v should be 1")
	}
	if m[1] == false {
		panic("fail: m[1] should be true")
	}
	if m[2] == false {
		panic("fail: m[2] should be true")
	}
}
