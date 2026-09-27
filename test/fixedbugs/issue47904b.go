// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type M map[int]int

var sink any

func main() {
	sink = M{}
	p := new([8]int)
	m := make(map[any]int)
	m[p] = 42
	if m[p] != 42 {
		panic("incorrect lookup")
	}
}
