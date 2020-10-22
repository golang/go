// run

// Copyright 2020 The Go Authors. All rights reserved.  Use of this
// source code is governed by a BSD-style license that can be found in
// the LICENSE file.

package main

//go:notinheap
type NIH struct {
}

type T struct {
	x *NIH
	p *int
}

var y NIH
var z int

func main() {
	a := []T{{&y, &z}}
	a = append(a, T{&y, &z})
	if a[1].x == nil {
		panic("pointer not written")
	}
}
