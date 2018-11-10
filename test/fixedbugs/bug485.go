// run

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Gccgo chose the wrong embedded method when the same type appeared
// at different levels and the correct choice was not the first
// appearance of the type in a depth-first search.

package main

type embedded string

func (s embedded) val() string {
	return string(s)
}

type A struct {
	embedded
}

type B struct {
	A
	embedded
}

func main() {
	b := &B{
		A: A{
			embedded: "a",
		},
		embedded: "b",
	}
	s := b.val()
	if s != "b" {
		panic(s)
	}
}
