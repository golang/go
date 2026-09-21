// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type S [5]*byte

//go:noinline
func f() S {
	return S{}
}

var sink *S

//go:noinline
func g() (s S) {
	sink = &s
	s = f()
	return
}

func main() {
	for i := range 1000000 {
		s := g()
		if s != (S{}) {
			println(s[0], s[1], s[2], s[3], s[4])
			panic(i)
		}
	}
}
