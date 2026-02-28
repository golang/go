// run

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This test is not for a fix of 29312 proper, but for the patch that
// makes sure we at least don't have a security hole because of 29312.

// This code generates lots of types. The binary should contain
// a runtime.slicetype for each of the following 253 types:
//
//     []*pwn
//     [][]*pwn
//     ...
//     [][]...[][]*pwn          - 249 total "[]"
//     [][]...[][][]*pwn        - 250 total "[]"
//     [][]...[][][][]*pwn      - 251 total "[]"
//     [][]...[][][][][]*pwn    - 252 total "[]"
//     [][]...[][][][][][]*pwn  - 253 total "[]"
//
// The type names for these types are as follows. Because we truncate
// the name at depth 250, the last few names are all identical:
//
//     type:[]*"".pwn
//     type:[][]*"".pwn
//     ...
//     type:[][]...[][]*pwn       - 249 total "[]"
//     type:[][]...[][][]*<...>   - 250 total "[]"
//     type:[][]...[][][][]<...>  - 251 total "[]"
//     type:[][]...[][][][]<...>  - 252 total "[]" (but only 251 "[]" in the name)
//     type:[][]...[][][][]<...>  - 253 total "[]" (but only 251 "[]" in the name)
//
// Because the names of the last 3 types are all identical, the
// compiler will generate only a single runtime.slicetype data
// structure for all 3 underlying types. It turns out the compiler
// generates just the 251-entry one. There aren't any
// runtime.slicetypes generated for the final two types.
//
// The compiler passes type:[]...[]<...> (251 total "[]") to
// fmt.Sprintf (instead of the correct 253 one). But the data
// structure at runtime actually has 253 nesting levels. So we end up
// calling String on something that is of type [][]*pwn instead of
// something of type *pwn. The way arg passing in Go works, the
// backing store pointer for the outer slice becomes the "this"
// pointer of the String method, which points to the inner []*pwn
// slice.  The String method then modifies the length of that inner
// slice.
package main

import "fmt"

type pwn struct {
	a [3]uint
}

func (this *pwn) String() string {
	this.a[1] = 7 // update length
	return ""
}

func main() {
	var a pwn
	s := [][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][][]*pwn{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{&a}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}} // depth 253
	fmt.Sprint(s)
	n := len(s[0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0][0]) // depth 252, type []*pwn
	if n != 1 {
		panic(fmt.Sprintf("length was changed, want 1 got %d", n))
	}
}
