// errorcheck -0 -m -l

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test escape analysis for unique.

package escape

import "unique"

type T string

func f1(s string) unique.Handle[string] { // ERROR "s does not escape$"
	return unique.Make(s)
}

func f1a(s []byte) unique.Handle[string] { // ERROR "s does not escape$"
	return unique.Make(string(s)) // ERROR "string\(s\) does not escape$"
}

func gen[S ~string](s S) unique.Handle[S] {
	return unique.Make(s)
}

func f2(s T) unique.Handle[T] { // ERROR "s does not escape$"
	return unique.Make(s)
}

func f3(s T) unique.Handle[T] { // ERROR "s does not escape$"
	return gen(s)
}

type pair struct {
	s1 string
	s2 string
}

func f4(s1 string, s2 string) unique.Handle[pair] { // ERROR "s1 does not escape$" "s2 does not escape$"
	return unique.Make(pair{s1, s2})
}

type viaInterface struct {
	s any
}

func f5(s string) unique.Handle[viaInterface] { // ERROR "leaking param: s$"
	return unique.Make(viaInterface{s}) // ERROR "s escapes to heap$"
}

var sink any

func f6(s string) unique.Handle[string] { // ERROR "leaking param: s$"
	sink = s // ERROR "s escapes to heap$"
	return unique.Make(s)
}

func f6a(s []byte) unique.Handle[string] { // ERROR "leaking param: s$"
	sink = s                      // ERROR "s escapes to heap$"
	return unique.Make(string(s)) // ERROR "string\(s\) does not escape$"
}
