// errorcheck

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that illegal composite literals are detected.
// Does not compile.

package main

var m map[int][3]int

func f() [3]int

func fp() *[3]int

var mp map[int]*[3]int

var (
	_ = [3]int{1, 2, 3}[:] // ERROR "cannot slice unaddressable value"
	_ = m[0][:]            // ERROR "cannot slice unaddressable value"
	_ = f()[:]             // ERROR "cannot slice unaddressable value"

	_ = 301[:]  // ERROR "cannot slice|attempt to slice object that is not"
	_ = 3.1[:]  // ERROR "cannot slice|attempt to slice object that is not"
	_ = true[:] // ERROR "cannot slice|attempt to slice object that is not"

	// these are okay because they are slicing a pointer to an array
	_ = (&[3]int{1, 2, 3})[:]
	_ = mp[0][:]
	_ = fp()[:]
)

type T struct {
	i    int
	f    float64
	s    string
	next *T
}

type TP *T
type Ti int

var (
	_ = &T{0, 0, "", nil}               // ok
	_ = &T{i: 0, f: 0, s: "", next: {}} // ERROR "missing type in composite literal|omit types within composite literal"
	_ = &T{0, 0, "", {}}                // ERROR "missing type in composite literal|omit types within composite literal"
	_ = TP{i: 0, f: 0, s: ""}           // ERROR "invalid composite literal type TP"
	_ = &Ti{}                           // ERROR "invalid composite literal type Ti|expected.*type for composite literal"
)

type M map[T]T

var (
	_ = M{{i: 1}: {i: 2}}
	_ = M{T{i: 1}: {i: 2}}
	_ = M{{i: 1}: T{i: 2}}
	_ = M{T{i: 1}: T{i: 2}}
)

type S struct{ s [1]*M1 }
type M1 map[S]int

var _ = M1{{s: [1]*M1{&M1{{}: 1}}}: 2}
