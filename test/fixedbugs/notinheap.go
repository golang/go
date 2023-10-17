// errorcheck -+

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test type-checking errors for go:notinheap.

package p

//go:notinheap
type nih struct{}

type embed4 map[nih]int // ERROR "incomplete \(or unallocatable\) map key not allowed"

type embed5 map[int]nih // ERROR "incomplete \(or unallocatable\) map value not allowed"

type emebd6 chan nih // ERROR "chan of incomplete \(or unallocatable\) type not allowed"

type okay1 *nih

type okay2 []nih

type okay3 func(x nih) nih

type okay4 interface {
	f(x nih) nih
}

// Type conversions don't let you sneak past notinheap.

type t1 struct{ x int }

//go:notinheap
type t2 t1

//go:notinheap
type t3 byte

//go:notinheap
type t4 rune

var sink interface{}

func i() {
	sink = new(t1)                     // no error
	sink = (*t2)(new(t1))              // ERROR "cannot convert(.|\n)*t2 is incomplete \(or unallocatable\)"
	sink = (*t2)(new(struct{ x int })) // ERROR "cannot convert(.|\n)*t2 is incomplete \(or unallocatable\)"
	sink = []t3("foo")                 // ERROR "cannot convert(.|\n)*t3 is incomplete \(or unallocatable\)"
	sink = []t4("bar")                 // ERROR "cannot convert(.|\n)*t4 is incomplete \(or unallocatable\)"
}
