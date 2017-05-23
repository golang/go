// errorcheck -0 -N -d=nil

// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that nil checks are inserted.
// Optimization is disabled, so redundant checks are not removed.

package p

type Struct struct {
	X int
	Y float64
}

type BigStruct struct {
	X int
	Y float64
	A [1 << 20]int
	Z string
}

type Empty struct {
}

type Empty1 struct {
	Empty
}

var (
	intp       *int
	arrayp     *[10]int
	array0p    *[0]int
	bigarrayp  *[1 << 26]int
	structp    *Struct
	bigstructp *BigStruct
	emptyp     *Empty
	empty1p    *Empty1
)

func f1() {
	_ = *intp    // ERROR "nil check"
	_ = *arrayp  // ERROR "nil check"
	_ = *array0p // ERROR "nil check"
	_ = *array0p // ERROR "nil check"
	_ = *intp    // ERROR "nil check"
	_ = *arrayp  // ERROR "nil check"
	_ = *structp // ERROR "nil check"
	_ = *emptyp  // ERROR "nil check"
	_ = *arrayp  // ERROR "nil check"
}

func f2() {
	var (
		intp       *int
		arrayp     *[10]int
		array0p    *[0]int
		bigarrayp  *[1 << 20]int
		structp    *Struct
		bigstructp *BigStruct
		emptyp     *Empty
		empty1p    *Empty1
	)

	_ = *intp       // ERROR "nil check"
	_ = *arrayp     // ERROR "nil check"
	_ = *array0p    // ERROR "nil check"
	_ = *array0p    // ERROR "nil check"
	_ = *intp       // ERROR "nil check"
	_ = *arrayp     // ERROR "nil check"
	_ = *structp    // ERROR "nil check"
	_ = *emptyp     // ERROR "nil check"
	_ = *arrayp     // ERROR "nil check"
	_ = *bigarrayp  // ERROR "nil check"
	_ = *bigstructp // ERROR "nil check"
	_ = *empty1p    // ERROR "nil check"
}

func fx10k() *[10000]int

var b bool

func f3(x *[10000]int) {
	// Using a huge type and huge offsets so the compiler
	// does not expect the memory hardware to fault.
	_ = x[9999] // ERROR "nil check"

	for {
		if x[9999] != 0 { // ERROR "nil check"
			break
		}
	}

	x = fx10k()
	_ = x[9999] // ERROR "nil check"
	if b {
		_ = x[9999] // ERROR "nil check"
	} else {
		_ = x[9999] // ERROR "nil check"
	}
	_ = x[9999] // ERROR "nil check"

	x = fx10k()
	if b {
		_ = x[9999] // ERROR "nil check"
	} else {
		_ = x[9999] // ERROR "nil check"
	}
	_ = x[9999] // ERROR "nil check"

	fx10k()
	// This one is a bit redundant, if we figured out that
	// x wasn't going to change across the function call.
	// But it's a little complex to do and in practice doesn't
	// matter enough.
	_ = x[9999] // ERROR "nil check"
}

func f3a() {
	x := fx10k()
	y := fx10k()
	z := fx10k()
	_ = &x[9] // ERROR "nil check"
	y = z
	_ = &x[9] // ERROR "nil check"
	x = y
	_ = &x[9] // ERROR "nil check"
}

func f3b() {
	x := fx10k()
	y := fx10k()
	_ = &x[9] // ERROR "nil check"
	y = x
	_ = &x[9] // ERROR "nil check"
	x = y
	_ = &x[9] // ERROR "nil check"
}

func fx10() *[10]int

func f4(x *[10]int) {
	// Most of these have no checks because a real memory reference follows,
	// and the offset is small enough that if x is nil, the address will still be
	// in the first unmapped page of memory.

	_ = x[9] // ERROR "nil check"

	for {
		if x[9] != 0 { // ERROR "nil check"
			break
		}
	}

	x = fx10()
	_ = x[9] // ERROR "nil check"
	if b {
		_ = x[9] // ERROR "nil check"
	} else {
		_ = x[9] // ERROR "nil check"
	}
	_ = x[9] // ERROR "nil check"

	x = fx10()
	if b {
		_ = x[9] // ERROR "nil check"
	} else {
		_ = &x[9] // ERROR "nil check"
	}
	_ = x[9] // ERROR "nil check"

	fx10()
	_ = x[9] // ERROR "nil check"

	x = fx10()
	y := fx10()
	_ = &x[9] // ERROR "nil check"
	y = x
	_ = &x[9] // ERROR "nil check"
	x = y
	_ = &x[9] // ERROR "nil check"
}

func f5(m map[string]struct{}) bool {
	// Existence-only map lookups should not generate a nil check
	_, ok := m[""]
	return ok
}
