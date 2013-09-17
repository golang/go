// errorcheck -0 -d=nil

// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that nil checks are removed.
// Optimization is enabled.

package p

type Struct struct {
	X int
	Y float64
}

type BigStruct struct {
	X int
	Y float64
	A [1<<20]int
	Z string
}

type Empty struct {
}

type Empty1 struct {
	Empty
}

var (
	intp *int
	arrayp *[10]int
	array0p *[0]int
	bigarrayp *[1<<26]int
	structp *Struct
	bigstructp *BigStruct
	emptyp *Empty
	empty1p *Empty1
)

func f1() {
	_ = *intp // ERROR "generated nil check"
	
	// This one should be removed but the block copy needs
	// to be turned into its own pseudo-op in order to see
	// the indirect.
	_ = *arrayp // ERROR "generated nil check"
	
	// 0-byte indirect doesn't suffice
	_ = *array0p // ERROR "generated nil check"
	_ = *array0p // ERROR "removed repeated nil check" 386

	_ = *intp // ERROR "removed repeated nil check"
	_ = *arrayp // ERROR "removed repeated nil check"
	_ = *structp // ERROR "generated nil check"
	_ = *emptyp // ERROR "generated nil check"
	_ = *arrayp // ERROR "removed repeated nil check"
}

func f2() {
	var (
		intp *int
		arrayp *[10]int
		array0p *[0]int
		bigarrayp *[1<<20]int
		structp *Struct
		bigstructp *BigStruct
		emptyp *Empty
		empty1p *Empty1
	)

	_ = *intp // ERROR "generated nil check"
	_ = *arrayp // ERROR "generated nil check"
	_ = *array0p // ERROR "generated nil check"
	_ = *array0p // ERROR "removed repeated nil check"
	_ = *intp // ERROR "removed repeated nil check"
	_ = *arrayp // ERROR "removed repeated nil check"
	_ = *structp // ERROR "generated nil check"
	_ = *emptyp // ERROR "generated nil check"
	_ = *arrayp // ERROR "removed repeated nil check"
	_ = *bigarrayp // ERROR "generated nil check" ARM removed nil check before indirect!!
	_ = *bigstructp // ERROR "generated nil check"
	_ = *empty1p // ERROR "generated nil check"
}

func fx10k() *[10000]int
var b bool


func f3(x *[10000]int) {
	// Using a huge type and huge offsets so the compiler
	// does not expect the memory hardware to fault.
	_ = x[9999] // ERROR "generated nil check"
	
	for {
		if x[9999] != 0 { // ERROR "generated nil check"
			break
		}
	}
	
	x = fx10k() 
	_ = x[9999] // ERROR "generated nil check"
	if b {
		_ = x[9999] // ERROR "removed repeated nil check"
	} else {
		_ = x[9999] // ERROR "removed repeated nil check"
	}	
	_ = x[9999] // ERROR "generated nil check"

	x = fx10k() 
	if b {
		_ = x[9999] // ERROR "generated nil check"
	} else {
		_ = x[9999] // ERROR "generated nil check"
	}	
	_ = x[9999] // ERROR "generated nil check"
	
	fx10k()
	// This one is a bit redundant, if we figured out that
	// x wasn't going to change across the function call.
	// But it's a little complex to do and in practice doesn't
	// matter enough.
	_ = x[9999] // ERROR "generated nil check"
}

func f3a() {
	x := fx10k()
	y := fx10k()
	z := fx10k()
	_ = &x[9] // ERROR "generated nil check"
	y = z
	_ = &x[9] // ERROR "removed repeated nil check"
	x = y
	_ = &x[9] // ERROR "generated nil check"
}

func f3b() {
	x := fx10k()
	y := fx10k()
	_ = &x[9] // ERROR "generated nil check"
	y = x
	_ = &x[9] // ERROR "removed repeated nil check"
	x = y
	_ = &x[9] // ERROR "removed repeated nil check"
}

func fx10() *[10]int 

func f4(x *[10]int) {
	// Most of these have no checks because a real memory reference follows,
	// and the offset is small enough that if x is nil, the address will still be
	// in the first unmapped page of memory.

	_ = x[9] // ERROR "removed nil check before indirect"
	
	for {
		if x[9] != 0 { // ERROR "removed nil check before indirect"
			break
		}
	}
	
	x = fx10() 
	_ = x[9] // ERROR "removed nil check before indirect"
	if b {
		_ = x[9] // ERROR "removed nil check before indirect"
	} else {
		_ = x[9] // ERROR "removed nil check before indirect"
	}
	_ = x[9] // ERROR "removed nil check before indirect"

	x = fx10() 
	if b {
		_ = x[9] // ERROR "removed nil check before indirect"
	} else {
		_ = &x[9] // ERROR "generated nil check"
	}	
	_ = x[9] // ERROR "removed nil check before indirect"
	
	fx10()
	_ = x[9] // ERROR "removed nil check before indirect"
	
	x = fx10()
	y := fx10()
	_ = &x[9] // ERROR "generated nil check"
	y = x
	_ = &x[9] // ERROR "removed repeated nil check"
	x = y
	_ = &x[9] // ERROR "removed repeated nil check"
}

