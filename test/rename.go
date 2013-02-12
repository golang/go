// run

// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that predeclared names can be redeclared by the user.

package main

import (
	"fmt"
	"runtime"
)

func main() {
	n :=
		append +
			bool +
			byte +
			complex +
			complex64 +
			complex128 +
			cap +
			close +
			delete +
			error +
			false +
			float32 +
			float64 +
			imag +
			int +
			int8 +
			int16 +
			int32 +
			int64 +
			len +
			make +
			new +
			nil +
			panic +
			print +
			println +
			real +
			recover +
			rune +
			string +
			true +
			uint +
			uint8 +
			uint16 +
			uint32 +
			uint64 +
			uintptr +
			iota
	if n != NUM*(NUM-1)/2 {
		fmt.Println("BUG: wrong n", n, NUM*(NUM-1)/2)
		runtime.Breakpoint() // panic is inaccessible
	}
}

const (
	// cannot use iota here, because iota = 38 below
	append     = 1
	bool       = 2
	byte       = 3
	complex    = 4
	complex64  = 5
	complex128 = 6
	cap        = 7
	close      = 8
	delete     = 9
	error      = 10
	false      = 11
	float32    = 12
	float64    = 13
	imag       = 14
	int        = 15
	int8       = 16
	int16      = 17
	int32      = 18
	int64      = 19
	len        = 20
	make       = 21
	new        = 22
	nil        = 23
	panic      = 24
	print      = 25
	println    = 26
	real       = 27
	recover    = 28
	rune       = 29
	string     = 30
	true       = 31
	uint       = 32
	uint8      = 33
	uint16     = 34
	uint32     = 35
	uint64     = 36
	uintptr    = 37
	iota       = 38
	NUM        = 39
)
