// run

// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that predeclared names can be redeclared by the user.

package main

import "fmt"

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
	}
}

const (
	append = iota
	bool
	byte
	complex
	complex64
	complex128
	cap
	close
	delete
	error
	false
	float32
	float64
	imag
	int
	int8
	int16
	int32
	int64
	len
	make
	new
	nil
	panic
	print
	println
	real
	recover
	rune
	string
	true
	uint
	uint8
	uint16
	uint32
	uint64
	uintptr
	NUM
	iota = 0
)
