// errorcheck

// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that renamed identifiers no longer have their old meaning.
// Does not compile.

package main

func main() {
	var n byte       // ERROR "not a type|expected type"
	var y = float32(0) // ERROR "cannot call|expected function"
	const (
		a = 1 + iota // ERROR "string|incompatible types" "convert iota"
	)

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
	iota = "123"
)
