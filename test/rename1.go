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
	iota       = "38"
)
