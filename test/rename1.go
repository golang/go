// errchk $G -e $D/$F.go

// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	var n byte       // ERROR "not a type|expected type"
	var y = float(0) // ERROR "cannot call|expected function"
	const (
		a = 1 + iota // ERROR "string|incompatible types" "convert iota"
	)

}

const (
	bool    = 1
	byte    = 2
	float   = 3
	float32 = 4
	float64 = 5
	int     = 6
	int8    = 7
	int16   = 8
	int32   = 9
	int64   = 10
	uint    = 11
	uint8   = 12
	uint16  = 13
	uint32  = 14
	uint64  = 15
	uintptr = 16
	true    = 17
	false   = 18
	iota    = "abc"
	nil     = 20
	cap     = 21
	len     = 22
	make    = 23
	new     = 24
	panic   = 25
	print   = 26
	println = 27
)
