// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

func main() {
	n :=
		bool +
			byte +
			float +
			float32 +
			float64 +
			int +
			int8 +
			int16 +
			int32 +
			int64 +
			uint +
			uint8 +
			uint16 +
			uint32 +
			uint64 +
			uintptr +
			true +
			false +
			iota +
			nil +
			cap +
			len +
			make +
			new +
			panic +
			print +
			println
	if n != 27*28/2 {
		fmt.Println("BUG: wrong n", n, 27*28/2)
	}
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
	iota    = 19
	nil     = 20
	cap     = 21
	len     = 22
	make    = 23
	new     = 24
	panic   = 25
	print   = 26
	println = 27
)
