// $G $F.go && $L $F.$A &&./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "os"

func main() {
	var i uint64 =
		' ' +
		'a' +
		'ä' +
		'本' +
		'\a' +
		'\b' +
		'\f' +
		'\n' +
		'\r' +
		'\t' +
		'\v' +
		'\\' +
		'\'' +
		'\000' +
		'\123' +
		'\x00' +
		'\xca' +
		'\xFE' +
		'\u0123' +
		'\ubabe' +
		'\U0010FFFF' +
		'\U000ebabe'
	if '\U000ebabe' != 0x000ebabe {
		print("ebabe wrong\n")
		os.Exit(1)
	}
	if i != 0x20e213 {
		print("number is ", i, " should be ", 0x20e213, "\n")
		os.Exit(1)
	}
}
