// [ $GOOS != nacl ] || exit 0  # do not bother on NaCl
// $G $D/$F.go && $L $F.$A &&
//	((! sh -c ./$A.out) >/dev/null 2>&1 || echo BUG: should fail)

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "unsafe"

var dummy [512<<20]byte	// give us a big address space
var q *[1<<30]byte
func main() {
	// the test only tests what we intend to test
	// if dummy starts in the first 256 MB of memory.
	// otherwise there might not be anything mapped
	// at the address that might be accidentally
	// dereferenced below.
	if uintptr(unsafe.Pointer(&dummy)) > 256<<20 {
		panic("dummy too far out")
	}

	// The problem here is that indexing into p[] with a large
	// enough index can jump out of the unmapped section
	// at the beginning of memory and into valid memory.
	//
	// To avoid needing a check on every slice beyond the
	// usual len and cap, we require the *array -> slice
	// conversion to do the check.
	var x []byte
	var y = &x
	*y = q[0:]	// should crash (uses arraytoslice runtime routine)
}
