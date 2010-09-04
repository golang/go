// [ $GOOS != nacl ] || exit 0  # do not bother on NaCl
// $G $D/$F.go && $L $F.$A &&
//	((! sh -c ./$A.out) >/dev/null 2>&1 || echo BUG: should fail)

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "unsafe"

var x byte

func main() {
	var p *[1<<30]byte = nil
	x = 123

	// The problem here is not the use of unsafe:
	// it is that indexing into p[] with a large
	// enough index jumps out of the unmapped section
	// at the beginning of memory and into valid memory.
	// Pointer offsets and array indices, if they are
	// very large, need to dereference the base pointer
	// to trigger a trap.
	println(p[uintptr(unsafe.Pointer(&x))])	// should crash
}
