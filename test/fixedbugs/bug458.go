// compile

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 4200: 6g crashes when a type is larger than 4GB.

package main

import "unsafe"

// N=16 on 32-bit arches, 256 on 64-bit arches.
// On 32-bit arches we don't want to test types
// that are over 4GB large.
const N = 1 << unsafe.Sizeof(uintptr(0))

type T [N][10][10][10][10][3]byte

func F(t *T) byte {
	return t[0][0][0][0][0][0]
}
