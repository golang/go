// compile

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Make sure we can linkname to memmove with an unsafe.Pointer argument.

package p

import "unsafe"

//go:linkname memmove runtime.memmove
func memmove(to, from unsafe.Pointer, n uintptr)

var V1, V2 int

func F() {
	memmove(unsafe.Pointer(&V1), unsafe.Pointer(&V2), unsafe.Sizeof(int(0)))
}
