// $G $D/$F.go || echo BUG: bug246

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "unsafe"

func main() {
	// works
	addr := uintptr(0)
	_ = (*int)(unsafe.Pointer(addr))

	// fails
	_ = (*int)(unsafe.Pointer(uintptr(0)))
}
