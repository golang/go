// compile -d=ssa/check/on

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 36723: fail to compile on PPC64 when SSA check is on.

package p

import "unsafe"

type T struct {
	a, b, c, d uint8
	x          [10]int32
}

func F(p *T, i uintptr) int32 {
	// load p.x[i] using unsafe, derived from runtime.pcdatastart
	_ = *p
	return *(*int32)(add(unsafe.Pointer(&p.d), unsafe.Sizeof(p.d)+i*unsafe.Sizeof(p.x[0])))
}

func add(p unsafe.Pointer, x uintptr) unsafe.Pointer {
	return unsafe.Pointer(uintptr(p) + x)
}
