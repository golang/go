// asmcheck

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

import "unsafe"

func f(p unsafe.Pointer, x, y uintptr) int64 {
	p = unsafe.Pointer(uintptr(p) + x + y)
	// amd64:`MOVQ\s\(.*\)\(.*\*1\), `
	// arm64:`MOVD\s\(R[0-9]+\)\(R[0-9]+\), `
	return *(*int64)(p)
}
