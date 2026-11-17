// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simdstable

// extra.go is a second file in package iface that intentionally does not import
// "simd". It validates that midway injects its private @simd/@bridge imports
// and rewrites both SIMD-dependent functions (ExtraSimdLen) and package-level
// variable initializers (ExtraSimdInit) in files without direct "simd" imports.
package iface

import (
	"unsafe"
)

func makeUnsafePtr(p *int) unsafe.Pointer {
	return unsafe.Pointer(p)
}

func ExtraSimdLen() int {
	return Generic[MyInt8s](0)
}

var ExtraSimdInit = Generic[MyInt8s](2)
