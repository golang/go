// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

package subtle

import (
	"crypto/internal/fips140deps/cpu"
	"crypto/internal/impl"
)

var useLSX = cpu.LOONG64HasLSX
var useLASX = cpu.LOONG64HasLASX

func init() {
	impl.Register("subtle", "LSX", &useLSX)
	impl.Register("subtle", "LASX", &useLASX)
}

//go:noescape
func xorBytesBasic(dst, a, b *byte, n int)

//go:noescape
func xorBytesLSX(dst, a, b *byte, n int)

//go:noescape
func xorBytesLASX(dst, a, b *byte, n int)

func xorBytes(dst, a, b *byte, n int) {
	if useLASX {
		xorBytesLASX(dst, a, b, n)
	} else if useLSX {
		xorBytesLSX(dst, a, b, n)
	} else {
		xorBytesBasic(dst, a, b, n)
	}
}
