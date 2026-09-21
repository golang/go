// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build riscv64 && !purego

package subtle

import (
	"crypto/internal/fips140deps/cpu"
)

//go:noescape
func xorBytesRISCV64(dst, a, b *byte, n int, hasV bool)

func xorBytes(dst, a, b *byte, n int) {
	xorBytesRISCV64(dst, a, b, n, cpu.RISCV64HasV)
}
