// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

package sha512

import (
	"crypto/internal/fips140deps/cpu"
	"crypto/internal/impl"
)

var supportsZvknhb = cpu.RISCV64HasZvknhb && cpu.RISCV64HasZvbb && cpu.RISCV64VLENB >= 32

func init() {
	impl.Register("sha512", "Zvknhb", &supportsZvknhb)
}

//go:noescape
func blockZvknhb(dig *Digest, p []byte)

//go:noescape
func blockScalar(dig *Digest, p []byte)

func block(dig *Digest, p []byte) {
	if supportsZvknhb {
		blockZvknhb(dig, p)
	} else {
		blockScalar(dig, p)
	}
}
