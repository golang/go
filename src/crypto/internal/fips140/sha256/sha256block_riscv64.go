// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

package sha256

import (
	"crypto/internal/fips140deps/cpu"
	"crypto/internal/impl"
)

var supportsZvknha = cpu.RISCV64HasZvknha && cpu.RISCV64HasZvbb && cpu.RISCV64VLENB >= 16

func init() {
	impl.Register("sha256", "Zvknha", &supportsZvknha)
}

//go:noescape
func blockZvknha(dig *Digest, p []byte)

//go:noescape
func blockScalar(dig *Digest, p []byte)

func block(dig *Digest, p []byte) {
	if supportsZvknha {
		blockZvknha(dig, p)
	} else {
		blockScalar(dig, p)
	}
}
