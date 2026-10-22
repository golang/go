// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build loong64 && !purego

package sha512

import (
	"crypto/internal/fips140deps/cpu"
	"crypto/internal/impl"
)

var supportsLSX = cpu.LOONG64HasLSX

func init() {
	impl.Register("sha512", "LSX", &supportsLSX)
}

//go:noescape
func blockScalar(dig *Digest, p []byte)

//go:noescape
func blockLSX(dig *Digest, p []byte)

func block(dig *Digest, p []byte) {
	if supportsLSX {
		blockLSX(dig, p)
	} else {
		blockScalar(dig, p)
	}
}
