// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

package sha512

import (
	"crypto/internal/fips140deps/cpu"
	"crypto/internal/impl"
)

var useAVX2 = cpu.X86HasAVX && cpu.X86HasAVX2 && cpu.X86HasBMI2

func init() {
	impl.Register("sha512", "AVX2", &useAVX2)
}

//go:noescape
func blockAVX2(dig *Digest, p []byte)

//go:noescape
func blockAMD64(dig *Digest, p []byte)

func block(dig *Digest, p []byte) {
	if useAVX2 {
		blockAVX2(dig, p)
	} else {
		blockAMD64(dig, p)
	}
}
