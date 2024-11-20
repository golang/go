// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

package sha256

import (
	"crypto/internal/fipsdeps/cpu"
	"crypto/internal/impl"
)

var useAVX2 = cpu.X86HasAVX && cpu.X86HasAVX2 && cpu.X86HasBMI2
var useSHANI = cpu.X86HasAVX && cpu.X86HasSHA && cpu.X86HasSSE41 && cpu.X86HasSSSE3

func init() {
	impl.Register("sha256", "AVX2", &useAVX2)
	impl.Register("sha256", "SHA-NI", &useSHANI)
}

//go:noescape
func blockAMD64(dig *Digest, p []byte)

//go:noescape
func blockAVX2(dig *Digest, p []byte)

//go:noescape
func blockSHANI(dig *Digest, p []byte)

func block(dig *Digest, p []byte) {
	if useSHANI {
		blockSHANI(dig, p)
	} else if useAVX2 {
		blockAVX2(dig, p)
	} else {
		blockAMD64(dig, p)
	}
}
