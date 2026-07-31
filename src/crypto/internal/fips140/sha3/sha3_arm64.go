// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

package sha3

import (
	"crypto/internal/fips140deps/cpu"
	"crypto/internal/impl"
	"runtime"
)

// On non-Apple ARM64, using only one of the two SIMD lanes of the SHA-3
// instructions is slower than the pure Go implementation.
//
// Checking GOOS is a bit blunt, as it also excludes Asahi Linux; we might
// consider checking the MIDR model in the future.
var useSHA3 = cpu.ARM64HasSHA3 && (runtime.GOOS == "darwin" || runtime.GOOS == "ios")
var useSHA3x2 = cpu.ARM64HasSHA3

func init() {
	impl.Register("sha3", "Armv8.2", &useSHA3)
	impl.Register("sha3", "Armv8.2-x2", &useSHA3x2)
}

//go:noescape
func keccakF1600x2NEON(a0, a1 *[200]byte)

func keccakF1600(a *[200]byte) {
	if useSHA3 {
		keccakF1600x2NEON(a, nil)
	} else {
		keccakF1600Generic(a)
	}
}

func keccakF1600Multi(a []*[200]byte) {
	if useSHA3x2 {
		for len(a) >= 2 {
			keccakF1600x2NEON(a[0], a[1])
			a = a[2:]
		}
		if len(a) == 1 {
			keccakF1600(a[0])
		}
	} else {
		keccakF1600MultiGeneric(a)
	}
}

func (d *Digest) write(p []byte) (n int, err error) {
	return d.writeGeneric(p)
}
func (d *Digest) read(out []byte) (n int, err error) {
	return d.readGeneric(out)
}
func (d *Digest) sum(b []byte) []byte {
	return d.sumGeneric(b)
}
