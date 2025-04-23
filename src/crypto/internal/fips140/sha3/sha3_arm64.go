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

// On non-Apple ARM64, the SHA-3 instructions are apparently slower than the
// pure Go implementation. Checking GOOS is a bit blunt, as it also excludes
// Asahi Linux; we might consider checking the MIDR model in the future.
var useSHA3 = cpu.ARM64HasSHA3 && runtime.GOOS == "darwin"

func init() {
	impl.Register("sha3", "Armv8.2", &useSHA3)
}

//go:noescape
func keccakF1600NEON(a *[200]byte)

func keccakF1600(a *[200]byte) {
	if useSHA3 {
		keccakF1600NEON(a)
	} else {
		keccakF1600Generic(a)
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
