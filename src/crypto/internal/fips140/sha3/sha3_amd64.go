// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

package sha3

import (
	"crypto/internal/fips140deps/cpu"
	"crypto/internal/impl"
)

var useAVX2 = cpu.X86HasAVX && cpu.X86HasAVX2

func init() {
	impl.Register("sha3", "AVX2", &useAVX2)
}

//go:noescape
func keccakF1600(a *[200]byte)

// keccakF1600x4AVX2 permutes four states at once. The same state may be
// passed for multiple lanes: every lane is loaded before any is stored, and
// duplicate lanes compute (and store) the same output.
//
//go:noescape
func keccakF1600x4AVX2(a0, a1, a2, a3 *[200]byte)

func keccakF1600Multi(a []*[200]byte) {
	if !useAVX2 {
		keccakF1600MultiGeneric(a)
		return
	}
	for len(a) > 1 {
		// Point unused lanes at one of the used states: a four-lane permutation
		// is cheaper than two or three scalar ones, and the kernel loads all
		// four states before storing any of them.
		a0, a1, a2, a3 := a[0], a[1], a[0], a[1]
		a = a[2:]
		if len(a) > 0 {
			a2 = a[0]
			a = a[1:]
		}
		if len(a) > 0 {
			a3 = a[0]
			a = a[1:]
		}
		keccakF1600x4AVX2(a0, a1, a2, a3)
	}
	if len(a) > 0 {
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
