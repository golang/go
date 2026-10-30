// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

package sha3

import (
	"crypto/internal/fips140deps/cpu"
	"crypto/internal/impl"
)

var useLSX = cpu.LOONG64HasLSX

func init() {
	impl.Register("sha3", "LSX", &useLSX)
}

//go:noescape
func keccakF1600LSX(a *[200]byte)

func keccakF1600(a *[200]byte) {
	if useLSX {
		keccakF1600LSX(a)
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
