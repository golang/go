// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

package sha3

//go:noescape
func keccakF1600(a *[200]byte)

func (d *Digest) write(p []byte) (n int, err error) {
	return d.writeGeneric(p)
}
func (d *Digest) read(out []byte) (n int, err error) {
	return d.readGeneric(out)
}
func (d *Digest) sum(b []byte) []byte {
	return d.sumGeneric(b)
}
