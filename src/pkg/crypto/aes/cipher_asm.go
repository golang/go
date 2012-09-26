// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build amd64

package aes

// defined in asm_$GOARCH.s
func hasAsm() bool
func encryptBlockAsm(nr int, xk *uint32, dst, src *byte)
func decryptBlockAsm(nr int, xk *uint32, dst, src *byte)
func expandKeyAsm(nr int, key *byte, enc *uint32, dec *uint32)

var useAsm = hasAsm()

func encryptBlock(xk []uint32, dst, src []byte) {
	if useAsm {
		encryptBlockAsm(len(xk)/4-1, &xk[0], &dst[0], &src[0])
	} else {
		encryptBlockGo(xk, dst, src)
	}
}
func decryptBlock(xk []uint32, dst, src []byte) {
	if useAsm {
		decryptBlockAsm(len(xk)/4-1, &xk[0], &dst[0], &src[0])
	} else {
		decryptBlockGo(xk, dst, src)
	}
}
func expandKey(key []byte, enc, dec []uint32) {
	if useAsm {
		rounds := 10
		switch len(key) {
		case 128 / 8:
			rounds = 10
		case 192 / 8:
			rounds = 12
		case 256 / 8:
			rounds = 14
		}
		expandKeyAsm(rounds, &key[0], &enc[0], &dec[0])
	} else {
		expandKeyGo(key, enc, dec)
	}
}
