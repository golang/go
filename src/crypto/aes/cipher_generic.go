// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !amd64

package aes

func encryptBlock(xk []uint32, dst, src []byte) {
	encryptBlockGo(xk, dst, src)
}

func decryptBlock(xk []uint32, dst, src []byte) {
	decryptBlockGo(xk, dst, src)
}

func expandKey(key []byte, enc, dec []uint32) {
	expandKeyGo(key, enc, dec)
}
