// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (!amd64 && !s390x && !ppc64 && !ppc64le && !arm64) || purego

package aes

type block struct {
	blockExpanded
}

func newBlock(c *Block, key []byte) *Block {
	newBlockExpanded(&c.blockExpanded, key)
	return c
}

func encryptBlock(c *Block, dst, src []byte) {
	encryptBlockGeneric(&c.blockExpanded, dst, src)
}

func decryptBlock(c *Block, dst, src []byte) {
	decryptBlockGeneric(&c.blockExpanded, dst, src)
}

func checkGenericIsExpected() {}
