// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (!amd64 && !s390x && !ppc64 && !ppc64le && !arm64) || purego

package aes

func newGCM(c *Block, nonceSize, tagSize int) (*GCM, error) {
	return nil, nil
}

type GCM struct{}

func (g *GCM) NonceSize() int {
	panic("not implemented")
}

func (g *GCM) Overhead() int {
	panic("not implemented")
}

func (g *GCM) Seal(dst, nonce, plaintext, additionalData []byte) []byte {
	panic("not implemented")
}

func (g *GCM) Open(dst, nonce, ciphertext, additionalData []byte) ([]byte, error) {
	panic("not implemented")
}
