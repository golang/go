// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This package implements Bruce Schneier's Blowfish encryption algorithm.
package blowfish

// The code is a port of Bruce Schneier's C implementation.
// See http://www.schneier.com/blowfish.html.

import (
	"os"
	"strconv"
)

// The Blowfish block size in bytes.
const BlockSize = 8

// A Cipher is an instance of Blowfish encryption using a particular key.
type Cipher struct {
	p              [18]uint32
	s0, s1, s2, s3 [256]uint32
}

type KeySizeError int

func (k KeySizeError) String() string {
	return "crypto/blowfish: invalid key size " + strconv.Itoa(int(k))
}

// NewCipher creates and returns a Cipher.
// The key argument should be the Blowfish key, 4 to 56 bytes.
func NewCipher(key []byte) (*Cipher, os.Error) {
	k := len(key)
	if k < 4 || k > 56 {
		return nil, KeySizeError(k)
	}
	var result Cipher
	expandKey(key, &result)
	return &result, nil
}

// BlockSize returns the Blowfish block size, 8 bytes.
// It is necessary to satisfy the Cipher interface in the
// package "crypto/block".
func (c *Cipher) BlockSize() int { return BlockSize }

// Encrypt encrypts the 8-byte buffer src using the key k
// and stores the result in dst.
// Note that for amounts of data larger than a block,
// it is not safe to just call Encrypt on successive blocks;
// instead, use an encryption mode like CBC (see crypto/block/cbc.go).
func (c *Cipher) Encrypt(dst, src []byte) {
	l := uint32(src[0])<<24 | uint32(src[1])<<16 | uint32(src[2])<<8 | uint32(src[3])
	r := uint32(src[4])<<24 | uint32(src[5])<<16 | uint32(src[6])<<8 | uint32(src[7])
	l, r = encryptBlock(l, r, c)
	dst[0], dst[1], dst[2], dst[3] = byte(l>>24), byte(l>>16), byte(l>>8), byte(l)
	dst[4], dst[5], dst[6], dst[7] = byte(r>>24), byte(r>>16), byte(r>>8), byte(r)
}

// Decrypt decrypts the 8-byte buffer src using the key k
// and stores the result in dst.
func (c *Cipher) Decrypt(dst, src []byte) {
	l := uint32(src[0])<<24 | uint32(src[1])<<16 | uint32(src[2])<<8 | uint32(src[3])
	r := uint32(src[4])<<24 | uint32(src[5])<<16 | uint32(src[6])<<8 | uint32(src[7])
	l, r = decryptBlock(l, r, c)
	dst[0], dst[1], dst[2], dst[3] = byte(l>>24), byte(l>>16), byte(l>>8), byte(l)
	dst[4], dst[5], dst[6], dst[7] = byte(r>>24), byte(r>>16), byte(r>>8), byte(r)
}

// Reset zeros the key data, so that it will no longer
// appear in the process's memory.
func (c *Cipher) Reset() {
	zero(c.p[0:])
	zero(c.s0[0:])
	zero(c.s1[0:])
	zero(c.s2[0:])
	zero(c.s3[0:])
}
