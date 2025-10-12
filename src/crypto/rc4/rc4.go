// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package rc4 implements RC4 encryption, as defined in Bruce Schneier's
// Applied Cryptography.
//
// RC4 is cryptographically broken and should not be used for secure
// applications.
package rc4

import (
	"crypto/internal/fips140/alias"
	"crypto/internal/fips140only"
	"errors"
	"strconv"
)

// A Cipher is an instance of RC4 using a particular key.
type Cipher struct {
	s    [256]uint32
	i, j uint8
}

type KeySizeError int

func (k KeySizeError) Error() string {
	return "crypto/rc4: invalid key size " + strconv.Itoa(int(k))
}

// NewCipher creates and returns a new [Cipher]. The key argument should be the
// RC4 key, at least 1 byte and at most 256 bytes.
func NewCipher(key []byte) (*Cipher, error) {
	if fips140only.Enabled {
		return nil, errors.New("crypto/rc4: use of RC4 is not allowed in FIPS 140-only mode")
	}
	k := len(key)
	if k < 1 || k > 256 {
		return nil, KeySizeError(k)
	}
	var c Cipher
	for i := 0; i < 256; i++ {
		c.s[i] = uint32(i)
	}
	var j uint8 = 0
	for i := 0; i < 256; i++ {
		j += uint8(c.s[i]) + key[i%k]
		c.s[i], c.s[j] = c.s[j], c.s[i]
	}
	return &c, nil
}

// Reset zeros the key data and makes the [Cipher] unusable.
//
// Deprecated: Reset can't guarantee that the key will be entirely removed from
// the process's memory.
func (c *Cipher) Reset() {
	clear(c.s[:])
	c.i, c.j = 0, 0
}

// XORKeyStream sets dst to the result of XORing src with the key stream.
// Dst and src must overlap entirely or not at all.
func (c *Cipher) XORKeyStream(dst, src []byte) {
	if len(src) == 0 {
		return
	}
	if alias.InexactOverlap(dst[:len(src)], src) {
		panic("crypto/rc4: invalid buffer overlap")
	}
	i, j := c.i, c.j
	_ = dst[len(src)-1]
	dst = dst[:len(src)] // eliminate bounds check from loop
	for k, v := range src {
		i += 1
		x := c.s[i]
		j += uint8(x)
		y := c.s[j]
		c.s[i], c.s[j] = y, x
		dst[k] = v ^ uint8(c.s[uint8(x+y)])
	}
	c.i, c.j = i, j
}
