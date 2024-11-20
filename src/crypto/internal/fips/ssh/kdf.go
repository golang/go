// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package ssh implements the SSH KDF as specified in RFC 4253,
// Section 7.2 and allowed by SP 800-135 Revision 1.
package ssh

import (
	"crypto/internal/fips"
	_ "crypto/internal/fips/check"
)

type Direction struct {
	ivTag     []byte
	keyTag    []byte
	macKeyTag []byte
}

var ServerKeys, ClientKeys Direction

func init() {
	ServerKeys = Direction{[]byte{'B'}, []byte{'D'}, []byte{'F'}}
	ClientKeys = Direction{[]byte{'A'}, []byte{'C'}, []byte{'E'}}
}

func Keys[Hash fips.Hash](hash func() Hash, d Direction,
	K, H, sessionID []byte,
	ivKeyLen, keyLen, macKeyLen int,
) (ivKey, key, macKey []byte) {

	h := hash()
	generateKeyMaterial := func(tag []byte, length int) []byte {
		var key []byte
		for len(key) < length {
			h.Reset()
			h.Write(K)
			h.Write(H)
			if len(key) == 0 {
				h.Write(tag)
				h.Write(sessionID)
			} else {
				h.Write(key)
			}
			key = h.Sum(key)
		}
		return key[:length]
	}

	ivKey = generateKeyMaterial(d.ivTag, ivKeyLen)
	key = generateKeyMaterial(d.keyTag, keyLen)
	macKey = generateKeyMaterial(d.macKeyTag, macKeyLen)

	return
}
