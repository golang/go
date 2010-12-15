// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/hmac"
	"crypto/rc4"
	"hash"
)

// A cipherSuite is a specific combination of key agreement, cipher and MAC
// function. All cipher suites currently assume RSA key agreement.
type cipherSuite struct {
	// the lengths, in bytes, of the key material needed for each component.
	keyLen, macLen, ivLen int
	cipher                func(key, iv []byte, isRead bool) interface{}
	mac                   func(macKey []byte) hash.Hash
}

var cipherSuites = map[uint16]*cipherSuite{
	TLS_RSA_WITH_RC4_128_SHA:     &cipherSuite{16, 20, 0, cipherRC4, hmacSHA1},
	TLS_RSA_WITH_AES_128_CBC_SHA: &cipherSuite{16, 20, 16, cipherAES, hmacSHA1},
}

func cipherRC4(key, iv []byte, isRead bool) interface{} {
	cipher, _ := rc4.NewCipher(key)
	return cipher
}

func cipherAES(key, iv []byte, isRead bool) interface{} {
	block, _ := aes.NewCipher(key)
	if isRead {
		return cipher.NewCBCDecrypter(block, iv)
	}
	return cipher.NewCBCEncrypter(block, iv)
}

func hmacSHA1(key []byte) hash.Hash {
	return hmac.NewSHA1(key)
}

// mutualCipherSuite returns a cipherSuite and its id given a list of supported
// ciphersuites and the id requested by the peer.
func mutualCipherSuite(have []uint16, want uint16) (suite *cipherSuite, id uint16) {
	for _, id := range have {
		if want == id {
			return cipherSuites[id], id
		}
	}
	return
}

// A list of the possible cipher suite ids. Taken from
// http://www.iana.org/assignments/tls-parameters/tls-parameters.xml
const (
	TLS_RSA_WITH_RC4_128_SHA     uint16 = 0x0005
	TLS_RSA_WITH_AES_128_CBC_SHA uint16 = 0x002f
)
