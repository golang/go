// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/hmac"
	"crypto/rc4"
	"crypto/x509"
	"hash"
	"os"
)

// a keyAgreement implements the client and server side of a TLS key agreement
// protocol by generating and processing key exchange messages.
type keyAgreement interface {
	// On the server side, the first two methods are called in order.

	// In the case that the key agreement protocol doesn't use a
	// ServerKeyExchange message, generateServerKeyExchange can return nil,
	// nil.
	generateServerKeyExchange(*Config, *clientHelloMsg, *serverHelloMsg) (*serverKeyExchangeMsg, os.Error)
	processClientKeyExchange(*Config, *clientKeyExchangeMsg) ([]byte, os.Error)

	// On the client side, the next two methods are called in order.

	// This method may not be called if the server doesn't send a
	// ServerKeyExchange message.
	processServerKeyExchange(*Config, *clientHelloMsg, *serverHelloMsg, *x509.Certificate, *serverKeyExchangeMsg) os.Error
	generateClientKeyExchange(*Config, *clientHelloMsg, *x509.Certificate) ([]byte, *clientKeyExchangeMsg, os.Error)
}

// A cipherSuite is a specific combination of key agreement, cipher and MAC
// function. All cipher suites currently assume RSA key agreement.
type cipherSuite struct {
	// the lengths, in bytes, of the key material needed for each component.
	keyLen int
	macLen int
	ivLen  int
	ka     func() keyAgreement
	// If elliptic is set, a server will only consider this ciphersuite if
	// the ClientHello indicated that the client supports an elliptic curve
	// and point format that we can handle.
	elliptic bool
	cipher   func(key, iv []byte, isRead bool) interface{}
	mac      func(macKey []byte) hash.Hash
}

var cipherSuites = map[uint16]*cipherSuite{
	TLS_RSA_WITH_RC4_128_SHA:           &cipherSuite{16, 20, 0, rsaKA, false, cipherRC4, hmacSHA1},
	TLS_RSA_WITH_AES_128_CBC_SHA:       &cipherSuite{16, 20, 16, rsaKA, false, cipherAES, hmacSHA1},
	TLS_ECDHE_RSA_WITH_RC4_128_SHA:     &cipherSuite{16, 20, 0, ecdheRSAKA, true, cipherRC4, hmacSHA1},
	TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA: &cipherSuite{16, 20, 16, ecdheRSAKA, true, cipherAES, hmacSHA1},
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

func rsaKA() keyAgreement {
	return rsaKeyAgreement{}
}

func ecdheRSAKA() keyAgreement {
	return new(ecdheRSAKeyAgreement)
}

// mutualCipherSuite returns a cipherSuite and its id given a list of supported
// ciphersuites and the id requested by the peer.
func mutualCipherSuite(have []uint16, want uint16) (suite *cipherSuite, id uint16) {
	for _, id := range have {
		if id == want {
			return cipherSuites[id], id
		}
	}
	return
}

// A list of the possible cipher suite ids. Taken from
// http://www.iana.org/assignments/tls-parameters/tls-parameters.xml
const (
	TLS_RSA_WITH_RC4_128_SHA           uint16 = 0x0005
	TLS_RSA_WITH_AES_128_CBC_SHA       uint16 = 0x002f
	TLS_ECDHE_RSA_WITH_RC4_128_SHA     uint16 = 0xc011
	TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA uint16 = 0xc013
)
