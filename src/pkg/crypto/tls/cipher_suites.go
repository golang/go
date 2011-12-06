// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/des"
	"crypto/hmac"
	"crypto/rc4"
	"crypto/sha1"
	"crypto/x509"
	"hash"
)

// a keyAgreement implements the client and server side of a TLS key agreement
// protocol by generating and processing key exchange messages.
type keyAgreement interface {
	// On the server side, the first two methods are called in order.

	// In the case that the key agreement protocol doesn't use a
	// ServerKeyExchange message, generateServerKeyExchange can return nil,
	// nil.
	generateServerKeyExchange(*Config, *clientHelloMsg, *serverHelloMsg) (*serverKeyExchangeMsg, error)
	processClientKeyExchange(*Config, *clientKeyExchangeMsg, uint16) ([]byte, error)

	// On the client side, the next two methods are called in order.

	// This method may not be called if the server doesn't send a
	// ServerKeyExchange message.
	processServerKeyExchange(*Config, *clientHelloMsg, *serverHelloMsg, *x509.Certificate, *serverKeyExchangeMsg) error
	generateClientKeyExchange(*Config, *clientHelloMsg, *x509.Certificate) ([]byte, *clientKeyExchangeMsg, error)
}

// A cipherSuite is a specific combination of key agreement, cipher and MAC
// function. All cipher suites currently assume RSA key agreement.
type cipherSuite struct {
	id uint16
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
	mac      func(version uint16, macKey []byte) macFunction
}

var cipherSuites = []*cipherSuite{
	{TLS_RSA_WITH_RC4_128_SHA, 16, 20, 0, rsaKA, false, cipherRC4, macSHA1},
	{TLS_RSA_WITH_3DES_EDE_CBC_SHA, 24, 20, 8, rsaKA, false, cipher3DES, macSHA1},
	{TLS_RSA_WITH_AES_128_CBC_SHA, 16, 20, 16, rsaKA, false, cipherAES, macSHA1},
	{TLS_ECDHE_RSA_WITH_RC4_128_SHA, 16, 20, 0, ecdheRSAKA, true, cipherRC4, macSHA1},
	{TLS_ECDHE_RSA_WITH_3DES_EDE_CBC_SHA, 24, 20, 8, ecdheRSAKA, true, cipher3DES, macSHA1},
	{TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA, 16, 20, 16, ecdheRSAKA, true, cipherAES, macSHA1},
}

func cipherRC4(key, iv []byte, isRead bool) interface{} {
	cipher, _ := rc4.NewCipher(key)
	return cipher
}

func cipher3DES(key, iv []byte, isRead bool) interface{} {
	block, _ := des.NewTripleDESCipher(key)
	if isRead {
		return cipher.NewCBCDecrypter(block, iv)
	}
	return cipher.NewCBCEncrypter(block, iv)
}

func cipherAES(key, iv []byte, isRead bool) interface{} {
	block, _ := aes.NewCipher(key)
	if isRead {
		return cipher.NewCBCDecrypter(block, iv)
	}
	return cipher.NewCBCEncrypter(block, iv)
}

// macSHA1 returns a macFunction for the given protocol version.
func macSHA1(version uint16, key []byte) macFunction {
	if version == versionSSL30 {
		mac := ssl30MAC{
			h:   sha1.New(),
			key: make([]byte, len(key)),
		}
		copy(mac.key, key)
		return mac
	}
	return tls10MAC{hmac.NewSHA1(key)}
}

type macFunction interface {
	Size() int
	MAC(digestBuf, seq, data []byte) []byte
}

// ssl30MAC implements the SSLv3 MAC function, as defined in
// www.mozilla.org/projects/security/pki/nss/ssl/draft302.txt section 5.2.3.1
type ssl30MAC struct {
	h   hash.Hash
	key []byte
}

func (s ssl30MAC) Size() int {
	return s.h.Size()
}

var ssl30Pad1 = [48]byte{0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36, 0x36}

var ssl30Pad2 = [48]byte{0x5c, 0x5c, 0x5c, 0x5c, 0x5c, 0x5c, 0x5c, 0x5c, 0x5c, 0x5c, 0x5c, 0x5c, 0x5c, 0x5c, 0x5c, 0x5c, 0x5c, 0x5c, 0x5c, 0x5c, 0x5c, 0x5c, 0x5c, 0x5c, 0x5c, 0x5c, 0x5c, 0x5c, 0x5c, 0x5c, 0x5c, 0x5c, 0x5c, 0x5c, 0x5c, 0x5c, 0x5c, 0x5c, 0x5c, 0x5c, 0x5c, 0x5c, 0x5c, 0x5c, 0x5c, 0x5c, 0x5c, 0x5c}

func (s ssl30MAC) MAC(digestBuf, seq, record []byte) []byte {
	padLength := 48
	if s.h.Size() == 20 {
		padLength = 40
	}

	s.h.Reset()
	s.h.Write(s.key)
	s.h.Write(ssl30Pad1[:padLength])
	s.h.Write(seq)
	s.h.Write(record[:1])
	s.h.Write(record[3:5])
	s.h.Write(record[recordHeaderLen:])
	digestBuf = s.h.Sum(digestBuf[:0])

	s.h.Reset()
	s.h.Write(s.key)
	s.h.Write(ssl30Pad2[:padLength])
	s.h.Write(digestBuf)
	return s.h.Sum(digestBuf[:0])
}

// tls10MAC implements the TLS 1.0 MAC function. RFC 2246, section 6.2.3.
type tls10MAC struct {
	h hash.Hash
}

func (s tls10MAC) Size() int {
	return s.h.Size()
}

func (s tls10MAC) MAC(digestBuf, seq, record []byte) []byte {
	s.h.Reset()
	s.h.Write(seq)
	s.h.Write(record)
	return s.h.Sum(digestBuf[:0])
}

func rsaKA() keyAgreement {
	return rsaKeyAgreement{}
}

func ecdheRSAKA() keyAgreement {
	return new(ecdheRSAKeyAgreement)
}

// mutualCipherSuite returns a cipherSuite given a list of supported
// ciphersuites and the id requested by the peer.
func mutualCipherSuite(have []uint16, want uint16) *cipherSuite {
	for _, id := range have {
		if id == want {
			for _, suite := range cipherSuites {
				if suite.id == want {
					return suite
				}
			}
			return nil
		}
	}
	return nil
}

// A list of the possible cipher suite ids. Taken from
// http://www.iana.org/assignments/tls-parameters/tls-parameters.xml
const (
	TLS_RSA_WITH_RC4_128_SHA            uint16 = 0x0005
	TLS_RSA_WITH_3DES_EDE_CBC_SHA       uint16 = 0x000a
	TLS_RSA_WITH_AES_128_CBC_SHA        uint16 = 0x002f
	TLS_ECDHE_RSA_WITH_RC4_128_SHA      uint16 = 0xc011
	TLS_ECDHE_RSA_WITH_3DES_EDE_CBC_SHA uint16 = 0xc012
	TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA  uint16 = 0xc013
)
