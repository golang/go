// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rsa

import (
	"big"
	"crypto"
	"crypto/subtle"
	"errors"
	"io"
)

// This file implements encryption and decryption using PKCS#1 v1.5 padding.

// EncryptPKCS1v15 encrypts the given message with RSA and the padding scheme from PKCS#1 v1.5.
// The message must be no longer than the length of the public modulus minus 11 bytes.
// WARNING: use of this function to encrypt plaintexts other than session keys
// is dangerous. Use RSA OAEP in new protocols.
func EncryptPKCS1v15(rand io.Reader, pub *PublicKey, msg []byte) (out []byte, err error) {
	k := (pub.N.BitLen() + 7) / 8
	if len(msg) > k-11 {
		err = MessageTooLongError{}
		return
	}

	// EM = 0x02 || PS || 0x00 || M
	em := make([]byte, k-1)
	em[0] = 2
	ps, mm := em[1:len(em)-len(msg)-1], em[len(em)-len(msg):]
	err = nonZeroRandomBytes(ps, rand)
	if err != nil {
		return
	}
	em[len(em)-len(msg)-1] = 0
	copy(mm, msg)

	m := new(big.Int).SetBytes(em)
	c := encrypt(new(big.Int), pub, m)
	out = c.Bytes()
	return
}

// DecryptPKCS1v15 decrypts a plaintext using RSA and the padding scheme from PKCS#1 v1.5.
// If rand != nil, it uses RSA blinding to avoid timing side-channel attacks.
func DecryptPKCS1v15(rand io.Reader, priv *PrivateKey, ciphertext []byte) (out []byte, err error) {
	valid, out, err := decryptPKCS1v15(rand, priv, ciphertext)
	if err == nil && valid == 0 {
		err = DecryptionError{}
	}

	return
}

// DecryptPKCS1v15SessionKey decrypts a session key using RSA and the padding scheme from PKCS#1 v1.5.
// If rand != nil, it uses RSA blinding to avoid timing side-channel attacks.
// It returns an error if the ciphertext is the wrong length or if the
// ciphertext is greater than the public modulus. Otherwise, no error is
// returned. If the padding is valid, the resulting plaintext message is copied
// into key. Otherwise, key is unchanged. These alternatives occur in constant
// time. It is intended that the user of this function generate a random
// session key beforehand and continue the protocol with the resulting value.
// This will remove any possibility that an attacker can learn any information
// about the plaintext.
// See ``Chosen Ciphertext Attacks Against Protocols Based on the RSA
// Encryption Standard PKCS #1'', Daniel Bleichenbacher, Advances in Cryptology
// (Crypto '98),
func DecryptPKCS1v15SessionKey(rand io.Reader, priv *PrivateKey, ciphertext []byte, key []byte) (err error) {
	k := (priv.N.BitLen() + 7) / 8
	if k-(len(key)+3+8) < 0 {
		err = DecryptionError{}
		return
	}

	valid, msg, err := decryptPKCS1v15(rand, priv, ciphertext)
	if err != nil {
		return
	}

	valid &= subtle.ConstantTimeEq(int32(len(msg)), int32(len(key)))
	subtle.ConstantTimeCopy(valid, key, msg)
	return
}

func decryptPKCS1v15(rand io.Reader, priv *PrivateKey, ciphertext []byte) (valid int, msg []byte, err error) {
	k := (priv.N.BitLen() + 7) / 8
	if k < 11 {
		err = DecryptionError{}
		return
	}

	c := new(big.Int).SetBytes(ciphertext)
	m, err := decrypt(rand, priv, c)
	if err != nil {
		return
	}

	em := leftPad(m.Bytes(), k)
	firstByteIsZero := subtle.ConstantTimeByteEq(em[0], 0)
	secondByteIsTwo := subtle.ConstantTimeByteEq(em[1], 2)

	// The remainder of the plaintext must be a string of non-zero random
	// octets, followed by a 0, followed by the message.
	//   lookingForIndex: 1 iff we are still looking for the zero.
	//   index: the offset of the first zero byte.
	var lookingForIndex, index int
	lookingForIndex = 1

	for i := 2; i < len(em); i++ {
		equals0 := subtle.ConstantTimeByteEq(em[i], 0)
		index = subtle.ConstantTimeSelect(lookingForIndex&equals0, i, index)
		lookingForIndex = subtle.ConstantTimeSelect(equals0, 0, lookingForIndex)
	}

	valid = firstByteIsZero & secondByteIsTwo & (^lookingForIndex & 1)
	msg = em[index+1:]
	return
}

// nonZeroRandomBytes fills the given slice with non-zero random octets.
func nonZeroRandomBytes(s []byte, rand io.Reader) (err error) {
	_, err = io.ReadFull(rand, s)
	if err != nil {
		return
	}

	for i := 0; i < len(s); i++ {
		for s[i] == 0 {
			_, err = io.ReadFull(rand, s[i:i+1])
			if err != nil {
				return
			}
			// In tests, the PRNG may return all zeros so we do
			// this to break the loop.
			s[i] ^= 0x42
		}
	}

	return
}

// These are ASN1 DER structures:
//   DigestInfo ::= SEQUENCE {
//     digestAlgorithm AlgorithmIdentifier,
//     digest OCTET STRING
//   }
// For performance, we don't use the generic ASN1 encoder. Rather, we
// precompute a prefix of the digest value that makes a valid ASN1 DER string
// with the correct contents.
var hashPrefixes = map[crypto.Hash][]byte{
	crypto.MD5:       {0x30, 0x20, 0x30, 0x0c, 0x06, 0x08, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x02, 0x05, 0x05, 0x00, 0x04, 0x10},
	crypto.SHA1:      {0x30, 0x21, 0x30, 0x09, 0x06, 0x05, 0x2b, 0x0e, 0x03, 0x02, 0x1a, 0x05, 0x00, 0x04, 0x14},
	crypto.SHA256:    {0x30, 0x31, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x01, 0x05, 0x00, 0x04, 0x20},
	crypto.SHA384:    {0x30, 0x41, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x02, 0x05, 0x00, 0x04, 0x30},
	crypto.SHA512:    {0x30, 0x51, 0x30, 0x0d, 0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x03, 0x05, 0x00, 0x04, 0x40},
	crypto.MD5SHA1:   {}, // A special TLS case which doesn't use an ASN1 prefix.
	crypto.RIPEMD160: {0x30, 0x20, 0x30, 0x08, 0x06, 0x06, 0x28, 0xcf, 0x06, 0x03, 0x00, 0x31, 0x04, 0x14},
}

// SignPKCS1v15 calculates the signature of hashed using RSASSA-PKCS1-V1_5-SIGN from RSA PKCS#1 v1.5.
// Note that hashed must be the result of hashing the input message using the
// given hash function.
func SignPKCS1v15(rand io.Reader, priv *PrivateKey, hash crypto.Hash, hashed []byte) (s []byte, err error) {
	hashLen, prefix, err := pkcs1v15HashInfo(hash, len(hashed))
	if err != nil {
		return
	}

	tLen := len(prefix) + hashLen
	k := (priv.N.BitLen() + 7) / 8
	if k < tLen+11 {
		return nil, MessageTooLongError{}
	}

	// EM = 0x00 || 0x01 || PS || 0x00 || T
	em := make([]byte, k)
	em[1] = 1
	for i := 2; i < k-tLen-1; i++ {
		em[i] = 0xff
	}
	copy(em[k-tLen:k-hashLen], prefix)
	copy(em[k-hashLen:k], hashed)

	m := new(big.Int).SetBytes(em)
	c, err := decrypt(rand, priv, m)
	if err == nil {
		s = c.Bytes()
	}
	return
}

// VerifyPKCS1v15 verifies an RSA PKCS#1 v1.5 signature.
// hashed is the result of hashing the input message using the given hash
// function and sig is the signature. A valid signature is indicated by
// returning a nil error.
func VerifyPKCS1v15(pub *PublicKey, hash crypto.Hash, hashed []byte, sig []byte) (err error) {
	hashLen, prefix, err := pkcs1v15HashInfo(hash, len(hashed))
	if err != nil {
		return
	}

	tLen := len(prefix) + hashLen
	k := (pub.N.BitLen() + 7) / 8
	if k < tLen+11 {
		err = VerificationError{}
		return
	}

	c := new(big.Int).SetBytes(sig)
	m := encrypt(new(big.Int), pub, c)
	em := leftPad(m.Bytes(), k)
	// EM = 0x00 || 0x01 || PS || 0x00 || T

	ok := subtle.ConstantTimeByteEq(em[0], 0)
	ok &= subtle.ConstantTimeByteEq(em[1], 1)
	ok &= subtle.ConstantTimeCompare(em[k-hashLen:k], hashed)
	ok &= subtle.ConstantTimeCompare(em[k-tLen:k-hashLen], prefix)
	ok &= subtle.ConstantTimeByteEq(em[k-tLen-1], 0)

	for i := 2; i < k-tLen-1; i++ {
		ok &= subtle.ConstantTimeByteEq(em[i], 0xff)
	}

	if ok != 1 {
		return VerificationError{}
	}

	return nil
}

func pkcs1v15HashInfo(hash crypto.Hash, inLen int) (hashLen int, prefix []byte, err error) {
	hashLen = hash.Size()
	if inLen != hashLen {
		return 0, nil, errors.New("input must be hashed message")
	}
	prefix, ok := hashPrefixes[hash]
	if !ok {
		return 0, nil, errors.New("unsupported hash function")
	}
	return
}
