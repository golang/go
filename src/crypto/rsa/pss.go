// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rsa

// This file implements the PSS signature scheme [1].
//
// [1] http://www.rsa.com/rsalabs/pkcs/files/h11300-wp-pkcs-1v2-2-rsa-cryptography-standard.pdf

import (
	"bytes"
	"crypto"
	"errors"
	"hash"
	"io"
	"math/big"
)

func emsaPSSEncode(mHash []byte, emBits int, salt []byte, hash hash.Hash) ([]byte, error) {
	// See [1], section 9.1.1
	hLen := hash.Size()
	sLen := len(salt)
	emLen := (emBits + 7) / 8

	// 1.  If the length of M is greater than the input limitation for the
	//     hash function (2^61 - 1 octets for SHA-1), output "message too
	//     long" and stop.
	//
	// 2.  Let mHash = Hash(M), an octet string of length hLen.

	if len(mHash) != hLen {
		return nil, errors.New("crypto/rsa: input must be hashed message")
	}

	// 3.  If emLen < hLen + sLen + 2, output "encoding error" and stop.

	if emLen < hLen+sLen+2 {
		return nil, errors.New("crypto/rsa: encoding error")
	}

	em := make([]byte, emLen)
	db := em[:emLen-sLen-hLen-2+1+sLen]
	h := em[emLen-sLen-hLen-2+1+sLen : emLen-1]

	// 4.  Generate a random octet string salt of length sLen; if sLen = 0,
	//     then salt is the empty string.
	//
	// 5.  Let
	//       M' = (0x)00 00 00 00 00 00 00 00 || mHash || salt;
	//
	//     M' is an octet string of length 8 + hLen + sLen with eight
	//     initial zero octets.
	//
	// 6.  Let H = Hash(M'), an octet string of length hLen.

	var prefix [8]byte

	hash.Write(prefix[:])
	hash.Write(mHash)
	hash.Write(salt)

	h = hash.Sum(h[:0])
	hash.Reset()

	// 7.  Generate an octet string PS consisting of emLen - sLen - hLen - 2
	//     zero octets.  The length of PS may be 0.
	//
	// 8.  Let DB = PS || 0x01 || salt; DB is an octet string of length
	//     emLen - hLen - 1.

	db[emLen-sLen-hLen-2] = 0x01
	copy(db[emLen-sLen-hLen-1:], salt)

	// 9.  Let dbMask = MGF(H, emLen - hLen - 1).
	//
	// 10. Let maskedDB = DB \xor dbMask.

	mgf1XOR(db, hash, h)

	// 11. Set the leftmost 8 * emLen - emBits bits of the leftmost octet in
	//     maskedDB to zero.

	db[0] &= (0xFF >> uint(8*emLen-emBits))

	// 12. Let EM = maskedDB || H || 0xbc.
	em[emLen-1] = 0xBC

	// 13. Output EM.
	return em, nil
}

func emsaPSSVerify(mHash, em []byte, emBits, sLen int, hash hash.Hash) error {
	// 1.  If the length of M is greater than the input limitation for the
	//     hash function (2^61 - 1 octets for SHA-1), output "inconsistent"
	//     and stop.
	//
	// 2.  Let mHash = Hash(M), an octet string of length hLen.
	hLen := hash.Size()
	if hLen != len(mHash) {
		return ErrVerification
	}

	// 3.  If emLen < hLen + sLen + 2, output "inconsistent" and stop.
	emLen := (emBits + 7) / 8
	if emLen < hLen+sLen+2 {
		return ErrVerification
	}

	// 4.  If the rightmost octet of EM does not have hexadecimal value
	//     0xbc, output "inconsistent" and stop.
	if em[len(em)-1] != 0xBC {
		return ErrVerification
	}

	// 5.  Let maskedDB be the leftmost emLen - hLen - 1 octets of EM, and
	//     let H be the next hLen octets.
	db := em[:emLen-hLen-1]
	h := em[emLen-hLen-1 : len(em)-1]

	// 6.  If the leftmost 8 * emLen - emBits bits of the leftmost octet in
	//     maskedDB are not all equal to zero, output "inconsistent" and
	//     stop.
	if em[0]&(0xFF<<uint(8-(8*emLen-emBits))) != 0 {
		return ErrVerification
	}

	// 7.  Let dbMask = MGF(H, emLen - hLen - 1).
	//
	// 8.  Let DB = maskedDB \xor dbMask.
	mgf1XOR(db, hash, h)

	// 9.  Set the leftmost 8 * emLen - emBits bits of the leftmost octet in DB
	//     to zero.
	db[0] &= (0xFF >> uint(8*emLen-emBits))

	if sLen == PSSSaltLengthAuto {
	FindSaltLength:
		for sLen = emLen - (hLen + 2); sLen >= 0; sLen-- {
			switch db[emLen-hLen-sLen-2] {
			case 1:
				break FindSaltLength
			case 0:
				continue
			default:
				return ErrVerification
			}
		}
		if sLen < 0 {
			return ErrVerification
		}
	} else {
		// 10. If the emLen - hLen - sLen - 2 leftmost octets of DB are not zero
		//     or if the octet at position emLen - hLen - sLen - 1 (the leftmost
		//     position is "position 1") does not have hexadecimal value 0x01,
		//     output "inconsistent" and stop.
		for _, e := range db[:emLen-hLen-sLen-2] {
			if e != 0x00 {
				return ErrVerification
			}
		}
		if db[emLen-hLen-sLen-2] != 0x01 {
			return ErrVerification
		}
	}

	// 11.  Let salt be the last sLen octets of DB.
	salt := db[len(db)-sLen:]

	// 12.  Let
	//          M' = (0x)00 00 00 00 00 00 00 00 || mHash || salt ;
	//     M' is an octet string of length 8 + hLen + sLen with eight
	//     initial zero octets.
	//
	// 13. Let H' = Hash(M'), an octet string of length hLen.
	var prefix [8]byte
	hash.Write(prefix[:])
	hash.Write(mHash)
	hash.Write(salt)

	h0 := hash.Sum(nil)

	// 14. If H = H', output "consistent." Otherwise, output "inconsistent."
	if !bytes.Equal(h0, h) {
		return ErrVerification
	}
	return nil
}

// signPSSWithSalt calculates the signature of hashed using PSS [1] with specified salt.
// Note that hashed must be the result of hashing the input message using the
// given hash function. salt is a random sequence of bytes whose length will be
// later used to verify the signature.
func signPSSWithSalt(rand io.Reader, priv *PrivateKey, hash crypto.Hash, hashed, salt []byte) (s []byte, err error) {
	nBits := priv.N.BitLen()
	em, err := emsaPSSEncode(hashed, nBits-1, salt, hash.New())
	if err != nil {
		return
	}
	m := new(big.Int).SetBytes(em)
	c, err := decrypt(rand, priv, m)
	if err != nil {
		return
	}
	s = make([]byte, (nBits+7)/8)
	copyWithLeftPad(s, c.Bytes())
	return
}

const (
	// PSSSaltLengthAuto causes the salt in a PSS signature to be as large
	// as possible when signing, and to be auto-detected when verifying.
	PSSSaltLengthAuto = 0
	// PSSSaltLengthEqualsHash causes the salt length to equal the length
	// of the hash used in the signature.
	PSSSaltLengthEqualsHash = -1
)

// PSSOptions contains options for creating and verifying PSS signatures.
type PSSOptions struct {
	// SaltLength controls the length of the salt used in the PSS
	// signature. It can either be a number of bytes, or one of the special
	// PSSSaltLength constants.
	SaltLength int

	// Hash, if not zero, overrides the hash function passed to SignPSS.
	// This is the only way to specify the hash function when using the
	// crypto.Signer interface.
	Hash crypto.Hash
}

// HashFunc returns pssOpts.Hash so that PSSOptions implements
// crypto.SignerOpts.
func (pssOpts *PSSOptions) HashFunc() crypto.Hash {
	return pssOpts.Hash
}

func (opts *PSSOptions) saltLength() int {
	if opts == nil {
		return PSSSaltLengthAuto
	}
	return opts.SaltLength
}

// SignPSS calculates the signature of hashed using RSASSA-PSS [1].
// Note that hashed must be the result of hashing the input message using the
// given hash function. The opts argument may be nil, in which case sensible
// defaults are used.
func SignPSS(rand io.Reader, priv *PrivateKey, hash crypto.Hash, hashed []byte, opts *PSSOptions) (s []byte, err error) {
	saltLength := opts.saltLength()
	switch saltLength {
	case PSSSaltLengthAuto:
		saltLength = (priv.N.BitLen()+7)/8 - 2 - hash.Size()
	case PSSSaltLengthEqualsHash:
		saltLength = hash.Size()
	}

	if opts.Hash != 0 {
		hash = opts.Hash
	}

	salt := make([]byte, saltLength)
	if _, err = io.ReadFull(rand, salt); err != nil {
		return
	}
	return signPSSWithSalt(rand, priv, hash, hashed, salt)
}

// VerifyPSS verifies a PSS signature.
// hashed is the result of hashing the input message using the given hash
// function and sig is the signature. A valid signature is indicated by
// returning a nil error. The opts argument may be nil, in which case sensible
// defaults are used.
func VerifyPSS(pub *PublicKey, hash crypto.Hash, hashed []byte, sig []byte, opts *PSSOptions) error {
	return verifyPSS(pub, hash, hashed, sig, opts.saltLength())
}

// verifyPSS verifies a PSS signature with the given salt length.
func verifyPSS(pub *PublicKey, hash crypto.Hash, hashed []byte, sig []byte, saltLen int) error {
	nBits := pub.N.BitLen()
	if len(sig) != (nBits+7)/8 {
		return ErrVerification
	}
	s := new(big.Int).SetBytes(sig)
	m := encrypt(new(big.Int), pub, s)
	emBits := nBits - 1
	emLen := (emBits + 7) / 8
	if emLen < len(m.Bytes()) {
		return ErrVerification
	}
	em := make([]byte, emLen)
	copyWithLeftPad(em, m.Bytes())
	if saltLen == PSSSaltLengthEqualsHash {
		saltLen = hash.Size()
	}
	return emsaPSSVerify(hashed, em, emBits, saltLen, hash.New())
}
