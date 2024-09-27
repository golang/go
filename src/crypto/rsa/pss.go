// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rsa

// This file implements the RSASSA-PSS signature scheme according to RFC 8017.

import (
	"bytes"
	"crypto"
	"crypto/internal/boring"
	"errors"
	"hash"
	"io"
)

// Per RFC 8017, Section 9.1
//
//     EM = MGF1 xor DB || H( 8*0x00 || mHash || salt ) || 0xbc
//
// where
//
//     DB = PS || 0x01 || salt
//
// and PS can be empty so
//
//     emLen = dbLen + hLen + 1 = psLen + sLen + hLen + 2
//

func emsaPSSEncode(mHash []byte, emBits int, salt []byte, hash hash.Hash) ([]byte, error) {
	// See RFC 8017, Section 9.1.1.

	hLen := hash.Size()
	sLen := len(salt)
	emLen := (emBits + 7) / 8

	// 1.  If the length of M is greater than the input limitation for the
	//     hash function (2^61 - 1 octets for SHA-1), output "message too
	//     long" and stop.
	//
	// 2.  Let mHash = Hash(M), an octet string of length hLen.

	if len(mHash) != hLen {
		return nil, errors.New("crypto/rsa: input must be hashed with given hash")
	}

	// 3.  If emLen < hLen + sLen + 2, output "encoding error" and stop.

	if emLen < hLen+sLen+2 {
		return nil, ErrMessageTooLong
	}

	em := make([]byte, emLen)
	psLen := emLen - sLen - hLen - 2
	db := em[:psLen+1+sLen]
	h := em[psLen+1+sLen : emLen-1]

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
	//     zero octets. The length of PS may be 0.
	//
	// 8.  Let DB = PS || 0x01 || salt; DB is an octet string of length
	//     emLen - hLen - 1.

	db[psLen] = 0x01
	copy(db[psLen+1:], salt)

	// 9.  Let dbMask = MGF(H, emLen - hLen - 1).
	//
	// 10. Let maskedDB = DB \xor dbMask.

	mgf1XOR(db, hash, h)

	// 11. Set the leftmost 8 * emLen - emBits bits of the leftmost octet in
	//     maskedDB to zero.

	db[0] &= 0xff >> (8*emLen - emBits)

	// 12. Let EM = maskedDB || H || 0xbc.
	em[emLen-1] = 0xbc

	// 13. Output EM.
	return em, nil
}

func emsaPSSVerify(mHash, em []byte, emBits, sLen int, hash hash.Hash) error {
	// See RFC 8017, Section 9.1.2.

	hLen := hash.Size()
	if sLen == PSSSaltLengthEqualsHash {
		sLen = hLen
	}
	emLen := (emBits + 7) / 8
	if emLen != len(em) {
		return errors.New("rsa: internal error: inconsistent length")
	}

	// 1.  If the length of M is greater than the input limitation for the
	//     hash function (2^61 - 1 octets for SHA-1), output "inconsistent"
	//     and stop.
	//
	// 2.  Let mHash = Hash(M), an octet string of length hLen.
	if hLen != len(mHash) {
		return ErrVerification
	}

	// 3.  If emLen < hLen + sLen + 2, output "inconsistent" and stop.
	if emLen < hLen+sLen+2 {
		return ErrVerification
	}

	// 4.  If the rightmost octet of EM does not have hexadecimal value
	//     0xbc, output "inconsistent" and stop.
	if em[emLen-1] != 0xbc {
		return ErrVerification
	}

	// 5.  Let maskedDB be the leftmost emLen - hLen - 1 octets of EM, and
	//     let H be the next hLen octets.
	db := em[:emLen-hLen-1]
	h := em[emLen-hLen-1 : emLen-1]

	// 6.  If the leftmost 8 * emLen - emBits bits of the leftmost octet in
	//     maskedDB are not all equal to zero, output "inconsistent" and
	//     stop.
	var bitMask byte = 0xff >> (8*emLen - emBits)
	if em[0] & ^bitMask != 0 {
		return ErrVerification
	}

	// 7.  Let dbMask = MGF(H, emLen - hLen - 1).
	//
	// 8.  Let DB = maskedDB \xor dbMask.
	mgf1XOR(db, hash, h)

	// 9.  Set the leftmost 8 * emLen - emBits bits of the leftmost octet in DB
	//     to zero.
	db[0] &= bitMask

	// If we don't know the salt length, look for the 0x01 delimiter.
	if sLen == PSSSaltLengthAuto {
		psLen := bytes.IndexByte(db, 0x01)
		if psLen < 0 {
			return ErrVerification
		}
		sLen = len(db) - psLen - 1
	}

	// 10. If the emLen - hLen - sLen - 2 leftmost octets of DB are not zero
	//     or if the octet at position emLen - hLen - sLen - 1 (the leftmost
	//     position is "position 1") does not have hexadecimal value 0x01,
	//     output "inconsistent" and stop.
	psLen := emLen - hLen - sLen - 2
	for _, e := range db[:psLen] {
		if e != 0x00 {
			return ErrVerification
		}
	}
	if db[psLen] != 0x01 {
		return ErrVerification
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
	if !bytes.Equal(h0, h) { // TODO: constant time?
		return ErrVerification
	}
	return nil
}

// signPSSWithSalt calculates the signature of hashed using PSS with specified salt.
// Note that hashed must be the result of hashing the input message using the
// given hash function. salt is a random sequence of bytes whose length will be
// later used to verify the signature.
func signPSSWithSalt(priv *PrivateKey, hash crypto.Hash, hashed, salt []byte) ([]byte, error) {
	emBits := priv.N.BitLen() - 1
	em, err := emsaPSSEncode(hashed, emBits, salt, hash.New())
	if err != nil {
		return nil, err
	}

	if boring.Enabled {
		bkey, err := boringPrivateKey(priv)
		if err != nil {
			return nil, err
		}
		// Note: BoringCrypto always does decrypt "withCheck".
		// (It's not just decrypt.)
		s, err := boring.DecryptRSANoPadding(bkey, em)
		if err != nil {
			return nil, err
		}
		return s, nil
	}

	// RFC 8017: "Note that the octet length of EM will be one less than k if
	// modBits - 1 is divisible by 8 and equal to k otherwise, where k is the
	// length in octets of the RSA modulus n." 🙄
	//
	// This is extremely annoying, as all other encrypt and decrypt inputs are
	// always the exact same size as the modulus. Since it only happens for
	// weird modulus sizes, fix it by padding inefficiently.
	if emLen, k := len(em), priv.Size(); emLen < k {
		emNew := make([]byte, k)
		copy(emNew[k-emLen:], em)
		em = emNew
	}

	return decrypt(priv, em, withCheck)
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
	// SaltLength controls the length of the salt used in the PSS signature. It
	// can either be a positive number of bytes, or one of the special
	// PSSSaltLength constants.
	SaltLength int

	// Hash is the hash function used to generate the message digest. If not
	// zero, it overrides the hash function passed to SignPSS. It's required
	// when using PrivateKey.Sign.
	Hash crypto.Hash
}

// HashFunc returns opts.Hash so that [PSSOptions] implements [crypto.SignerOpts].
func (opts *PSSOptions) HashFunc() crypto.Hash {
	return opts.Hash
}

func (opts *PSSOptions) saltLength() int {
	if opts == nil {
		return PSSSaltLengthAuto
	}
	return opts.SaltLength
}

var invalidSaltLenErr = errors.New("crypto/rsa: PSSOptions.SaltLength cannot be negative")

// SignPSS calculates the signature of digest using PSS.
//
// digest must be the result of hashing the input message using the given hash
// function. The opts argument may be nil, in which case sensible defaults are
// used. If opts.Hash is set, it overrides hash.
//
// The signature is randomized depending on the message, key, and salt size,
// using bytes from rand. Most applications should use [crypto/rand.Reader] as
// rand.
func SignPSS(rand io.Reader, priv *PrivateKey, hash crypto.Hash, digest []byte, opts *PSSOptions) ([]byte, error) {
	// Note that while we don't commit to deterministic execution with respect
	// to the rand stream, we also don't apply MaybeReadByte, so per Hyrum's Law
	// it's probably relied upon by some. It's a tolerable promise because a
	// well-specified number of random bytes is included in the signature, in a
	// well-specified way.

	if opts != nil && opts.Hash != 0 {
		hash = opts.Hash
	}

	if boring.Enabled && rand == boring.RandReader {
		bkey, err := boringPrivateKey(priv)
		if err != nil {
			return nil, err
		}
		return boring.SignRSAPSS(bkey, hash, digest, opts.saltLength())
	}
	boring.UnreachableExceptTests()

	saltLength := opts.saltLength()
	switch saltLength {
	case PSSSaltLengthAuto:
		saltLength = (priv.N.BitLen()-1+7)/8 - 2 - hash.Size()
		if saltLength < 0 {
			return nil, ErrMessageTooLong
		}
	case PSSSaltLengthEqualsHash:
		saltLength = hash.Size()
	default:
		// If we get here saltLength is either > 0 or < -1, in the
		// latter case we fail out.
		if saltLength <= 0 {
			return nil, invalidSaltLenErr
		}
	}
	salt := make([]byte, saltLength)
	if _, err := io.ReadFull(rand, salt); err != nil {
		return nil, err
	}
	return signPSSWithSalt(priv, hash, digest, salt)
}

// VerifyPSS verifies a PSS signature.
//
// A valid signature is indicated by returning a nil error. digest must be the
// result of hashing the input message using the given hash function. The opts
// argument may be nil, in which case sensible defaults are used. opts.Hash is
// ignored.
//
// The inputs are not considered confidential, and may leak through timing side
// channels, or if an attacker has control of part of the inputs.
func VerifyPSS(pub *PublicKey, hash crypto.Hash, digest []byte, sig []byte, opts *PSSOptions) error {
	if boring.Enabled {
		bkey, err := boringPublicKey(pub)
		if err != nil {
			return err
		}
		if err := boring.VerifyRSAPSS(bkey, hash, digest, sig, opts.saltLength()); err != nil {
			return ErrVerification
		}
		return nil
	}
	if len(sig) != pub.Size() {
		return ErrVerification
	}
	// Salt length must be either one of the special constants (-1 or 0)
	// or otherwise positive. If it is < PSSSaltLengthEqualsHash (-1)
	// we return an error.
	if opts.saltLength() < PSSSaltLengthEqualsHash {
		return invalidSaltLenErr
	}

	emBits := pub.N.BitLen() - 1
	emLen := (emBits + 7) / 8
	em, err := encrypt(pub, sig)
	if err != nil {
		return ErrVerification
	}

	// Like in signPSSWithSalt, deal with mismatches between emLen and the size
	// of the modulus. The spec would have us wire emLen into the encoding
	// function, but we'd rather always encode to the size of the modulus and
	// then strip leading zeroes if necessary. This only happens for weird
	// modulus sizes anyway.
	for len(em) > emLen && len(em) > 0 {
		if em[0] != 0 {
			return ErrVerification
		}
		em = em[1:]
	}

	return emsaPSSVerify(digest, em, emBits, opts.saltLength(), hash.New())
}
