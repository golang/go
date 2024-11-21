// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rsa

// This file implements the RSASSA-PSS signature scheme and the RSAES-OAEP
// encryption scheme according to RFC 8017, aka PKCS #1 v2.2.

import (
	"bytes"
	"crypto/internal/fips140"
	"crypto/internal/fips140/drbg"
	"crypto/internal/fips140/sha256"
	"crypto/internal/fips140/sha3"
	"crypto/internal/fips140/sha512"
	"crypto/internal/fips140/subtle"
	"errors"
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

// incCounter increments a four byte, big-endian counter.
func incCounter(c *[4]byte) {
	if c[3]++; c[3] != 0 {
		return
	}
	if c[2]++; c[2] != 0 {
		return
	}
	if c[1]++; c[1] != 0 {
		return
	}
	c[0]++
}

// mgf1XOR XORs the bytes in out with a mask generated using the MGF1 function
// specified in PKCS #1 v2.1.
func mgf1XOR(out []byte, hash fips140.Hash, seed []byte) {
	var counter [4]byte
	var digest []byte

	done := 0
	for done < len(out) {
		hash.Reset()
		hash.Write(seed)
		hash.Write(counter[0:4])
		digest = hash.Sum(digest[:0])

		for i := 0; i < len(digest) && done < len(out); i++ {
			out[done] ^= digest[i]
			done++
		}
		incCounter(&counter)
	}
}

func emsaPSSEncode(mHash []byte, emBits int, salt []byte, hash fips140.Hash) ([]byte, error) {
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

	hash.Reset()
	hash.Write(prefix[:])
	hash.Write(mHash)
	hash.Write(salt)

	h = hash.Sum(h[:0])

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

const pssSaltLengthAutodetect = -1

func emsaPSSVerify(mHash, em []byte, emBits, sLen int, hash fips140.Hash) error {
	// See RFC 8017, Section 9.1.2.

	hLen := hash.Size()
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
	if sLen == pssSaltLengthAutodetect {
		psLen := bytes.IndexByte(db, 0x01)
		if psLen < 0 {
			return ErrVerification
		}
		sLen = len(db) - psLen - 1
	}

	// FIPS 186-5, Section 5.4(g): "the length (in bytes) of the salt (sLen)
	// shall satisfy 0 ≤ sLen ≤ hLen".
	if sLen > hLen {
		fips140.RecordNonApproved()
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
	hash.Reset()
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

// PSSMaxSaltLength returns the maximum salt length for a given public key and
// hash function.
func PSSMaxSaltLength(pub *PublicKey, hash fips140.Hash) (int, error) {
	saltLength := (pub.N.BitLen()-1+7)/8 - 2 - hash.Size()
	if saltLength < 0 {
		return 0, ErrMessageTooLong
	}
	// FIPS 186-5, Section 5.4(g): "the length (in bytes) of the salt (sLen)
	// shall satisfy 0 ≤ sLen ≤ hLen".
	if fips140.Enabled && saltLength > hash.Size() {
		return hash.Size(), nil
	}
	return saltLength, nil
}

// SignPSS calculates the signature of hashed using RSASSA-PSS.
//
// In FIPS mode, rand is ignored and can be nil.
func SignPSS(rand io.Reader, priv *PrivateKey, hash fips140.Hash, hashed []byte, saltLength int) ([]byte, error) {
	fipsSelfTest()
	fips140.RecordApproved()
	checkApprovedHash(hash)

	// Note that while we don't commit to deterministic execution with respect
	// to the rand stream, we also don't apply MaybeReadByte, so per Hyrum's Law
	// it's probably relied upon by some. It's a tolerable promise because a
	// well-specified number of random bytes is included in the signature, in a
	// well-specified way.

	if saltLength < 0 {
		return nil, errors.New("crypto/rsa: salt length cannot be negative")
	}
	// FIPS 186-5, Section 5.4(g): "the length (in bytes) of the salt (sLen)
	// shall satisfy 0 ≤ sLen ≤ hLen".
	if saltLength > hash.Size() {
		fips140.RecordNonApproved()
	}
	salt := make([]byte, saltLength)
	if fips140.Enabled {
		drbg.Read(salt)
	} else {
		if _, err := io.ReadFull(rand, salt); err != nil {
			return nil, err
		}
	}

	emBits := priv.pub.N.BitLen() - 1
	em, err := emsaPSSEncode(hashed, emBits, salt, hash)
	if err != nil {
		return nil, err
	}

	// RFC 8017: "Note that the octet length of EM will be one less than k if
	// modBits - 1 is divisible by 8 and equal to k otherwise, where k is the
	// length in octets of the RSA modulus n." 🙄
	//
	// This is extremely annoying, as all other encrypt and decrypt inputs are
	// always the exact same size as the modulus. Since it only happens for
	// weird modulus sizes, fix it by padding inefficiently.
	if emLen, k := len(em), priv.pub.Size(); emLen < k {
		emNew := make([]byte, k)
		copy(emNew[k-emLen:], em)
		em = emNew
	}

	return decrypt(priv, em, withCheck)
}

// VerifyPSS verifies sig with RSASSA-PSS automatically detecting the salt length.
func VerifyPSS(pub *PublicKey, hash fips140.Hash, digest []byte, sig []byte) error {
	return verifyPSS(pub, hash, digest, sig, pssSaltLengthAutodetect)
}

// VerifyPSS verifies sig with RSASSA-PSS and an expected salt length.
func VerifyPSSWithSaltLength(pub *PublicKey, hash fips140.Hash, digest []byte, sig []byte, saltLength int) error {
	if saltLength < 0 {
		return errors.New("crypto/rsa: salt length cannot be negative")
	}
	return verifyPSS(pub, hash, digest, sig, saltLength)
}

func verifyPSS(pub *PublicKey, hash fips140.Hash, digest []byte, sig []byte, saltLength int) error {
	fipsSelfTest()
	fips140.RecordApproved()
	checkApprovedHash(hash)
	if err := checkPublicKey(pub); err != nil {
		return err
	}

	if len(sig) != pub.Size() {
		return ErrVerification
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

	return emsaPSSVerify(digest, em, emBits, saltLength, hash)
}

func checkApprovedHash(hash fips140.Hash) {
	switch hash.(type) {
	case *sha256.Digest, *sha512.Digest, *sha3.Digest:
	default:
		fips140.RecordNonApproved()
	}
}

// EncryptOAEP encrypts the given message with RSAES-OAEP.
//
// In FIPS mode, random is ignored and can be nil.
func EncryptOAEP(hash, mgfHash fips140.Hash, random io.Reader, pub *PublicKey, msg []byte, label []byte) ([]byte, error) {
	// Note that while we don't commit to deterministic execution with respect
	// to the random stream, we also don't apply MaybeReadByte, so per Hyrum's
	// Law it's probably relied upon by some. It's a tolerable promise because a
	// well-specified number of random bytes is included in the ciphertext, in a
	// well-specified way.

	fipsSelfTest()
	fips140.RecordApproved()
	checkApprovedHash(hash)
	if err := checkPublicKey(pub); err != nil {
		return nil, err
	}
	k := pub.Size()
	if len(msg) > k-2*hash.Size()-2 {
		return nil, ErrMessageTooLong
	}

	hash.Reset()
	hash.Write(label)
	lHash := hash.Sum(nil)

	em := make([]byte, k)
	seed := em[1 : 1+hash.Size()]
	db := em[1+hash.Size():]

	copy(db[0:hash.Size()], lHash)
	db[len(db)-len(msg)-1] = 1
	copy(db[len(db)-len(msg):], msg)

	if fips140.Enabled {
		drbg.Read(seed)
	} else {
		_, err := io.ReadFull(random, seed)
		if err != nil {
			return nil, err
		}
	}

	mgf1XOR(db, mgfHash, seed)
	mgf1XOR(seed, mgfHash, db)

	return encrypt(pub, em)
}

// DecryptOAEP decrypts ciphertext using RSAES-OAEP.
func DecryptOAEP(hash, mgfHash fips140.Hash, priv *PrivateKey, ciphertext []byte, label []byte) ([]byte, error) {
	fipsSelfTest()
	fips140.RecordApproved()
	checkApprovedHash(hash)

	k := priv.pub.Size()
	if len(ciphertext) > k ||
		k < hash.Size()*2+2 {
		return nil, ErrDecryption
	}

	em, err := decrypt(priv, ciphertext, noCheck)
	if err != nil {
		return nil, err
	}

	hash.Reset()
	hash.Write(label)
	lHash := hash.Sum(nil)

	firstByteIsZero := subtle.ConstantTimeByteEq(em[0], 0)

	seed := em[1 : hash.Size()+1]
	db := em[hash.Size()+1:]

	mgf1XOR(seed, mgfHash, db)
	mgf1XOR(db, mgfHash, seed)

	lHash2 := db[0:hash.Size()]

	// We have to validate the plaintext in constant time in order to avoid
	// attacks like: J. Manger. A Chosen Ciphertext Attack on RSA Optimal
	// Asymmetric Encryption Padding (OAEP) as Standardized in PKCS #1
	// v2.0. In J. Kilian, editor, Advances in Cryptology.
	lHash2Good := subtle.ConstantTimeCompare(lHash, lHash2)

	// The remainder of the plaintext must be zero or more 0x00, followed
	// by 0x01, followed by the message.
	//   lookingForIndex: 1 iff we are still looking for the 0x01
	//   index: the offset of the first 0x01 byte
	//   invalid: 1 iff we saw a non-zero byte before the 0x01.
	var lookingForIndex, index, invalid int
	lookingForIndex = 1
	rest := db[hash.Size():]

	for i := 0; i < len(rest); i++ {
		equals0 := subtle.ConstantTimeByteEq(rest[i], 0)
		equals1 := subtle.ConstantTimeByteEq(rest[i], 1)
		index = subtle.ConstantTimeSelect(lookingForIndex&equals1, i, index)
		lookingForIndex = subtle.ConstantTimeSelect(equals1, 0, lookingForIndex)
		invalid = subtle.ConstantTimeSelect(lookingForIndex&^equals0, 1, invalid)
	}

	if firstByteIsZero&lHash2Good&^invalid&^lookingForIndex != 1 {
		return nil, ErrDecryption
	}

	return rest[index+1:], nil
}
