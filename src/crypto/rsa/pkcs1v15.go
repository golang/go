// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rsa

import (
	"crypto/internal/boring"
	"crypto/internal/fips140/rsa"
	"crypto/internal/fips140only"
	"crypto/internal/rand"
	"crypto/subtle"
	"errors"
	"io"
)

// This file implements encryption and decryption using PKCS #1 v1.5 padding.

// PKCS1v15DecryptOptions is for passing options to PKCS #1 v1.5 decryption using
// the [crypto.Decrypter] interface.
//
// Deprecated: PKCS #1 v1.5 encryption is dangerous and should not be used.
// See [draft-irtf-cfrg-rsa-guidance-05] for more information. Use
// [EncryptOAEP] and [DecryptOAEP] instead.
//
// [draft-irtf-cfrg-rsa-guidance-05]: https://www.ietf.org/archive/id/draft-irtf-cfrg-rsa-guidance-05.html#name-rationale
type PKCS1v15DecryptOptions struct {
	// SessionKeyLen is the length of the session key that is being
	// decrypted. If not zero, then a padding error during decryption will
	// cause a random plaintext of this length to be returned rather than
	// an error. These alternatives happen in constant time.
	SessionKeyLen int
}

// EncryptPKCS1v15 encrypts the given message with RSA and the padding
// scheme from PKCS #1 v1.5.  The message must be no longer than the
// length of the public modulus minus 11 bytes.
//
// The random parameter is used as a source of entropy to ensure that encrypting
// the same message twice doesn't result in the same ciphertext. Since Go 1.26,
// a secure source of random bytes is always used, and the Reader is ignored
// unless GODEBUG=cryptocustomrand=1 is set. This setting will be removed in a
// future Go release. Instead, use [testing/cryptotest.SetGlobalRandom].
//
// Deprecated: PKCS #1 v1.5 encryption is dangerous and should not be used.
// See [draft-irtf-cfrg-rsa-guidance-05] for more information. Use
// [EncryptOAEP] and [DecryptOAEP] instead.
//
// [draft-irtf-cfrg-rsa-guidance-05]: https://www.ietf.org/archive/id/draft-irtf-cfrg-rsa-guidance-05.html#name-rationale
func EncryptPKCS1v15(random io.Reader, pub *PublicKey, msg []byte) ([]byte, error) {
	if fips140only.Enforced() {
		return nil, errors.New("crypto/rsa: use of PKCS#1 v1.5 encryption is not allowed in FIPS 140-only mode")
	}

	if err := checkPublicKeySize(pub); err != nil {
		return nil, err
	}

	k := pub.Size()
	if len(msg) > k-11 {
		return nil, ErrMessageTooLong
	}

	if boring.Enabled && rand.IsDefaultReader(random) {
		bkey, err := boringPublicKey(pub)
		if err != nil {
			return nil, err
		}
		return boring.EncryptRSAPKCS1(bkey, msg)
	}
	boring.UnreachableExceptTests()

	random = rand.CustomReader(random)

	// EM = 0x00 || 0x02 || PS || 0x00 || M
	em := make([]byte, k)
	em[1] = 2
	ps, mm := em[2:len(em)-len(msg)-1], em[len(em)-len(msg):]
	err := nonZeroRandomBytes(ps, random)
	if err != nil {
		return nil, err
	}
	em[len(em)-len(msg)-1] = 0
	copy(mm, msg)

	if boring.Enabled {
		var bkey *boring.PublicKeyRSA
		bkey, err = boringPublicKey(pub)
		if err != nil {
			return nil, err
		}
		return boring.EncryptRSANoPadding(bkey, em)
	}

	fk, err := fipsPublicKey(pub)
	if err != nil {
		return nil, err
	}
	return rsa.Encrypt(fk, em)
}

// DecryptPKCS1v15 decrypts a plaintext using RSA and the padding scheme from
// PKCS #1 v1.5. The random parameter is legacy and ignored, and it can be nil.
//
// Deprecated: PKCS #1 v1.5 encryption is dangerous and should not be used.
// Whether this function returns an error or not discloses secret information.
// If an attacker can cause this function to run repeatedly and learn whether
// each instance returned an error then they can decrypt and forge signatures as
// if they had the private key. See [draft-irtf-cfrg-rsa-guidance-05] for more
// information. Use [EncryptOAEP] and [DecryptOAEP] instead.
//
// [draft-irtf-cfrg-rsa-guidance-05]: https://www.ietf.org/archive/id/draft-irtf-cfrg-rsa-guidance-05.html#name-rationale
func DecryptPKCS1v15(random io.Reader, priv *PrivateKey, ciphertext []byte) ([]byte, error) {
	if err := checkPublicKeySize(&priv.PublicKey); err != nil {
		return nil, err
	}

	if boring.Enabled {
		bkey, err := boringPrivateKey(priv)
		if err != nil {
			return nil, err
		}
		out, err := boring.DecryptRSAPKCS1(bkey, ciphertext)
		if err != nil {
			return nil, ErrDecryption
		}
		return out, nil
	}

	valid, out, index, err := decryptPKCS1v15(priv, ciphertext)
	if err != nil {
		return nil, err
	}
	if valid == 0 {
		return nil, ErrDecryption
	}
	return out[index:], nil
}

// DecryptPKCS1v15SessionKey decrypts a session key using RSA and the padding
// scheme from PKCS #1 v1.5. The random parameter is legacy and ignored, and it
// can be nil.
//
// DecryptPKCS1v15SessionKey returns an error if the ciphertext is the wrong
// length or if the ciphertext is greater than the public modulus. Otherwise, no
// error is returned. If the padding is valid, the resulting plaintext message
// is copied into key. Otherwise, key is unchanged. These alternatives occur in
// constant time. It is intended that the user of this function generate a
// random session key beforehand and continue the protocol with the resulting
// value.
//
// Note that if the session key is too small then it may be possible for an
// attacker to brute-force it. If they can do that then they can learn whether a
// random value was used (because it'll be different for the same ciphertext)
// and thus whether the padding was correct. This also defeats the point of this
// function. Using at least a 16-byte key will protect against this attack.
//
// This method implements protections against Bleichenbacher chosen ciphertext
// attacks [0] described in RFC 3218 Section 2.3.2 [1]. While these protections
// make a Bleichenbacher attack significantly more difficult, the protections
// are only effective if the rest of the protocol which uses
// DecryptPKCS1v15SessionKey is designed with these considerations in mind. In
// particular, if any subsequent operations which use the decrypted session key
// leak any information about the key (e.g. whether it is a static or random
// key) then the mitigations are defeated. This method must be used extremely
// carefully, and typically should only be used when absolutely necessary for
// compatibility with an existing protocol (such as TLS) that is designed with
// these properties in mind.
//
//   - [0] “Chosen Ciphertext Attacks Against Protocols Based on the RSA Encryption
//     Standard PKCS #1”, Daniel Bleichenbacher, Advances in Cryptology (Crypto '98)
//   - [1] RFC 3218, Preventing the Million Message Attack on CMS,
//     https://www.rfc-editor.org/rfc/rfc3218.html
//
// Deprecated: PKCS #1 v1.5 encryption is dangerous and should not be used. The
// protections implemented by this function are limited and fragile, as
// explained above. See [draft-irtf-cfrg-rsa-guidance-05] for more information.
// Use [EncryptOAEP] and [DecryptOAEP] instead.
//
// [draft-irtf-cfrg-rsa-guidance-05]: https://www.ietf.org/archive/id/draft-irtf-cfrg-rsa-guidance-05.html#name-rationale
func DecryptPKCS1v15SessionKey(random io.Reader, priv *PrivateKey, ciphertext []byte, key []byte) error {
	if err := checkPublicKeySize(&priv.PublicKey); err != nil {
		return err
	}

	k := priv.Size()
	if k-(len(key)+3+8) < 0 {
		return ErrDecryption
	}

	valid, em, index, err := decryptPKCS1v15(priv, ciphertext)
	if err != nil {
		return err
	}

	if len(em) != k {
		// This should be impossible because decryptPKCS1v15 always
		// returns the full slice.
		return ErrDecryption
	}

	valid &= subtle.ConstantTimeEq(int32(len(em)-index), int32(len(key)))
	subtle.ConstantTimeCopy(valid, key, em[len(em)-len(key):])
	return nil
}

// decryptPKCS1v15 decrypts ciphertext using priv. It returns one or zero in
// valid that indicates whether the plaintext was correctly structured.
// In either case, the plaintext is returned in em so that it may be read
// independently of whether it was valid in order to maintain constant memory
// access patterns. If the plaintext was valid then index contains the index of
// the original message in em, to allow constant time padding removal.
func decryptPKCS1v15(priv *PrivateKey, ciphertext []byte) (valid int, em []byte, index int, err error) {
	if fips140only.Enforced() {
		return 0, nil, 0, errors.New("crypto/rsa: use of PKCS#1 v1.5 encryption is not allowed in FIPS 140-only mode")
	}

	k := priv.Size()
	if k < 11 {
		err = ErrDecryption
		return 0, nil, 0, err
	}

	if boring.Enabled {
		var bkey *boring.PrivateKeyRSA
		bkey, err = boringPrivateKey(priv)
		if err != nil {
			return 0, nil, 0, err
		}
		em, err = boring.DecryptRSANoPadding(bkey, ciphertext)
		if err != nil {
			return 0, nil, 0, ErrDecryption
		}
	} else {
		fk, err := fipsPrivateKey(priv)
		if err != nil {
			return 0, nil, 0, err
		}
		em, err = rsa.DecryptWithoutCheck(fk, ciphertext)
		if err != nil {
			return 0, nil, 0, ErrDecryption
		}
	}

	firstByteIsZero := subtle.ConstantTimeByteEq(em[0], 0)
	secondByteIsTwo := subtle.ConstantTimeByteEq(em[1], 2)

	// The remainder of the plaintext must be a string of non-zero random
	// octets, followed by a 0, followed by the message.
	//   lookingForIndex: 1 iff we are still looking for the zero.
	//   index: the offset of the first zero byte.
	lookingForIndex := 1

	for i := 2; i < len(em); i++ {
		equals0 := subtle.ConstantTimeByteEq(em[i], 0)
		index = subtle.ConstantTimeSelect(lookingForIndex&equals0, i, index)
		lookingForIndex = subtle.ConstantTimeSelect(equals0, 0, lookingForIndex)
	}

	// The PS padding must be at least 8 bytes long, and it starts two
	// bytes into em.
	validPS := subtle.ConstantTimeLessOrEq(2+8, index)

	valid = firstByteIsZero & secondByteIsTwo & (^lookingForIndex & 1) & validPS
	index = subtle.ConstantTimeSelect(valid, index+1, 0)
	return valid, em, index, nil
}

// nonZeroRandomBytes fills the given slice with non-zero random octets.
func nonZeroRandomBytes(s []byte, random io.Reader) (err error) {
	_, err = io.ReadFull(random, s)
	if err != nil {
		return
	}

	for i := 0; i < len(s); i++ {
		for s[i] == 0 {
			_, err = io.ReadFull(random, s[i:i+1])
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
