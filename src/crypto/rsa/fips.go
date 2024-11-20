// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rsa

import (
	"crypto"
	"crypto/internal/boring"
	"crypto/internal/fips140/rsa"
	"errors"
	"hash"
	"io"
)

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
	if err := checkPublicKeySize(&priv.PublicKey); err != nil {
		return nil, err
	}

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

	k, err := fipsPrivateKey(priv)
	if err != nil {
		return nil, err
	}
	h := hash.New()

	saltLength := opts.saltLength()
	switch saltLength {
	case PSSSaltLengthAuto:
		saltLength, err = rsa.PSSMaxSaltLength(k.PublicKey(), h)
		if err != nil {
			return nil, fipsError(err)
		}
	case PSSSaltLengthEqualsHash:
		saltLength = hash.Size()
	default:
		// If we get here saltLength is either > 0 or < -1, in the
		// latter case we fail out.
		if saltLength <= 0 {
			return nil, errors.New("crypto/rsa: invalid PSS salt length")
		}
	}

	return fipsError2(rsa.SignPSS(rand, k, h, digest, saltLength))
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
	if err := checkPublicKeySize(pub); err != nil {
		return err
	}

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

	k, err := fipsPublicKey(pub)
	if err != nil {
		return err
	}

	saltLength := opts.saltLength()
	switch saltLength {
	case PSSSaltLengthAuto:
		return fipsError(rsa.VerifyPSS(k, hash.New(), digest, sig))
	case PSSSaltLengthEqualsHash:
		return fipsError(rsa.VerifyPSSWithSaltLength(k, hash.New(), digest, sig, hash.Size()))
	default:
		return fipsError(rsa.VerifyPSSWithSaltLength(k, hash.New(), digest, sig, saltLength))
	}
}

// EncryptOAEP encrypts the given message with RSA-OAEP.
//
// OAEP is parameterised by a hash function that is used as a random oracle.
// Encryption and decryption of a given message must use the same hash function
// and sha256.New() is a reasonable choice.
//
// The random parameter is used as a source of entropy to ensure that
// encrypting the same message twice doesn't result in the same ciphertext.
// Most applications should use [crypto/rand.Reader] as random.
//
// The label parameter may contain arbitrary data that will not be encrypted,
// but which gives important context to the message. For example, if a given
// public key is used to encrypt two types of messages then distinct label
// values could be used to ensure that a ciphertext for one purpose cannot be
// used for another by an attacker. If not required it can be empty.
//
// The message must be no longer than the length of the public modulus minus
// twice the hash length, minus a further 2.
func EncryptOAEP(hash hash.Hash, random io.Reader, pub *PublicKey, msg []byte, label []byte) ([]byte, error) {
	if err := checkPublicKeySize(pub); err != nil {
		return nil, err
	}

	defer hash.Reset()

	if boring.Enabled && random == boring.RandReader {
		hash.Reset()
		k := pub.Size()
		if len(msg) > k-2*hash.Size()-2 {
			return nil, ErrMessageTooLong
		}
		bkey, err := boringPublicKey(pub)
		if err != nil {
			return nil, err
		}
		return boring.EncryptRSAOAEP(hash, hash, bkey, msg, label)
	}
	boring.UnreachableExceptTests()

	k, err := fipsPublicKey(pub)
	if err != nil {
		return nil, err
	}
	return fipsError2(rsa.EncryptOAEP(hash, hash, random, k, msg, label))
}

// DecryptOAEP decrypts ciphertext using RSA-OAEP.
//
// OAEP is parameterised by a hash function that is used as a random oracle.
// Encryption and decryption of a given message must use the same hash function
// and sha256.New() is a reasonable choice.
//
// The random parameter is legacy and ignored, and it can be nil.
//
// The label parameter must match the value given when encrypting. See
// [EncryptOAEP] for details.
func DecryptOAEP(hash hash.Hash, random io.Reader, priv *PrivateKey, ciphertext []byte, label []byte) ([]byte, error) {
	defer hash.Reset()
	return decryptOAEP(hash, hash, priv, ciphertext, label)
}

func decryptOAEP(hash, mgfHash hash.Hash, priv *PrivateKey, ciphertext []byte, label []byte) ([]byte, error) {
	if err := checkPublicKeySize(&priv.PublicKey); err != nil {
		return nil, err
	}

	if boring.Enabled {
		k := priv.Size()
		if len(ciphertext) > k ||
			k < hash.Size()*2+2 {
			return nil, ErrDecryption
		}
		bkey, err := boringPrivateKey(priv)
		if err != nil {
			return nil, err
		}
		out, err := boring.DecryptRSAOAEP(hash, mgfHash, bkey, ciphertext, label)
		if err != nil {
			return nil, ErrDecryption
		}
		return out, nil
	}

	k, err := fipsPrivateKey(priv)
	if err != nil {
		return nil, err
	}

	return fipsError2(rsa.DecryptOAEP(hash, mgfHash, k, ciphertext, label))
}

// SignPKCS1v15 calculates the signature of hashed using
// RSASSA-PKCS1-V1_5-SIGN from RSA PKCS #1 v1.5.  Note that hashed must
// be the result of hashing the input message using the given hash
// function. If hash is zero, hashed is signed directly. This isn't
// advisable except for interoperability.
//
// The random parameter is legacy and ignored, and it can be nil.
//
// This function is deterministic. Thus, if the set of possible
// messages is small, an attacker may be able to build a map from
// messages to signatures and identify the signed messages. As ever,
// signatures provide authenticity, not confidentiality.
func SignPKCS1v15(random io.Reader, priv *PrivateKey, hash crypto.Hash, hashed []byte) ([]byte, error) {
	if err := checkPublicKeySize(&priv.PublicKey); err != nil {
		return nil, err
	}

	if boring.Enabled {
		bkey, err := boringPrivateKey(priv)
		if err != nil {
			return nil, err
		}
		return boring.SignRSAPKCS1v15(bkey, hash, hashed)
	}

	k, err := fipsPrivateKey(priv)
	if err != nil {
		return nil, err
	}
	var hashName string
	if hash != crypto.Hash(0) {
		if len(hashed) != hash.Size() {
			return nil, errors.New("crypto/rsa: input must be hashed message")
		}
		hashName = hash.String()
	}
	return fipsError2(rsa.SignPKCS1v15(k, hashName, hashed))
}

// VerifyPKCS1v15 verifies an RSA PKCS #1 v1.5 signature.
// hashed is the result of hashing the input message using the given hash
// function and sig is the signature. A valid signature is indicated by
// returning a nil error. If hash is zero then hashed is used directly. This
// isn't advisable except for interoperability.
//
// The inputs are not considered confidential, and may leak through timing side
// channels, or if an attacker has control of part of the inputs.
func VerifyPKCS1v15(pub *PublicKey, hash crypto.Hash, hashed []byte, sig []byte) error {
	if err := checkPublicKeySize(pub); err != nil {
		return err
	}

	if boring.Enabled {
		bkey, err := boringPublicKey(pub)
		if err != nil {
			return err
		}
		if err := boring.VerifyRSAPKCS1v15(bkey, hash, hashed, sig); err != nil {
			return ErrVerification
		}
		return nil
	}

	k, err := fipsPublicKey(pub)
	if err != nil {
		return err
	}
	var hashName string
	if hash != crypto.Hash(0) {
		if len(hashed) != hash.Size() {
			return errors.New("crypto/rsa: input must be hashed message")
		}
		hashName = hash.String()
	}
	return fipsError(rsa.VerifyPKCS1v15(k, hashName, hashed, sig))
}

func fipsError(err error) error {
	switch err {
	case rsa.ErrDecryption:
		return ErrDecryption
	case rsa.ErrVerification:
		return ErrVerification
	case rsa.ErrMessageTooLong:
		return ErrMessageTooLong
	}
	return err
}

func fipsError2[T any](x T, err error) (T, error) {
	return x, fipsError(err)
}
