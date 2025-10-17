// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package crypto collects common cryptographic constants.
package crypto

import (
	"hash"
	"io"
	"strconv"
)

// Hash identifies a cryptographic hash function that is implemented in another
// package.
type Hash uint

// HashFunc simply returns the value of h so that [Hash] implements [SignerOpts].
func (h Hash) HashFunc() Hash {
	return h
}

func (h Hash) String() string {
	switch h {
	case MD4:
		return "MD4"
	case MD5:
		return "MD5"
	case SHA1:
		return "SHA-1"
	case SHA224:
		return "SHA-224"
	case SHA256:
		return "SHA-256"
	case SHA384:
		return "SHA-384"
	case SHA512:
		return "SHA-512"
	case MD5SHA1:
		return "MD5+SHA1"
	case RIPEMD160:
		return "RIPEMD-160"
	case SHA3_224:
		return "SHA3-224"
	case SHA3_256:
		return "SHA3-256"
	case SHA3_384:
		return "SHA3-384"
	case SHA3_512:
		return "SHA3-512"
	case SHA512_224:
		return "SHA-512/224"
	case SHA512_256:
		return "SHA-512/256"
	case BLAKE2s_256:
		return "BLAKE2s-256"
	case BLAKE2b_256:
		return "BLAKE2b-256"
	case BLAKE2b_384:
		return "BLAKE2b-384"
	case BLAKE2b_512:
		return "BLAKE2b-512"
	default:
		return "unknown hash value " + strconv.Itoa(int(h))
	}
}

const (
	MD4         Hash = 1 + iota // import golang.org/x/crypto/md4
	MD5                         // import crypto/md5
	SHA1                        // import crypto/sha1
	SHA224                      // import crypto/sha256
	SHA256                      // import crypto/sha256
	SHA384                      // import crypto/sha512
	SHA512                      // import crypto/sha512
	MD5SHA1                     // no implementation; MD5+SHA1 used for TLS RSA
	RIPEMD160                   // import golang.org/x/crypto/ripemd160
	SHA3_224                    // import crypto/sha3
	SHA3_256                    // import crypto/sha3
	SHA3_384                    // import crypto/sha3
	SHA3_512                    // import crypto/sha3
	SHA512_224                  // import crypto/sha512
	SHA512_256                  // import crypto/sha512
	BLAKE2s_256                 // import golang.org/x/crypto/blake2s
	BLAKE2b_256                 // import golang.org/x/crypto/blake2b
	BLAKE2b_384                 // import golang.org/x/crypto/blake2b
	BLAKE2b_512                 // import golang.org/x/crypto/blake2b
	maxHash
)

var digestSizes = []uint8{
	MD4:         16,
	MD5:         16,
	SHA1:        20,
	SHA224:      28,
	SHA256:      32,
	SHA384:      48,
	SHA512:      64,
	SHA512_224:  28,
	SHA512_256:  32,
	SHA3_224:    28,
	SHA3_256:    32,
	SHA3_384:    48,
	SHA3_512:    64,
	MD5SHA1:     36,
	RIPEMD160:   20,
	BLAKE2s_256: 32,
	BLAKE2b_256: 32,
	BLAKE2b_384: 48,
	BLAKE2b_512: 64,
}

// Size returns the length, in bytes, of a digest resulting from the given hash
// function. It doesn't require that the hash function in question be linked
// into the program.
func (h Hash) Size() int {
	if h > 0 && h < maxHash {
		return int(digestSizes[h])
	}
	panic("crypto: Size of unknown hash function")
}

var hashes = make([]func() hash.Hash, maxHash)

// New returns a new hash.Hash calculating the given hash function. New panics
// if the hash function is not linked into the binary.
func (h Hash) New() hash.Hash {
	if h > 0 && h < maxHash {
		f := hashes[h]
		if f != nil {
			return f()
		}
	}
	panic("crypto: requested hash function #" + strconv.Itoa(int(h)) + " is unavailable")
}

// Available reports whether the given hash function is linked into the binary.
func (h Hash) Available() bool {
	return h < maxHash && hashes[h] != nil
}

// RegisterHash registers a function that returns a new instance of the given
// hash function. This is intended to be called from the init function in
// packages that implement hash functions.
func RegisterHash(h Hash, f func() hash.Hash) {
	if h >= maxHash {
		panic("crypto: RegisterHash of unknown hash function")
	}
	hashes[h] = f
}

// PublicKey represents a public key using an unspecified algorithm.
//
// Although this type is an empty interface for backwards compatibility reasons,
// all public key types in the standard library implement the following interface
//
//	interface{
//	    Equal(x crypto.PublicKey) bool
//	}
//
// which can be used for increased type safety within applications.
type PublicKey any

// PrivateKey represents a private key using an unspecified algorithm.
//
// Although this type is an empty interface for backwards compatibility reasons,
// all private key types in the standard library implement the following interface
//
//	interface{
//	    Public() crypto.PublicKey
//	    Equal(x crypto.PrivateKey) bool
//	}
//
// as well as purpose-specific interfaces such as [Signer] and [Decrypter], which
// can be used for increased type safety within applications.
type PrivateKey any

// Signer is an interface for an opaque private key that can be used for
// signing operations. For example, an RSA key kept in a hardware module.
type Signer interface {
	// Public returns the public key corresponding to the opaque,
	// private key.
	Public() PublicKey

	// Sign signs digest with the private key, possibly using entropy from
	// rand. For an RSA key, the resulting signature should be either a
	// PKCS #1 v1.5 or PSS signature (as indicated by opts). For an (EC)DSA
	// key, it should be a DER-serialised, ASN.1 signature structure.
	//
	// Hash implements the SignerOpts interface and, in most cases, one can
	// simply pass in the hash function used as opts. Sign may also attempt
	// to type assert opts to other types in order to obtain algorithm
	// specific values. See the documentation in each package for details.
	//
	// Note that when a signature of a hash of a larger message is needed,
	// the caller is responsible for hashing the larger message and passing
	// the hash (as digest) and the hash function (as opts) to Sign.
	Sign(rand io.Reader, digest []byte, opts SignerOpts) (signature []byte, err error)
}

// MessageSigner is an interface for an opaque private key that can be used for
// signing operations where the message is not pre-hashed by the caller.
// It is a superset of the Signer interface so that it can be passed to APIs
// which accept Signer, which may try to do an interface upgrade.
//
// MessageSigner.SignMessage and MessageSigner.Sign should produce the same
// result given the same opts. In particular, MessageSigner.SignMessage should
// only accept a zero opts.HashFunc if the Signer would also accept messages
// which are not pre-hashed.
//
// Implementations which do not provide the pre-hashed Sign API should implement
// Signer.Sign by always returning an error.
type MessageSigner interface {
	Signer
	SignMessage(rand io.Reader, msg []byte, opts SignerOpts) (signature []byte, err error)
}

// SignerOpts contains options for signing with a [Signer].
type SignerOpts interface {
	// HashFunc returns an identifier for the hash function used to produce
	// the message passed to Signer.Sign, or else zero to indicate that no
	// hashing was done.
	HashFunc() Hash
}

// Decrypter is an interface for an opaque private key that can be used for
// asymmetric decryption operations. An example would be an RSA key
// kept in a hardware module.
type Decrypter interface {
	// Public returns the public key corresponding to the opaque,
	// private key.
	Public() PublicKey

	// Decrypt decrypts msg. The opts argument should be appropriate for
	// the primitive used. See the documentation in each implementation for
	// details.
	Decrypt(rand io.Reader, msg []byte, opts DecrypterOpts) (plaintext []byte, err error)
}

type DecrypterOpts any

// SignMessage signs msg with signer. If signer implements [MessageSigner],
// [MessageSigner.SignMessage] is called directly. Otherwise, msg is hashed
// with opts.HashFunc() and signed with [Signer.Sign].
func SignMessage(signer Signer, rand io.Reader, msg []byte, opts SignerOpts) (signature []byte, err error) {
	if ms, ok := signer.(MessageSigner); ok {
		return ms.SignMessage(rand, msg, opts)
	}
	if opts.HashFunc() != 0 {
		h := opts.HashFunc().New()
		h.Write(msg)
		msg = h.Sum(nil)
	}
	return signer.Sign(rand, msg, opts)
}
