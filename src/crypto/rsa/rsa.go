// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package rsa implements RSA encryption as specified in PKCS #1 and RFC 8017.
//
// RSA is a single, fundamental operation that is used in this package to
// implement either public-key encryption or public-key signatures.
//
// The original specification for encryption and signatures with RSA is PKCS #1
// and the terms "RSA encryption" and "RSA signatures" by default refer to
// PKCS #1 version 1.5. However, that specification has flaws and new designs
// should use version 2, usually called by just OAEP and PSS, where
// possible.
//
// Two sets of interfaces are included in this package. When a more abstract
// interface isn't necessary, there are functions for encrypting/decrypting
// with v1.5/OAEP and signing/verifying with v1.5/PSS. If one needs to abstract
// over the public key primitive, the PrivateKey type implements the
// Decrypter and Signer interfaces from the crypto package.
//
// Operations involving private keys are implemented using constant-time
// algorithms, except for [GenerateKey], [PrivateKey.Precompute], and
// [PrivateKey.Validate].
package rsa

import (
	"crypto"
	"crypto/internal/bigmod"
	"crypto/internal/boring"
	"crypto/internal/boring/bbig"
	"crypto/internal/randutil"
	"crypto/rand"
	"crypto/subtle"
	"encoding/binary"
	"errors"
	"hash"
	"io"
	"math"
	"math/big"
)

var bigOne = big.NewInt(1)

// A PublicKey represents the public part of an RSA key.
//
// The value of the modulus N is considered secret by this library and protected
// from leaking through timing side-channels. However, neither the value of the
// exponent E nor the precise bit size of N are similarly protected.
type PublicKey struct {
	N *big.Int // modulus
	E int      // public exponent
}

// Any methods implemented on PublicKey might need to also be implemented on
// PrivateKey, as the latter embeds the former and will expose its methods.

// Size returns the modulus size in bytes. Raw signatures and ciphertexts
// for or by this public key will have the same size.
func (pub *PublicKey) Size() int {
	return (pub.N.BitLen() + 7) / 8
}

// Equal reports whether pub and x have the same value.
func (pub *PublicKey) Equal(x crypto.PublicKey) bool {
	xx, ok := x.(*PublicKey)
	if !ok {
		return false
	}
	return bigIntEqual(pub.N, xx.N) && pub.E == xx.E
}

// OAEPOptions is an interface for passing options to OAEP decryption using the
// crypto.Decrypter interface.
type OAEPOptions struct {
	// Hash is the hash function that will be used when generating the mask.
	Hash crypto.Hash

	// MGFHash is the hash function used for MGF1.
	// If zero, Hash is used instead.
	MGFHash crypto.Hash

	// Label is an arbitrary byte string that must be equal to the value
	// used when encrypting.
	Label []byte
}

var (
	errPublicModulus       = errors.New("crypto/rsa: missing public modulus")
	errPublicExponentSmall = errors.New("crypto/rsa: public exponent too small")
	errPublicExponentLarge = errors.New("crypto/rsa: public exponent too large")
)

// checkPub sanity checks the public key before we use it.
// We require pub.E to fit into a 32-bit integer so that we
// do not have different behavior depending on whether
// int is 32 or 64 bits. See also
// https://www.imperialviolet.org/2012/03/16/rsae.html.
func checkPub(pub *PublicKey) error {
	if pub.N == nil {
		return errPublicModulus
	}
	if pub.E < 2 {
		return errPublicExponentSmall
	}
	if pub.E > 1<<31-1 {
		return errPublicExponentLarge
	}
	return nil
}

// A PrivateKey represents an RSA key
type PrivateKey struct {
	PublicKey            // public part.
	D         *big.Int   // private exponent
	Primes    []*big.Int // prime factors of N, has >= 2 elements.

	// Precomputed contains precomputed values that speed up RSA operations,
	// if available. It must be generated by calling PrivateKey.Precompute and
	// must not be modified.
	Precomputed PrecomputedValues
}

// Public returns the public key corresponding to priv.
func (priv *PrivateKey) Public() crypto.PublicKey {
	return &priv.PublicKey
}

// Equal reports whether priv and x have equivalent values. It ignores
// Precomputed values.
func (priv *PrivateKey) Equal(x crypto.PrivateKey) bool {
	xx, ok := x.(*PrivateKey)
	if !ok {
		return false
	}
	if !priv.PublicKey.Equal(&xx.PublicKey) || !bigIntEqual(priv.D, xx.D) {
		return false
	}
	if len(priv.Primes) != len(xx.Primes) {
		return false
	}
	for i := range priv.Primes {
		if !bigIntEqual(priv.Primes[i], xx.Primes[i]) {
			return false
		}
	}
	return true
}

// bigIntEqual reports whether a and b are equal leaking only their bit length
// through timing side-channels.
func bigIntEqual(a, b *big.Int) bool {
	return subtle.ConstantTimeCompare(a.Bytes(), b.Bytes()) == 1
}

// Sign signs digest with priv, reading randomness from rand. If opts is a
// *[PSSOptions] then the PSS algorithm will be used, otherwise PKCS #1 v1.5 will
// be used. digest must be the result of hashing the input message using
// opts.HashFunc().
//
// This method implements [crypto.Signer], which is an interface to support keys
// where the private part is kept in, for example, a hardware module. Common
// uses should use the Sign* functions in this package directly.
func (priv *PrivateKey) Sign(rand io.Reader, digest []byte, opts crypto.SignerOpts) ([]byte, error) {
	if pssOpts, ok := opts.(*PSSOptions); ok {
		return SignPSS(rand, priv, pssOpts.Hash, digest, pssOpts)
	}

	return SignPKCS1v15(rand, priv, opts.HashFunc(), digest)
}

// Decrypt decrypts ciphertext with priv. If opts is nil or of type
// *[PKCS1v15DecryptOptions] then PKCS #1 v1.5 decryption is performed. Otherwise
// opts must have type *[OAEPOptions] and OAEP decryption is done.
func (priv *PrivateKey) Decrypt(rand io.Reader, ciphertext []byte, opts crypto.DecrypterOpts) (plaintext []byte, err error) {
	if opts == nil {
		return DecryptPKCS1v15(rand, priv, ciphertext)
	}

	switch opts := opts.(type) {
	case *OAEPOptions:
		if opts.MGFHash == 0 {
			return decryptOAEP(opts.Hash.New(), opts.Hash.New(), rand, priv, ciphertext, opts.Label)
		} else {
			return decryptOAEP(opts.Hash.New(), opts.MGFHash.New(), rand, priv, ciphertext, opts.Label)
		}

	case *PKCS1v15DecryptOptions:
		if l := opts.SessionKeyLen; l > 0 {
			plaintext = make([]byte, l)
			if _, err := io.ReadFull(rand, plaintext); err != nil {
				return nil, err
			}
			if err := DecryptPKCS1v15SessionKey(rand, priv, ciphertext, plaintext); err != nil {
				return nil, err
			}
			return plaintext, nil
		} else {
			return DecryptPKCS1v15(rand, priv, ciphertext)
		}

	default:
		return nil, errors.New("crypto/rsa: invalid options for Decrypt")
	}
}

type PrecomputedValues struct {
	Dp, Dq *big.Int // D mod (P-1) (or mod Q-1)
	Qinv   *big.Int // Q^-1 mod P

	// CRTValues is used for the 3rd and subsequent primes. Due to a
	// historical accident, the CRT for the first two primes is handled
	// differently in PKCS #1 and interoperability is sufficiently
	// important that we mirror this.
	//
	// Deprecated: These values are still filled in by Precompute for
	// backwards compatibility but are not used. Multi-prime RSA is very rare,
	// and is implemented by this package without CRT optimizations to limit
	// complexity.
	CRTValues []CRTValue

	n, p, q *bigmod.Modulus // moduli for CRT with Montgomery precomputed constants
}

// CRTValue contains the precomputed Chinese remainder theorem values.
type CRTValue struct {
	Exp   *big.Int // D mod (prime-1).
	Coeff *big.Int // R·Coeff ≡ 1 mod Prime.
	R     *big.Int // product of primes prior to this (inc p and q).
}

// Validate performs basic sanity checks on the key.
// It returns nil if the key is valid, or else an error describing a problem.
func (priv *PrivateKey) Validate() error {
	if err := checkPub(&priv.PublicKey); err != nil {
		return err
	}

	// Check that Πprimes == n.
	N, err := bigmod.NewModulusFromBig(priv.N)
	if err != nil {
		return err
	}
	bigOneNat, err := bigmod.NewNat().SetBytes(bigOne.Bytes(), N)
	if err != nil {
		return err
	}
	modulus, err := bigmod.NewNat().SetBytes(bigOne.Bytes(), N)
	if err != nil {
		return err
	}
	for _, prime := range priv.Primes {
		// Any primes ≤ 1 will cause divide-by-zero panics later.
		nprime, err := bigmod.NewNat().SetBytes(prime.Bytes(), N)
		if err != nil {
			return err
		}
		if d := nprime.Cmp(bigOneNat); d <= 0 {
			return errors.New("crypto/rsa: invalid prime value")
		}
		modulus.MulMod(nprime, N)
	}
	if modulus.Equal(N.Nat()) != 0 {
		return errors.New("crypto/rsa: invalid modulus")
	}

	// NIST SP 800-56B REV.2 6.4.1.1 3.a
	// Key-pair consistency verifying that m = (m^e)^d mod n for some integer m satisfying 1 < m < (n − 1).
	m, err := bigmod.NewNat().SetBytes([]byte{2}, N)
	ebytes := make([]byte, 8)
	binary.BigEndian.PutUint64(ebytes, uint64(priv.E))
	mm := bigmod.NewNat().Exp(m, ebytes, N)
	mm.Exp(mm, priv.D.Bytes(), N)
	if mm.Equal(m) != 1 {
		return errors.New("crypto/rsa: key-pair consistency check failed")
	}

	if len(priv.Primes) > 2 {
		// Check that de ≡ 1 mod p-1, for each prime.
		// This implies that e is coprime to each p-1 as e has a multiplicative
		// inverse. Therefore e is coprime to lcm(p-1,q-1,r-1,...) =
		// exponent(ℤ/nℤ). It also implies that a^de ≡ a mod p as a^(p-1) ≡ 1
		// mod p. Thus a^de ≡ a mod n for all a coprime to n, as required.
		congruence := new(big.Int)
		de := new(big.Int).SetInt64(int64(priv.E))
		de.Mul(de, priv.D)
		for _, prime := range priv.Primes {
			pminus1 := new(big.Int).Sub(prime, bigOne)
			congruence.Mod(de, pminus1)
			if congruence.Cmp(bigOne) != 0 {
				return errors.New("crypto/rsa: invalid exponents")
			}
		}
	} else {
		// 6.4.1.2.1.D rsakpv1-crt
		//
		// (npub == p × q)
		pBytes := priv.Primes[0].Bytes()
		qBytes := priv.Primes[1].Bytes()
		p, err := bigmod.NewNat().SetBytes(pBytes, N)
		if err != nil {
			return err
		}
		q, err := bigmod.NewNat().SetBytes(qBytes, N)
		if err != nil {
			return err
		}
		product := bigmod.NewNat().Mul(p, q, N)
		if r := product.Cmp(N.Nat()); r != 0 {
			return errors.New("crypto/rsa: invalid RSA key pair")
		}

		// 6.4.1.2.1.F rsakpv1-crt
		priv.Precompute()

		// Step a: 1 < dP < (p – 1).
		pminus1 := bigmod.NewNat().Set(p)
		pminus1.Sub(bigOneNat)
		pminus1big := new(big.Int).SetBytes(pminus1.Bytes(N))
		pminus1mod, err := bigmod.NewModulusFromBig(pminus1big)
		dP, err := bigmod.NewNat().SetBytes(priv.Precomputed.Dp.Bytes(), N)
		if err != nil {
			return err
		}
		// dP := bigmod.NewNat().Mod(d, pminus1mod)
		res1 := bigOneNat.Cmp(dP)
		res2 := dP.Cmp(pminus1)
		if res1 != -1 || res2 != -1 {
			return errors.New("crypto/rsa: (step A) invalid RSA key pair")
		}

		// Step b: 1 < dQ < (q – 1).
		dQ, err := bigmod.NewNat().SetBytes(priv.Precomputed.Dq.Bytes(), N)
		if err != nil {
			return err
		}
		qminus1 := bigmod.NewNat().Set(q)
		qminus1.Sub(bigOneNat)
		qminus1big := new(big.Int).SetBytes(qminus1.Bytes(N))
		qminus1mod, err := bigmod.NewModulusFromBig(qminus1big)
		res1 = bigOneNat.Cmp(dQ)
		res2 = dQ.Cmp(qminus1)
		if res1 != -1 || res2 != -1 {
			return errors.New("crypto/rsa: invalid RSA key pair")
		}
		// Step c: 1 < qInv < p.
		qInv, err := bigmod.NewNat().SetBytes(priv.Precomputed.Qinv.Bytes(), N)
		if err != nil {
			return err
		}
		res1 = bigOneNat.Cmp(qInv)
		res2 = qInv.Cmp(p)
		if res1 != -1 || res2 != -1 {
			return errors.New("crypto/rsa: invalid RSA key pair")
		}
		// Step d: 1 = (dP × epub) mod (p – 1).
		buf := make([]byte, 8)
		binary.BigEndian.PutUint64(buf, uint64(priv.E))
		epub, err := bigmod.NewNat().SetBytes(buf, N)
		if err != nil {
			return err
		}
		if bigmod.NewNat().Mod(
			bigmod.NewNat().Mul(dP, epub, N),
			pminus1mod).Equal(bigOneNat) != 1 {
			return errors.New("crypto/rsa: invalid RSA key pair")
		}
		// Step e: 1 = (dQ × epub) mod (q – 1).
		if bigmod.NewNat().Mod(
			bigmod.NewNat().Mul(dQ, epub, N),
			qminus1mod).Equal(bigOneNat) != 1 {
			return errors.New("crypto/rsa: invalid RSA key pair")
		}
		// Step f: 1 = (qInv × q) mod p.
		pmod, err := bigmod.NewModulusFromBig(priv.Primes[0])
		if err != nil {
			return err
		}
		if bigmod.NewNat().Mod(
			bigmod.NewNat().Mul(qInv, q, N),
			pmod).Equal(bigOneNat) != 1 {
			return errors.New("crypto/rsa: invalid RSA key pair")
		}
	}
	return nil
}

// GenerateKey generates a random RSA private key of the given bit size.
//
// Most applications should use [crypto/rand.Reader] as rand. Note that the
// returned key does not depend deterministically on the bytes read from rand,
// and may change between calls and/or between versions.
func GenerateKey(random io.Reader, bits int) (*PrivateKey, error) {
	return GenerateMultiPrimeKey(random, 2, bits)
}

// GenerateMultiPrimeKey generates a multi-prime RSA keypair of the given bit
// size and the given random source.
//
// Table 1 in "[On the Security of Multi-prime RSA]" suggests maximum numbers of
// primes for a given bit size.
//
// Although the public keys are compatible (actually, indistinguishable) from
// the 2-prime case, the private keys are not. Thus it may not be possible to
// export multi-prime private keys in certain formats or to subsequently import
// them into other code.
//
// This package does not implement CRT optimizations for multi-prime RSA, so the
// keys with more than two primes will have worse performance.
//
// Deprecated: The use of this function with a number of primes different from
// two is not recommended for the above security, compatibility, and performance
// reasons. Use [GenerateKey] instead.
//
// [On the Security of Multi-prime RSA]: http://www.cacr.math.uwaterloo.ca/techreports/2006/cacr2006-16.pdf
func GenerateMultiPrimeKey(random io.Reader, nprimes int, bits int) (*PrivateKey, error) {
	randutil.MaybeReadByte(random)

	if boring.Enabled && random == boring.RandReader && nprimes == 2 &&
		(bits == 2048 || bits == 3072 || bits == 4096) {
		bN, bE, bD, bP, bQ, bDp, bDq, bQinv, err := boring.GenerateKeyRSA(bits)
		if err != nil {
			return nil, err
		}
		N := bbig.Dec(bN)
		E := bbig.Dec(bE)
		D := bbig.Dec(bD)
		P := bbig.Dec(bP)
		Q := bbig.Dec(bQ)
		Dp := bbig.Dec(bDp)
		Dq := bbig.Dec(bDq)
		Qinv := bbig.Dec(bQinv)
		e64 := E.Int64()
		if !E.IsInt64() || int64(int(e64)) != e64 {
			return nil, errors.New("crypto/rsa: generated key exponent too large")
		}

		mn, err := bigmod.NewModulusFromBig(N)
		if err != nil {
			return nil, err
		}
		mp, err := bigmod.NewModulusFromBig(P)
		if err != nil {
			return nil, err
		}
		mq, err := bigmod.NewModulusFromBig(Q)
		if err != nil {
			return nil, err
		}

		key := &PrivateKey{
			PublicKey: PublicKey{
				N: N,
				E: int(e64),
			},
			D:      D,
			Primes: []*big.Int{P, Q},
			Precomputed: PrecomputedValues{
				Dp:        Dp,
				Dq:        Dq,
				Qinv:      Qinv,
				CRTValues: make([]CRTValue, 0), // non-nil, to match Precompute
				n:         mn,
				p:         mp,
				q:         mq,
			},
		}
		return key, nil
	}

	priv := new(PrivateKey)
	priv.E = 65537

	if nprimes < 2 {
		return nil, errors.New("crypto/rsa: GenerateMultiPrimeKey: nprimes must be >= 2")
	}

	if bits < 64 {
		primeLimit := float64(uint64(1) << uint(bits/nprimes))
		// pi approximates the number of primes less than primeLimit
		pi := primeLimit / (math.Log(primeLimit) - 1)
		// Generated primes start with 11 (in binary) so we can only
		// use a quarter of them.
		pi /= 4
		// Use a factor of two to ensure that key generation terminates
		// in a reasonable amount of time.
		pi /= 2
		if pi <= float64(nprimes) {
			return nil, errors.New("crypto/rsa: too few primes of given length to generate an RSA key")
		}
	}

	primes := make([]*big.Int, nprimes)

NextSetOfPrimes:
	for {
		todo := bits
		// crypto/rand should set the top two bits in each prime.
		// Thus each prime has the form
		//   p_i = 2^bitlen(p_i) × 0.11... (in base 2).
		// And the product is:
		//   P = 2^todo × α
		// where α is the product of nprimes numbers of the form 0.11...
		//
		// If α < 1/2 (which can happen for nprimes > 2), we need to
		// shift todo to compensate for lost bits: the mean value of 0.11...
		// is 7/8, so todo + shift - nprimes * log2(7/8) ~= bits - 1/2
		// will give good results.
		if nprimes >= 7 {
			todo += (nprimes - 2) / 5
		}
		for i := 0; i < nprimes; i++ {
			var err error
			primes[i], err = rand.Prime(random, todo/(nprimes-i))
			if err != nil {
				return nil, err
			}
			todo -= primes[i].BitLen()
		}

		// Make sure that primes is pairwise unequal.
		for i, prime := range primes {
			for j := 0; j < i; j++ {
				if prime.Cmp(primes[j]) == 0 {
					continue NextSetOfPrimes
				}
			}
		}

		n := new(big.Int).Set(bigOne)
		totient := new(big.Int).Set(bigOne)
		pminus1 := new(big.Int)
		for _, prime := range primes {
			n.Mul(n, prime)
			pminus1.Sub(prime, bigOne)
			totient.Mul(totient, pminus1)
		}
		if n.BitLen() != bits {
			// This should never happen for nprimes == 2 because
			// crypto/rand should set the top two bits in each prime.
			// For nprimes > 2 we hope it does not happen often.
			continue NextSetOfPrimes
		}

		priv.D = new(big.Int)
		e := big.NewInt(int64(priv.E))
		ok := priv.D.ModInverse(e, totient)

		if ok != nil {
			priv.Primes = primes
			priv.N = n
			break
		}
	}

	priv.Precompute()
	return priv, nil
}

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
func mgf1XOR(out []byte, hash hash.Hash, seed []byte) {
	var counter [4]byte
	var digest []byte

	done := 0
	for done < len(out) {
		hash.Write(seed)
		hash.Write(counter[0:4])
		digest = hash.Sum(digest[:0])
		hash.Reset()

		for i := 0; i < len(digest) && done < len(out); i++ {
			out[done] ^= digest[i]
			done++
		}
		incCounter(&counter)
	}
}

// ErrMessageTooLong is returned when attempting to encrypt or sign a message
// which is too large for the size of the key. When using [SignPSS], this can also
// be returned if the size of the salt is too large.
var ErrMessageTooLong = errors.New("crypto/rsa: message too long for RSA key size")

func encrypt(pub *PublicKey, plaintext []byte) ([]byte, error) {
	boring.Unreachable()

	N, err := bigmod.NewModulusFromBig(pub.N)
	if err != nil {
		return nil, err
	}
	m, err := bigmod.NewNat().SetBytes(plaintext, N)
	if err != nil {
		return nil, err
	}
	e := uint(pub.E)

	return bigmod.NewNat().ExpShortVarTime(m, e, N).Bytes(N), nil
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
	// Note that while we don't commit to deterministic execution with respect
	// to the random stream, we also don't apply MaybeReadByte, so per Hyrum's
	// Law it's probably relied upon by some. It's a tolerable promise because a
	// well-specified number of random bytes is included in the ciphertext, in a
	// well-specified way.

	if err := checkPub(pub); err != nil {
		return nil, err
	}
	hash.Reset()
	k := pub.Size()
	if len(msg) > k-2*hash.Size()-2 {
		return nil, ErrMessageTooLong
	}

	if boring.Enabled && random == boring.RandReader {
		bkey, err := boringPublicKey(pub)
		if err != nil {
			return nil, err
		}
		return boring.EncryptRSAOAEP(hash, hash, bkey, msg, label)
	}
	boring.UnreachableExceptTests()

	hash.Write(label)
	lHash := hash.Sum(nil)
	hash.Reset()

	em := make([]byte, k)
	seed := em[1 : 1+hash.Size()]
	db := em[1+hash.Size():]

	copy(db[0:hash.Size()], lHash)
	db[len(db)-len(msg)-1] = 1
	copy(db[len(db)-len(msg):], msg)

	_, err := io.ReadFull(random, seed)
	if err != nil {
		return nil, err
	}

	mgf1XOR(db, hash, seed)
	mgf1XOR(seed, hash, db)

	if boring.Enabled {
		var bkey *boring.PublicKeyRSA
		bkey, err = boringPublicKey(pub)
		if err != nil {
			return nil, err
		}
		return boring.EncryptRSANoPadding(bkey, em)
	}

	return encrypt(pub, em)
}

// ErrDecryption represents a failure to decrypt a message.
// It is deliberately vague to avoid adaptive attacks.
var ErrDecryption = errors.New("crypto/rsa: decryption error")

// ErrVerification represents a failure to verify a signature.
// It is deliberately vague to avoid adaptive attacks.
var ErrVerification = errors.New("crypto/rsa: verification error")

// Precompute performs some calculations that speed up private key operations
// in the future.
func (priv *PrivateKey) Precompute() {
	if priv.Precomputed.n == nil && len(priv.Primes) == 2 {
		// Precomputed values _should_ always be valid, but if they aren't
		// just return. We could also panic.
		var err error
		priv.Precomputed.n, err = bigmod.NewModulusFromBig(priv.N)
		if err != nil {
			return
		}
		priv.Precomputed.p, err = bigmod.NewModulusFromBig(priv.Primes[0])
		if err != nil {
			// Unset previous values, so we either have everything or nothing
			priv.Precomputed.n = nil
			return
		}
		priv.Precomputed.q, err = bigmod.NewModulusFromBig(priv.Primes[1])
		if err != nil {
			// Unset previous values, so we either have everything or nothing
			priv.Precomputed.n, priv.Precomputed.p = nil, nil
			return
		}
	}

	// Fill in the backwards-compatibility *big.Int values.
	if priv.Precomputed.Dp != nil {
		return
	}

	priv.Precomputed.Dp = new(big.Int).Sub(priv.Primes[0], bigOne)
	priv.Precomputed.Dp.Mod(priv.D, priv.Precomputed.Dp)

	priv.Precomputed.Dq = new(big.Int).Sub(priv.Primes[1], bigOne)
	priv.Precomputed.Dq.Mod(priv.D, priv.Precomputed.Dq)

	priv.Precomputed.Qinv = new(big.Int).ModInverse(priv.Primes[1], priv.Primes[0])

	r := new(big.Int).Mul(priv.Primes[0], priv.Primes[1])
	priv.Precomputed.CRTValues = make([]CRTValue, len(priv.Primes)-2)
	for i := 2; i < len(priv.Primes); i++ {
		prime := priv.Primes[i]
		values := &priv.Precomputed.CRTValues[i-2]

		values.Exp = new(big.Int).Sub(prime, bigOne)
		values.Exp.Mod(priv.D, values.Exp)

		values.R = new(big.Int).Set(r)
		values.Coeff = new(big.Int).ModInverse(r, prime)

		r.Mul(r, prime)
	}
}

const withCheck = true
const noCheck = false

// decrypt performs an RSA decryption of ciphertext into out. If check is true,
// m^e is calculated and compared with ciphertext, in order to defend against
// errors in the CRT computation.
func decrypt(priv *PrivateKey, ciphertext []byte, check bool) ([]byte, error) {
	if len(priv.Primes) <= 2 {
		boring.Unreachable()
	}

	var (
		err  error
		m, c *bigmod.Nat
		N    *bigmod.Modulus
		t0   = bigmod.NewNat()
	)
	if priv.Precomputed.n == nil {
		N, err = bigmod.NewModulusFromBig(priv.N)
		if err != nil {
			return nil, ErrDecryption
		}
		c, err = bigmod.NewNat().SetBytes(ciphertext, N)
		if err != nil {
			return nil, ErrDecryption
		}
		m = bigmod.NewNat().Exp(c, priv.D.Bytes(), N)
	} else {
		N = priv.Precomputed.n
		P, Q := priv.Precomputed.p, priv.Precomputed.q
		Qinv, err := bigmod.NewNat().SetBytes(priv.Precomputed.Qinv.Bytes(), P)
		if err != nil {
			return nil, ErrDecryption
		}
		c, err = bigmod.NewNat().SetBytes(ciphertext, N)
		if err != nil {
			return nil, ErrDecryption
		}

		// m = c ^ Dp mod p
		m = bigmod.NewNat().Exp(t0.Mod(c, P), priv.Precomputed.Dp.Bytes(), P)
		// m2 = c ^ Dq mod q
		m2 := bigmod.NewNat().Exp(t0.Mod(c, Q), priv.Precomputed.Dq.Bytes(), Q)
		// m = m - m2 mod p
		m.SubMod(t0.Mod(m2, P), P)
		// m = m * Qinv mod p
		m.MulMod(Qinv, P)
		// m = m * q mod N
		m.ExpandFor(N).MulMod(t0.Mod(Q.Nat(), N), N)
		// m = m + m2 mod N
		m.Add(m2.ExpandFor(N), N)
	}

	if check {
		c1 := bigmod.NewNat().ExpShortVarTime(m, uint(priv.E), N)
		if c1.Equal(c) != 1 {
			return nil, ErrDecryption
		}
	}

	return m.Bytes(N), nil
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
	return decryptOAEP(hash, hash, random, priv, ciphertext, label)
}

func decryptOAEP(hash, mgfHash hash.Hash, random io.Reader, priv *PrivateKey, ciphertext []byte, label []byte) ([]byte, error) {
	if err := checkPub(&priv.PublicKey); err != nil {
		return nil, err
	}
	k := priv.Size()
	if len(ciphertext) > k ||
		k < hash.Size()*2+2 {
		return nil, ErrDecryption
	}

	if boring.Enabled {
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

	em, err := decrypt(priv, ciphertext, noCheck)
	if err != nil {
		return nil, err
	}

	hash.Write(label)
	lHash := hash.Sum(nil)
	hash.Reset()

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
