// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package rsa implements RSA encryption as specified in PKCS#1.
package rsa

// TODO(agl): Add support for PSS padding.

import (
	"crypto/rand"
	"crypto/subtle"
	"errors"
	"hash"
	"io"
	"math/big"
)

var bigZero = big.NewInt(0)
var bigOne = big.NewInt(1)

// A PublicKey represents the public part of an RSA key.
type PublicKey struct {
	N *big.Int // modulus
	E int      // public exponent
}

// A PrivateKey represents an RSA key
type PrivateKey struct {
	PublicKey            // public part.
	D         *big.Int   // private exponent
	Primes    []*big.Int // prime factors of N, has >= 2 elements.

	// Precomputed contains precomputed values that speed up private
	// operations, if available.
	Precomputed PrecomputedValues
}

type PrecomputedValues struct {
	Dp, Dq *big.Int // D mod (P-1) (or mod Q-1) 
	Qinv   *big.Int // Q^-1 mod Q

	// CRTValues is used for the 3rd and subsequent primes. Due to a
	// historical accident, the CRT for the first two primes is handled
	// differently in PKCS#1 and interoperability is sufficiently
	// important that we mirror this.
	CRTValues []CRTValue
}

// CRTValue contains the precomputed chinese remainder theorem values.
type CRTValue struct {
	Exp   *big.Int // D mod (prime-1).
	Coeff *big.Int // R·Coeff ≡ 1 mod Prime.
	R     *big.Int // product of primes prior to this (inc p and q).
}

// Validate performs basic sanity checks on the key.
// It returns nil if the key is valid, or else an error describing a problem.
func (priv *PrivateKey) Validate() error {
	// Check that the prime factors are actually prime. Note that this is
	// just a sanity check. Since the random witnesses chosen by
	// ProbablyPrime are deterministic, given the candidate number, it's
	// easy for an attack to generate composites that pass this test.
	for _, prime := range priv.Primes {
		if !prime.ProbablyPrime(20) {
			return errors.New("prime factor is composite")
		}
	}

	// Check that Πprimes == n.
	modulus := new(big.Int).Set(bigOne)
	for _, prime := range priv.Primes {
		modulus.Mul(modulus, prime)
	}
	if modulus.Cmp(priv.N) != 0 {
		return errors.New("invalid modulus")
	}
	// Check that e and totient(Πprimes) are coprime.
	totient := new(big.Int).Set(bigOne)
	for _, prime := range priv.Primes {
		pminus1 := new(big.Int).Sub(prime, bigOne)
		totient.Mul(totient, pminus1)
	}
	e := big.NewInt(int64(priv.E))
	gcd := new(big.Int)
	x := new(big.Int)
	y := new(big.Int)
	gcd.GCD(x, y, totient, e)
	if gcd.Cmp(bigOne) != 0 {
		return errors.New("invalid public exponent E")
	}
	// Check that de ≡ 1 (mod totient(Πprimes))
	de := new(big.Int).Mul(priv.D, e)
	de.Mod(de, totient)
	if de.Cmp(bigOne) != 0 {
		return errors.New("invalid private exponent D")
	}
	return nil
}

// GenerateKey generates an RSA keypair of the given bit size.
func GenerateKey(random io.Reader, bits int) (priv *PrivateKey, err error) {
	return GenerateMultiPrimeKey(random, 2, bits)
}

// GenerateMultiPrimeKey generates a multi-prime RSA keypair of the given bit
// size, as suggested in [1]. Although the public keys are compatible
// (actually, indistinguishable) from the 2-prime case, the private keys are
// not. Thus it may not be possible to export multi-prime private keys in
// certain formats or to subsequently import them into other code.
//
// Table 1 in [2] suggests maximum numbers of primes for a given size.
//
// [1] US patent 4405829 (1972, expired)
// [2] http://www.cacr.math.uwaterloo.ca/techreports/2006/cacr2006-16.pdf
func GenerateMultiPrimeKey(random io.Reader, nprimes int, bits int) (priv *PrivateKey, err error) {
	priv = new(PrivateKey)
	priv.E = 65537

	if nprimes < 2 {
		return nil, errors.New("rsa.GenerateMultiPrimeKey: nprimes must be >= 2")
	}

	primes := make([]*big.Int, nprimes)

NextSetOfPrimes:
	for {
		todo := bits
		for i := 0; i < nprimes; i++ {
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

		g := new(big.Int)
		priv.D = new(big.Int)
		y := new(big.Int)
		e := big.NewInt(int64(priv.E))
		g.GCD(priv.D, y, e, totient)

		if g.Cmp(bigOne) == 0 {
			priv.D.Add(priv.D, totient)
			priv.Primes = primes
			priv.N = n

			break
		}
	}

	priv.Precompute()
	return
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
// specified in PKCS#1 v2.1.
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

// MessageTooLongError is returned when attempting to encrypt a message which
// is too large for the size of the public key.
type MessageTooLongError struct{}

func (MessageTooLongError) Error() string {
	return "message too long for RSA public key size"
}

func encrypt(c *big.Int, pub *PublicKey, m *big.Int) *big.Int {
	e := big.NewInt(int64(pub.E))
	c.Exp(m, e, pub.N)
	return c
}

// EncryptOAEP encrypts the given message with RSA-OAEP.
// The message must be no longer than the length of the public modulus less
// twice the hash length plus 2.
func EncryptOAEP(hash hash.Hash, random io.Reader, pub *PublicKey, msg []byte, label []byte) (out []byte, err error) {
	hash.Reset()
	k := (pub.N.BitLen() + 7) / 8
	if len(msg) > k-2*hash.Size()-2 {
		err = MessageTooLongError{}
		return
	}

	hash.Write(label)
	lHash := hash.Sum(nil)
	hash.Reset()

	em := make([]byte, k)
	seed := em[1 : 1+hash.Size()]
	db := em[1+hash.Size():]

	copy(db[0:hash.Size()], lHash)
	db[len(db)-len(msg)-1] = 1
	copy(db[len(db)-len(msg):], msg)

	_, err = io.ReadFull(random, seed)
	if err != nil {
		return
	}

	mgf1XOR(db, hash, seed)
	mgf1XOR(seed, hash, db)

	m := new(big.Int)
	m.SetBytes(em)
	c := encrypt(new(big.Int), pub, m)
	out = c.Bytes()

	if len(out) < k {
		// If the output is too small, we need to left-pad with zeros.
		t := make([]byte, k)
		copy(t[k-len(out):], out)
		out = t
	}

	return
}

// A DecryptionError represents a failure to decrypt a message.
// It is deliberately vague to avoid adaptive attacks.
type DecryptionError struct{}

func (DecryptionError) Error() string { return "RSA decryption error" }

// A VerificationError represents a failure to verify a signature.
// It is deliberately vague to avoid adaptive attacks.
type VerificationError struct{}

func (VerificationError) Error() string { return "RSA verification error" }

// modInverse returns ia, the inverse of a in the multiplicative group of prime
// order n. It requires that a be a member of the group (i.e. less than n).
func modInverse(a, n *big.Int) (ia *big.Int, ok bool) {
	g := new(big.Int)
	x := new(big.Int)
	y := new(big.Int)
	g.GCD(x, y, a, n)
	if g.Cmp(bigOne) != 0 {
		// In this case, a and n aren't coprime and we cannot calculate
		// the inverse. This happens because the values of n are nearly
		// prime (being the product of two primes) rather than truly
		// prime.
		return
	}

	if x.Cmp(bigOne) < 0 {
		// 0 is not the multiplicative inverse of any element so, if x
		// < 1, then x is negative.
		x.Add(x, n)
	}

	return x, true
}

// Precompute performs some calculations that speed up private key operations
// in the future.
func (priv *PrivateKey) Precompute() {
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

// decrypt performs an RSA decryption, resulting in a plaintext integer. If a
// random source is given, RSA blinding is used.
func decrypt(random io.Reader, priv *PrivateKey, c *big.Int) (m *big.Int, err error) {
	// TODO(agl): can we get away with reusing blinds?
	if c.Cmp(priv.N) > 0 {
		err = DecryptionError{}
		return
	}

	var ir *big.Int
	if random != nil {
		// Blinding enabled. Blinding involves multiplying c by r^e.
		// Then the decryption operation performs (m^e * r^e)^d mod n
		// which equals mr mod n. The factor of r can then be removed
		// by multiplying by the multiplicative inverse of r.

		var r *big.Int

		for {
			r, err = rand.Int(random, priv.N)
			if err != nil {
				return
			}
			if r.Cmp(bigZero) == 0 {
				r = bigOne
			}
			var ok bool
			ir, ok = modInverse(r, priv.N)
			if ok {
				break
			}
		}
		bigE := big.NewInt(int64(priv.E))
		rpowe := new(big.Int).Exp(r, bigE, priv.N)
		cCopy := new(big.Int).Set(c)
		cCopy.Mul(cCopy, rpowe)
		cCopy.Mod(cCopy, priv.N)
		c = cCopy
	}

	if priv.Precomputed.Dp == nil {
		m = new(big.Int).Exp(c, priv.D, priv.N)
	} else {
		// We have the precalculated values needed for the CRT.
		m = new(big.Int).Exp(c, priv.Precomputed.Dp, priv.Primes[0])
		m2 := new(big.Int).Exp(c, priv.Precomputed.Dq, priv.Primes[1])
		m.Sub(m, m2)
		if m.Sign() < 0 {
			m.Add(m, priv.Primes[0])
		}
		m.Mul(m, priv.Precomputed.Qinv)
		m.Mod(m, priv.Primes[0])
		m.Mul(m, priv.Primes[1])
		m.Add(m, m2)

		for i, values := range priv.Precomputed.CRTValues {
			prime := priv.Primes[2+i]
			m2.Exp(c, values.Exp, prime)
			m2.Sub(m2, m)
			m2.Mul(m2, values.Coeff)
			m2.Mod(m2, prime)
			if m2.Sign() < 0 {
				m2.Add(m2, prime)
			}
			m2.Mul(m2, values.R)
			m.Add(m, m2)
		}
	}

	if ir != nil {
		// Unblind.
		m.Mul(m, ir)
		m.Mod(m, priv.N)
	}

	return
}

// DecryptOAEP decrypts ciphertext using RSA-OAEP.
// If random != nil, DecryptOAEP uses RSA blinding to avoid timing side-channel attacks.
func DecryptOAEP(hash hash.Hash, random io.Reader, priv *PrivateKey, ciphertext []byte, label []byte) (msg []byte, err error) {
	k := (priv.N.BitLen() + 7) / 8
	if len(ciphertext) > k ||
		k < hash.Size()*2+2 {
		err = DecryptionError{}
		return
	}

	c := new(big.Int).SetBytes(ciphertext)

	m, err := decrypt(random, priv, c)
	if err != nil {
		return
	}

	hash.Write(label)
	lHash := hash.Sum(nil)
	hash.Reset()

	// Converting the plaintext number to bytes will strip any
	// leading zeros so we may have to left pad. We do this unconditionally
	// to avoid leaking timing information. (Although we still probably
	// leak the number of leading zeros. It's not clear that we can do
	// anything about this.)
	em := leftPad(m.Bytes(), k)

	firstByteIsZero := subtle.ConstantTimeByteEq(em[0], 0)

	seed := em[1 : hash.Size()+1]
	db := em[hash.Size()+1:]

	mgf1XOR(seed, hash, db)
	mgf1XOR(db, hash, seed)

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
		err = DecryptionError{}
		return
	}

	msg = rest[index+1:]
	return
}

// leftPad returns a new slice of length size. The contents of input are right
// aligned in the new slice.
func leftPad(input []byte, size int) (out []byte) {
	n := len(input)
	if n > size {
		n = size
	}
	out = make([]byte, size)
	copy(out[len(out)-n:], input)
	return
}
