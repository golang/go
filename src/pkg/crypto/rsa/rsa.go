// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This package implements RSA encryption as specified in PKCS#1.
package rsa

// TODO(agl): Add support for PSS padding.

import (
	"big"
	"crypto/subtle"
	"hash"
	"io"
	"os"
)

var bigZero = big.NewInt(0)
var bigOne = big.NewInt(1)

// randomPrime returns a number, p, of the given size, such that p is prime
// with high probability.
func randomPrime(rand io.Reader, bits int) (p *big.Int, err os.Error) {
	if bits < 1 {
		err = os.EINVAL
	}

	bytes := make([]byte, (bits+7)/8)
	p = new(big.Int)

	for {
		_, err = io.ReadFull(rand, bytes)
		if err != nil {
			return
		}

		// Don't let the value be too small.
		bytes[0] |= 0x80
		// Make the value odd since an even number this large certainly isn't prime.
		bytes[len(bytes)-1] |= 1

		p.SetBytes(bytes)
		if big.ProbablyPrime(p, 20) {
			return
		}
	}

	return
}

// randomNumber returns a uniform random value in [0, max).
func randomNumber(rand io.Reader, max *big.Int) (n *big.Int, err os.Error) {
	k := (max.BitLen() + 7) / 8

	// r is the number of bits in the used in the most significant byte of
	// max.
	r := uint(max.BitLen() % 8)
	if r == 0 {
		r = 8
	}

	bytes := make([]byte, k)
	n = new(big.Int)

	for {
		_, err = io.ReadFull(rand, bytes)
		if err != nil {
			return
		}

		// Clear bits in the first byte to increase the probability
		// that the candidate is < max.
		bytes[0] &= uint8(int(1<<r) - 1)

		n.SetBytes(bytes)
		if n.Cmp(max) < 0 {
			return
		}
	}

	return
}

// A PublicKey represents the public part of an RSA key.
type PublicKey struct {
	N *big.Int // modulus
	E int      // public exponent
}

// A PrivateKey represents an RSA key
type PrivateKey struct {
	PublicKey          // public part.
	D         *big.Int // private exponent
	P, Q      *big.Int // prime factors of N
}

// Validate performs basic sanity checks on the key.
// It returns nil if the key is valid, or else an os.Error describing a problem.

func (priv PrivateKey) Validate() os.Error {
	// Check that p and q are prime. Note that this is just a sanity
	// check. Since the random witnesses chosen by ProbablyPrime are
	// deterministic, given the candidate number, it's easy for an attack
	// to generate composites that pass this test.
	if !big.ProbablyPrime(priv.P, 20) {
		return os.ErrorString("P is composite")
	}
	if !big.ProbablyPrime(priv.Q, 20) {
		return os.ErrorString("Q is composite")
	}

	// Check that p*q == n.
	modulus := new(big.Int).Mul(priv.P, priv.Q)
	if modulus.Cmp(priv.N) != 0 {
		return os.ErrorString("invalid modulus")
	}
	// Check that e and totient(p, q) are coprime.
	pminus1 := new(big.Int).Sub(priv.P, bigOne)
	qminus1 := new(big.Int).Sub(priv.Q, bigOne)
	totient := new(big.Int).Mul(pminus1, qminus1)
	e := big.NewInt(int64(priv.E))
	gcd := new(big.Int)
	x := new(big.Int)
	y := new(big.Int)
	big.GcdInt(gcd, x, y, totient, e)
	if gcd.Cmp(bigOne) != 0 {
		return os.ErrorString("invalid public exponent E")
	}
	// Check that de ≡ 1 (mod totient(p, q))
	de := new(big.Int).Mul(priv.D, e)
	de.Mod(de, totient)
	if de.Cmp(bigOne) != 0 {
		return os.ErrorString("invalid private exponent D")
	}
	return nil
}

// GenerateKeyPair generates an RSA keypair of the given bit size.
func GenerateKey(rand io.Reader, bits int) (priv *PrivateKey, err os.Error) {
	priv = new(PrivateKey)
	// Smaller public exponents lead to faster public key
	// operations. Since the exponent must be coprime to
	// (p-1)(q-1), the smallest possible value is 3. Some have
	// suggested that a larger exponent (often 2**16+1) be used
	// since previous implementation bugs[1] were avoided when this
	// was the case. However, there are no current reasons not to use
	// small exponents.
	// [1] http://marc.info/?l=cryptography&m=115694833312008&w=2
	priv.E = 3

	pminus1 := new(big.Int)
	qminus1 := new(big.Int)
	totient := new(big.Int)

	for {
		p, err := randomPrime(rand, bits/2)
		if err != nil {
			return nil, err
		}

		q, err := randomPrime(rand, bits/2)
		if err != nil {
			return nil, err
		}

		if p.Cmp(q) == 0 {
			continue
		}

		n := new(big.Int).Mul(p, q)
		pminus1.Sub(p, bigOne)
		qminus1.Sub(q, bigOne)
		totient.Mul(pminus1, qminus1)

		g := new(big.Int)
		priv.D = new(big.Int)
		y := new(big.Int)
		e := big.NewInt(int64(priv.E))
		big.GcdInt(g, priv.D, y, e, totient)

		if g.Cmp(bigOne) == 0 {
			priv.D.Add(priv.D, totient)
			priv.P = p
			priv.Q = q
			priv.N = n

			break
		}
	}

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

	done := 0
	for done < len(out) {
		hash.Write(seed)
		hash.Write(counter[0:4])
		digest := hash.Sum()
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

func (MessageTooLongError) String() string {
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
func EncryptOAEP(hash hash.Hash, rand io.Reader, pub *PublicKey, msg []byte, label []byte) (out []byte, err os.Error) {
	hash.Reset()
	k := (pub.N.BitLen() + 7) / 8
	if len(msg) > k-2*hash.Size()-2 {
		err = MessageTooLongError{}
		return
	}

	hash.Write(label)
	lHash := hash.Sum()
	hash.Reset()

	em := make([]byte, k)
	seed := em[1 : 1+hash.Size()]
	db := em[1+hash.Size():]

	copy(db[0:hash.Size()], lHash)
	db[len(db)-len(msg)-1] = 1
	copy(db[len(db)-len(msg):], msg)

	_, err = io.ReadFull(rand, seed)
	if err != nil {
		return
	}

	mgf1XOR(db, hash, seed)
	mgf1XOR(seed, hash, db)

	m := new(big.Int)
	m.SetBytes(em)
	c := encrypt(new(big.Int), pub, m)
	out = c.Bytes()
	return
}

// A DecryptionError represents a failure to decrypt a message.
// It is deliberately vague to avoid adaptive attacks.
type DecryptionError struct{}

func (DecryptionError) String() string { return "RSA decryption error" }

// A VerificationError represents a failure to verify a signature.
// It is deliberately vague to avoid adaptive attacks.
type VerificationError struct{}

func (VerificationError) String() string { return "RSA verification error" }

// modInverse returns ia, the inverse of a in the multiplicative group of prime
// order n. It requires that a be a member of the group (i.e. less than n).
func modInverse(a, n *big.Int) (ia *big.Int, ok bool) {
	g := new(big.Int)
	x := new(big.Int)
	y := new(big.Int)
	big.GcdInt(g, x, y, a, n)
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

// decrypt performs an RSA decryption, resulting in a plaintext integer. If a
// random source is given, RSA blinding is used.
func decrypt(rand io.Reader, priv *PrivateKey, c *big.Int) (m *big.Int, err os.Error) {
	// TODO(agl): can we get away with reusing blinds?
	if c.Cmp(priv.N) > 0 {
		err = DecryptionError{}
		return
	}

	var ir *big.Int
	if rand != nil {
		// Blinding enabled. Blinding involves multiplying c by r^e.
		// Then the decryption operation performs (m^e * r^e)^d mod n
		// which equals mr mod n. The factor of r can then be removed
		// by multipling by the multiplicative inverse of r.

		var r *big.Int

		for {
			r, err = randomNumber(rand, priv.N)
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
		c.Mul(c, rpowe)
		c.Mod(c, priv.N)
	}

	m = new(big.Int).Exp(c, priv.D, priv.N)

	if ir != nil {
		// Unblind.
		m.Mul(m, ir)
		m.Mod(m, priv.N)
	}

	return
}

// DecryptOAEP decrypts ciphertext using RSA-OAEP.
// If rand != nil, DecryptOAEP uses RSA blinding to avoid timing side-channel attacks.
func DecryptOAEP(hash hash.Hash, rand io.Reader, priv *PrivateKey, ciphertext []byte, label []byte) (msg []byte, err os.Error) {
	k := (priv.N.BitLen() + 7) / 8
	if len(ciphertext) > k ||
		k < hash.Size()*2+2 {
		err = DecryptionError{}
		return
	}

	c := new(big.Int).SetBytes(ciphertext)

	m, err := decrypt(rand, priv, c)
	if err != nil {
		return
	}

	hash.Write(label)
	lHash := hash.Sum()
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
