// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package big

import "math/rand"

// ProbablyPrime performs n Miller-Rabin tests to check whether x is prime.
// If x is prime, it returns true.
// If x is not prime, it returns false with probability at least 1 - ¼ⁿ.
//
// It is not suitable for judging primes that an adversary may have crafted
// to fool this test.
func (x *Int) ProbablyPrime(n int) bool {
	if n <= 0 {
		panic("non-positive n for ProbablyPrime")
	}
	return !x.neg && x.abs.probablyPrime(n)
}

// probablyPrime performs n Miller-Rabin tests to check whether x is prime.
// If x is prime, it returns true.
// If x is not prime, it returns false with probability at least 1 - ¼ⁿ.
//
// It is not suitable for judging primes that an adversary may have crafted
// to fool this test.
func (n nat) probablyPrime(reps int) bool {
	if len(n) == 0 {
		return false
	}

	if len(n) == 1 {
		if n[0] < 2 {
			return false
		}

		if n[0]%2 == 0 {
			return n[0] == 2
		}

		// We have to exclude these cases because we reject all
		// multiples of these numbers below.
		switch n[0] {
		case 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53:
			return true
		}
	}

	if n[0]&1 == 0 {
		return false // n is even
	}

	const primesProduct32 = 0xC0CFD797         // Π {p ∈ primes, 2 < p <= 29}
	const primesProduct64 = 0xE221F97C30E94E1D // Π {p ∈ primes, 2 < p <= 53}

	var r Word
	switch _W {
	case 32:
		r = n.modW(primesProduct32)
	case 64:
		r = n.modW(primesProduct64 & _M)
	default:
		panic("Unknown word size")
	}

	if r%3 == 0 || r%5 == 0 || r%7 == 0 || r%11 == 0 ||
		r%13 == 0 || r%17 == 0 || r%19 == 0 || r%23 == 0 || r%29 == 0 {
		return false
	}

	if _W == 64 && (r%31 == 0 || r%37 == 0 || r%41 == 0 ||
		r%43 == 0 || r%47 == 0 || r%53 == 0) {
		return false
	}

	nm1 := nat(nil).sub(n, natOne)
	// determine q, k such that nm1 = q << k
	k := nm1.trailingZeroBits()
	q := nat(nil).shr(nm1, k)

	nm3 := nat(nil).sub(nm1, natTwo)
	rand := rand.New(rand.NewSource(int64(n[0])))

	var x, y, quotient nat
	nm3Len := nm3.bitLen()

NextRandom:
	for i := 0; i < reps; i++ {
		x = x.random(rand, nm3, nm3Len)
		x = x.add(x, natTwo)
		y = y.expNN(x, q, n)
		if y.cmp(natOne) == 0 || y.cmp(nm1) == 0 {
			continue
		}
		for j := uint(1); j < k; j++ {
			y = y.mul(y, y)
			quotient, y = quotient.div(y, y, n)
			if y.cmp(nm1) == 0 {
				continue NextRandom
			}
			if y.cmp(natOne) == 0 {
				return false
			}
		}
		return false
	}

	return true
}
