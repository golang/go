// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rand

import (
	"big"
	"io"
	"os"
)

// Prime returns a number, p, of the given size, such that p is prime
// with high probability.
func Prime(rand io.Reader, bits int) (p *big.Int, err error) {
	if bits < 1 {
		err = os.EINVAL
	}

	b := uint(bits % 8)
	if b == 0 {
		b = 8
	}

	bytes := make([]byte, (bits+7)/8)
	p = new(big.Int)

	for {
		_, err = io.ReadFull(rand, bytes)
		if err != nil {
			return nil, err
		}

		// Clear bits in the first byte to make sure the candidate has a size <= bits.
		bytes[0] &= uint8(int(1<<b) - 1)
		// Don't let the value be too small, i.e, set the most significant bit.
		bytes[0] |= 1 << (b - 1)
		// Make the value odd since an even number this large certainly isn't prime.
		bytes[len(bytes)-1] |= 1

		p.SetBytes(bytes)
		if big.ProbablyPrime(p, 20) {
			return
		}
	}

	return
}

// Int returns a uniform random value in [0, max).
func Int(rand io.Reader, max *big.Int) (n *big.Int, err error) {
	k := (max.BitLen() + 7) / 8

	// b is the number of bits in the most significant byte of max.
	b := uint(max.BitLen() % 8)
	if b == 0 {
		b = 8
	}

	bytes := make([]byte, k)
	n = new(big.Int)

	for {
		_, err = io.ReadFull(rand, bytes)
		if err != nil {
			return nil, err
		}

		// Clear bits in the first byte to increase the probability
		// that the candidate is < max.
		bytes[0] &= uint8(int(1<<b) - 1)

		n.SetBytes(bytes)
		if n.Cmp(max) < 0 {
			return
		}
	}

	return
}
