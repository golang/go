// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

package sha3

import (
	"crypto/internal/fips/subtle"
	"internal/cpu"
)

// This file contains code for using the 'compute intermediate
// message digest' (KIMD) and 'compute last message digest' (KLMD)
// instructions to compute SHA-3 and SHAKE hashes on IBM Z. See
// [z/Architecture Principles of Operation, Fourteen Edition].
//
// [z/Architecture Principles of Operation, Fourteen Edition]: https://www.ibm.com/docs/en/module_1678991624569/pdf/SA22-7832-13.pdf

func keccakF1600(a *[200]byte) {
	keccakF1600Generic(a)
}

// codes represent 7-bit KIMD/KLMD function codes as defined in
// the Principles of Operation.
type code uint64

const (
	// Function codes for KIMD/KLMD, from Figure 7-207.
	sha3_224  code = 32
	sha3_256  code = 33
	sha3_384  code = 34
	sha3_512  code = 35
	shake_128 code = 36
	shake_256 code = 37
	nopad          = 0x100
)

// kimd is a wrapper for the 'compute intermediate message digest' instruction.
// src is absorbed into the sponge state a.
// len(src) must be a multiple of the rate for the given function code.
//
//go:noescape
func kimd(function code, a *[200]byte, src []byte)

// klmd is a wrapper for the 'compute last message digest' instruction.
// src is padded and absorbed into the sponge state a.
//
// If the function is a SHAKE XOF, the sponge is then optionally squeezed into
// dst by first applying the permutation and then copying the output until dst
// runs out. If len(dst) is a multiple of rate (including zero), the final
// permutation is not applied. If the nopad bit of function is set and len(src)
// is zero, only squeezing is performed.
//
//go:noescape
func klmd(function code, a *[200]byte, dst, src []byte)

func (d *Digest) write(p []byte) (n int, err error) {
	if d.state != spongeAbsorbing {
		panic("sha3: Write after Read")
	}
	if !cpu.S390X.HasSHA3 {
		return d.writeGeneric(p)
	}

	n = len(p)

	// If there is buffered input in the state, keep XOR'ing.
	if d.n > 0 {
		x := subtle.XORBytes(d.a[d.n:d.rate], d.a[d.n:d.rate], p)
		d.n += x
		p = p[x:]
	}

	// If the sponge is full, apply the permutation.
	if d.n == d.rate {
		// Absorbing a "rate"ful of zeroes effectively XORs the state with
		// zeroes (a no-op) and then runs the permutation. The actual function
		// doesn't matter, they all run the same permutation.
		kimd(shake_128, &d.a, make([]byte, rateK256))
		d.n = 0
	}

	// Absorb full blocks with KIMD.
	if len(p) >= d.rate {
		wholeBlocks := len(p) / d.rate * d.rate
		kimd(d.function(), &d.a, p[:wholeBlocks])
		p = p[wholeBlocks:]
	}

	// If there is any trailing input, XOR it into the state.
	if len(p) > 0 {
		d.n += subtle.XORBytes(d.a[d.n:d.rate], d.a[d.n:d.rate], p)
	}

	return
}

func (d *Digest) sum(b []byte) []byte {
	if d.state != spongeAbsorbing {
		panic("sha3: Sum after Read")
	}
	if !cpu.S390X.HasSHA3 ||
		d.dsbyte != dsbyteSHA3 && d.dsbyte != dsbyteShake {
		return d.sumGeneric(b)
	}

	// Copy the state to preserve the original.
	a := d.a

	// We "absorb" a buffer of zeroes as long as the amount of input we already
	// XOR'd into the sponge, to skip over it. The max cap is specified to avoid
	// an allocation.
	buf := make([]byte, d.n, rateK256)
	function := d.function()
	switch function {
	case sha3_224, sha3_256, sha3_384, sha3_512:
		klmd(function, &a, nil, buf)
		return append(b, a[:d.outputLen]...)
	case shake_128, shake_256:
		h := make([]byte, d.outputLen, 64)
		klmd(function, &a, h, buf)
		return append(b, h...)
	default:
		panic("sha3: unknown function")
	}
}

func (d *Digest) read(out []byte) (n int, err error) {
	if !cpu.S390X.HasSHA3 || d.dsbyte != dsbyteShake {
		return d.readGeneric(out)
	}

	n = len(out)

	if d.state == spongeAbsorbing {
		d.state = spongeSqueezing

		// We "absorb" a buffer of zeroes as long as the amount of input we
		// already XOR'd into the sponge, to skip over it. The max cap is
		// specified to avoid an allocation.
		buf := make([]byte, d.n, rateK256)
		klmd(d.function(), &d.a, out, buf)
	} else {
		// We have "buffered" output still to copy.
		if d.n < d.rate {
			x := copy(out, d.a[d.n:d.rate])
			d.n += x
			out = out[x:]
		}
		if len(out) == 0 {
			return
		}

		klmd(d.function()|nopad, &d.a, out, nil)
	}

	if len(out)%d.rate == 0 {
		// The final permutation was not performed,
		// so there is no "buffered" output.
		d.n = d.rate
	} else {
		d.n = len(out) % d.rate
	}

	return
}

func (d *Digest) function() code {
	switch d.rate {
	case rateK256:
		return shake_128
	case rateK448:
		return sha3_224
	case rateK512:
		if d.dsbyte == dsbyteSHA3 {
			return sha3_256
		} else {
			return shake_256
		}
	case rateK768:
		return sha3_384
	case rateK1024:
		return sha3_512
	default:
		panic("invalid rate")
	}
}
