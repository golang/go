// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sha3

// spongeDirection indicates the direction bytes are flowing through the sponge.
type spongeDirection int

const (
	// spongeAbsorbing indicates that the sponge is absorbing input.
	spongeAbsorbing spongeDirection = iota
	// spongeSqueezing indicates that the sponge is being squeezed.
	spongeSqueezing
)

const (
	// maxRate is the maximum size of the internal buffer. SHAKE-256
	// currently needs the largest buffer.
	maxRate = 168
)

type state struct {
	// Generic sponge components.
	a    [25]uint64 // main state of the hash
	rate int        // the number of bytes of state to use

	// dsbyte contains the "domain separation" bits and the first bit of
	// the padding. Sections 6.1 and 6.2 of [1] separate the outputs of the
	// SHA-3 and SHAKE functions by appending bitstrings to the message.
	// Using a little-endian bit-ordering convention, these are "01" for SHA-3
	// and "1111" for SHAKE, or 00000010b and 00001111b, respectively. Then the
	// padding rule from section 5.1 is applied to pad the message to a multiple
	// of the rate, which involves adding a "1" bit, zero or more "0" bits, and
	// a final "1" bit. We merge the first "1" bit from the padding into dsbyte,
	// giving 00000110b (0x06) and 00011111b (0x1f).
	// [1] http://csrc.nist.gov/publications/drafts/fips-202/fips_202_draft.pdf
	//     "Draft FIPS 202: SHA-3 Standard: Permutation-Based Hash and
	//      Extendable-Output Functions (May 2014)"
	dsbyte byte

	i, n    int // storage[i:n] is the buffer, i is only used while squeezing
	storage [maxRate]byte

	// Specific to SHA-3 and SHAKE.
	outputLen int             // the default output size in bytes
	state     spongeDirection // whether the sponge is absorbing or squeezing
}

// BlockSize returns the rate of sponge underlying this hash function.
func (d *state) BlockSize() int { return d.rate }

// Size returns the output size of the hash function in bytes.
func (d *state) Size() int { return d.outputLen }

// Reset clears the internal state by zeroing the sponge state and
// the buffer indexes, and setting Sponge.state to absorbing.
func (d *state) Reset() {
	// Zero the permutation's state.
	for i := range d.a {
		d.a[i] = 0
	}
	d.state = spongeAbsorbing
	d.i, d.n = 0, 0
}

func (d *state) clone() *state {
	ret := *d
	return &ret
}

// permute applies the KeccakF-1600 permutation. It handles
// any input-output buffering.
func (d *state) permute() {
	switch d.state {
	case spongeAbsorbing:
		// If we're absorbing, we need to xor the input into the state
		// before applying the permutation.
		xorIn(d, d.storage[:d.rate])
		d.n = 0
		keccakF1600(&d.a)
	case spongeSqueezing:
		// If we're squeezing, we need to apply the permutation before
		// copying more output.
		keccakF1600(&d.a)
		d.i = 0
		copyOut(d, d.storage[:d.rate])
	}
}

// pads appends the domain separation bits in dsbyte, applies
// the multi-bitrate 10..1 padding rule, and permutes the state.
func (d *state) padAndPermute() {
	// Pad with this instance's domain-separator bits. We know that there's
	// at least one byte of space in d.buf because, if it were full,
	// permute would have been called to empty it. dsbyte also contains the
	// first one bit for the padding. See the comment in the state struct.
	d.storage[d.n] = d.dsbyte
	d.n++
	for d.n < d.rate {
		d.storage[d.n] = 0
		d.n++
	}
	// This adds the final one bit for the padding. Because of the way that
	// bits are numbered from the LSB upwards, the final bit is the MSB of
	// the last byte.
	d.storage[d.rate-1] ^= 0x80
	// Apply the permutation
	d.permute()
	d.state = spongeSqueezing
	d.n = d.rate
	copyOut(d, d.storage[:d.rate])
}

// Write absorbs more data into the hash's state. It panics if any
// output has already been read.
func (d *state) Write(p []byte) (written int, err error) {
	if d.state != spongeAbsorbing {
		panic("sha3: Write after Read")
	}
	written = len(p)

	for len(p) > 0 {
		if d.n == 0 && len(p) >= d.rate {
			// The fast path; absorb a full "rate" bytes of input and apply the permutation.
			xorIn(d, p[:d.rate])
			p = p[d.rate:]
			keccakF1600(&d.a)
		} else {
			// The slow path; buffer the input until we can fill the sponge, and then xor it in.
			todo := d.rate - d.n
			if todo > len(p) {
				todo = len(p)
			}
			d.n += copy(d.storage[d.n:], p[:todo])
			p = p[todo:]

			// If the sponge is full, apply the permutation.
			if d.n == d.rate {
				d.permute()
			}
		}
	}

	return
}

// Read squeezes an arbitrary number of bytes from the sponge.
func (d *state) Read(out []byte) (n int, err error) {
	// If we're still absorbing, pad and apply the permutation.
	if d.state == spongeAbsorbing {
		d.padAndPermute()
	}

	n = len(out)

	// Now, do the squeezing.
	for len(out) > 0 {
		n := copy(out, d.storage[d.i:d.n])
		d.i += n
		out = out[n:]

		// Apply the permutation if we've squeezed the sponge dry.
		if d.i == d.rate {
			d.permute()
		}
	}

	return
}

// Sum applies padding to the hash state and then squeezes out the desired
// number of output bytes. It panics if any output has already been read.
func (d *state) Sum(in []byte) []byte {
	if d.state != spongeAbsorbing {
		panic("sha3: Sum after Read")
	}

	// Make a copy of the original hash so that caller can keep writing
	// and summing.
	dup := d.clone()
	hash := make([]byte, dup.outputLen, 64) // explicit cap to allow stack allocation
	dup.Read(hash)
	return append(in, hash...)
}
