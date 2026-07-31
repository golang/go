// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sha3

import (
	"crypto/internal/fips140"
	"slices"
)

// ReadMulti reads output (like [SHAKE.Read] would) from each element of s into
// the corresponding element of out. It behaves exactly like
//
//	for i := range s {
//	    s[i].Read(out[i])
//	}
//
// Depending on the platform, groups of states with the same rate, sponge phase,
// and output offset may be processed together, which can be faster than reading
// them one at a time.
//
// ReadMulti panics if len(s) != len(out). The output slices may have different
// lengths, including zero, and must not overlap each other.
func ReadMulti(s []*SHAKE, out [][]byte) {
	if len(s) != len(out) {
		panic("sha3: ReadMulti: mismatched lengths")
	}
	fips140.RecordApproved()
	for len(s) > 0 {
		k := min(batchSize, len(s))
		if k >= 2 && batchable(s[:k]) {
			readMultiGroup(s[:k], out[:k])
		} else {
			for i := range s[:k] {
				s[i].d.read(out[i])
			}
		}
		s, out = s[k:], out[k:]
	}
}

// batchSize is the maximum number of states passed to a Keccak-f[1600] kernel
// together. It matches the widest kernel arity (four lanes with AVX2).
const batchSize = 4

// batchable reports whether the states can be read in lockstep.
func batchable(s []*SHAKE) bool {
	// The states must be non-nil and distinct.
	for i, x := range s {
		if x == nil {
			return false
		}
		if slices.Contains(s[i+1:], x) {
			return false
		}
	}
	// The states must have the same rate, sponge phase, and output offset.
	d0 := &s[0].d
	for _, x := range s[1:] {
		if x.d.rate != d0.rate || x.d.state != d0.state {
			return false
		}
		if d0.state == spongeSqueezing && x.d.n != d0.n {
			return false
		}
	}
	return true
}

// readMultiGroup reads o[i] from s[i] for a group of two to batchSize lockstep
// states (as checked by [batchable]), running their permutations together.
func readMultiGroup(s []*SHAKE, o [][]byte) {
	// Copy the output slices, to advance them below without corrupting the
	// caller's slice. The fixed-size array with indexed stores (not append
	// or copy) keeps escape analysis from moving the buffers to the heap.
	var out [batchSize][]byte
	for i := range o {
		out[i] = o[i]
	}

	// If the states are still absorbing, pad and permute all of them,
	// like [Digest.padAndPermute].
	if s[0].d.state == spongeAbsorbing {
		var p [batchSize]*[200]byte
		for i := range s {
			d := &s[i].d
			d.a[d.n] ^= d.dsbyte
			d.a[d.rate-1] ^= 0x80
			d.state = spongeSqueezing
			d.n = 0
			p[i] = &d.a
		}
		keccakF1600Multi(p[:len(s)])
	}

	rate := s[0].d.rate
	for {
		var p [batchSize]*[200]byte
		m := 0
		for i := range s {
			d := &s[i].d
			// Copy out any remaining buffered output.
			if len(out[i]) > 0 && d.n < rate {
				x := copy(out[i], d.a[d.n:rate])
				d.n += x
				out[i] = out[i][x:]
			}
			// If this state still needs to produce more,
			// add it to the permutation batch.
			if len(out[i]) > 0 {
				p[m] = &d.a
				m++
				d.n = 0
			}
		}
		if m == 0 {
			return
		}
		keccakF1600Multi(p[:m])
	}
}

// keccakF1600MultiGeneric applies the permutation to all states.
func keccakF1600MultiGeneric(a []*[200]byte) {
	for _, s := range a {
		keccakF1600(s)
	}
}
