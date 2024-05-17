// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package chacha8rand implements a pseudorandom generator
// based on ChaCha8. It is used by both runtime and math/rand/v2
// and must have minimal dependencies.
package chacha8rand

import "internal/byteorder"

const (
	ctrInc = 4  // increment counter by 4 between block calls
	ctrMax = 16 // reseed when counter reaches 16
	chunk  = 32 // each chunk produced by block is 32 uint64s
	reseed = 4  // reseed with 4 words
)

// block is the chacha8rand block function.
func block(seed *[4]uint64, blocks *[32]uint64, counter uint32)

// A State holds the state for a single random generator.
// It must be used from one goroutine at a time.
// If used by multiple goroutines at a time, the goroutines
// may see the same random values, but the code will not
// crash or cause out-of-bounds memory accesses.
type State struct {
	buf  [32]uint64
	seed [4]uint64
	i    uint32
	n    uint32
	c    uint32

	// readVal contains remainder of 63-bit integer used for bytes
	// generation during most recent Read call.
	// It is saved so next Read call can start where the previous
	// one finished.
	readVal uint64
	// readPos indicates the number of low-order bytes of readVal
	// that are still valid.
	readPos int8
}

// Next returns the next random value, along with a boolean
// indicating whether one was available.
// If one is not available, the caller should call Refill
// and then repeat the call to Next.
//
// Next is //go:nosplit to allow its use in the runtime
// with per-m data without holding the per-m lock.
//
//go:nosplit
func (s *State) Next() (uint64, bool) {
	i := s.i
	if i >= s.n {
		return 0, false
	}
	s.i = i + 1
	return s.buf[i&31], true // i&31 eliminates bounds check
}

// Init seeds the State with the given seed value.
func (s *State) Init(seed [32]byte) {
	s.Init64([4]uint64{
		byteorder.LeUint64(seed[0*8:]),
		byteorder.LeUint64(seed[1*8:]),
		byteorder.LeUint64(seed[2*8:]),
		byteorder.LeUint64(seed[3*8:]),
	})
}

// Init64 seeds the state with the given seed value.
func (s *State) Init64(seed [4]uint64) {
	s.seed = seed
	block(&s.seed, &s.buf, 0)
	s.c = 0
	s.i = 0
	s.readPos = 0
	s.readVal = 0
	s.n = chunk
}

// Refill refills the state with more random values.
// After a call to Refill, an immediate call to Next will succeed
// (unless multiple goroutines are incorrectly sharing a state).
func (s *State) Refill() {
	s.c += ctrInc
	if s.c == ctrMax {
		// Reseed with generated uint64s for forward secrecy.
		// Normally this is done immediately after computing a block,
		// but we do it immediately before computing the next block,
		// to allow a much smaller serialized state (just the seed plus offset).
		// This gives a delayed benefit for the forward secrecy
		// (you can reconstruct the recent past given a memory dump),
		// which we deem acceptable in exchange for the reduced size.
		s.seed[0] = s.buf[len(s.buf)-reseed+0]
		s.seed[1] = s.buf[len(s.buf)-reseed+1]
		s.seed[2] = s.buf[len(s.buf)-reseed+2]
		s.seed[3] = s.buf[len(s.buf)-reseed+3]
		s.c = 0
	}
	block(&s.seed, &s.buf, s.c)
	s.i = 0
	s.n = uint32(len(s.buf))
	if s.c == ctrMax-ctrInc {
		s.n = uint32(len(s.buf)) - reseed
	}
}

// Reseed reseeds the state with new random values.
// After a call to Reseed, any previously returned random values
// have been erased from the memory of the state and cannot be
// recovered.
func (s *State) Reseed() {
	var seed [4]uint64
	for i := range seed {
		for {
			x, ok := s.Next()
			if ok {
				seed[i] = x
				break
			}
			s.Refill()
		}
	}
	s.Init64(seed)
}

// Marshal marshals the state into a byte slice.
// Marshal and Unmarshal are functions, not methods,
// so that they will not be linked into the runtime
// when it uses the State struct, since the runtime
// does not need these.
func Marshal(s *State) []byte {
	data := make([]byte, 6*8)
	copy(data, "chacha8:")
	used := (s.c/ctrInc)*chunk + s.i
	byteorder.BePutUint64(data[1*8:], uint64(used))
	for i, seed := range s.seed {
		byteorder.LePutUint64(data[(2+i)*8:], seed)
	}
	return data
}

type errUnmarshalChaCha8 struct{}

func (*errUnmarshalChaCha8) Error() string {
	return "invalid ChaCha8 encoding"
}

// Unmarshal unmarshals the state from a byte slice.
func Unmarshal(s *State, data []byte) error {
	if len(data) != 6*8 || string(data[:8]) != "chacha8:" {
		return new(errUnmarshalChaCha8)
	}
	used := byteorder.BeUint64(data[1*8:])
	if used > (ctrMax/ctrInc)*chunk-reseed {
		return new(errUnmarshalChaCha8)
	}
	for i := range s.seed {
		s.seed[i] = byteorder.LeUint64(data[(2+i)*8:])
	}
	s.c = ctrInc * (uint32(used) / chunk)
	block(&s.seed, &s.buf, s.c)
	s.i = uint32(used) % chunk
	s.n = chunk
	if s.c == ctrMax-ctrInc {
		s.n = chunk - reseed
	}
	return nil
}

// Read reads random bytes from the state into p.
func Read(s *State, p []byte) (n int) {
	pos := s.readPos
	val := s.readVal
	var ok bool
	for n = 0; n < len(p); n++ {
		if pos == 0 {
			for {
				val, ok = s.Next()
				if ok {
					break
				}
				s.Refill()
			}
			pos = 7
		}
		p[n] = byte(val)
		val >>= 8
		pos--
	}
	s.readPos = pos
	s.readVal = val
	return
}
