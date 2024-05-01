// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package chacha8rand implements a pseudorandom generator
// based on ChaCha8. It is used by both runtime and math/rand/v2
// and must have no dependencies.
package chacha8rand

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
		leUint64(seed[0*8:]),
		leUint64(seed[1*8:]),
		leUint64(seed[2*8:]),
		leUint64(seed[3*8:]),
	})
}

// Init64 seeds the state with the given seed value.
func (s *State) Init64(seed [4]uint64) {
	s.seed = seed
	block(&s.seed, &s.buf, 0)
	s.c = 0
	s.i = 0
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
	bePutUint64(data[1*8:], uint64(used))
	for i, seed := range s.seed {
		lePutUint64(data[(2+i)*8:], seed)
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
	used := beUint64(data[1*8:])
	if used > (ctrMax/ctrInc)*chunk-reseed {
		return new(errUnmarshalChaCha8)
	}
	for i := range s.seed {
		s.seed[i] = leUint64(data[(2+i)*8:])
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

// binary.bigEndian.Uint64, copied to avoid dependency
func beUint64(b []byte) uint64 {
	_ = b[7] // bounds check hint to compiler; see golang.org/issue/14808
	return uint64(b[7]) | uint64(b[6])<<8 | uint64(b[5])<<16 | uint64(b[4])<<24 |
		uint64(b[3])<<32 | uint64(b[2])<<40 | uint64(b[1])<<48 | uint64(b[0])<<56
}

// binary.bigEndian.PutUint64, copied to avoid dependency
func bePutUint64(b []byte, v uint64) {
	_ = b[7] // early bounds check to guarantee safety of writes below
	b[0] = byte(v >> 56)
	b[1] = byte(v >> 48)
	b[2] = byte(v >> 40)
	b[3] = byte(v >> 32)
	b[4] = byte(v >> 24)
	b[5] = byte(v >> 16)
	b[6] = byte(v >> 8)
	b[7] = byte(v)
}

// binary.littleEndian.Uint64, copied to avoid dependency
func leUint64(b []byte) uint64 {
	_ = b[7] // bounds check hint to compiler; see golang.org/issue/14808
	return uint64(b[0]) | uint64(b[1])<<8 | uint64(b[2])<<16 | uint64(b[3])<<24 |
		uint64(b[4])<<32 | uint64(b[5])<<40 | uint64(b[6])<<48 | uint64(b[7])<<56
}

// binary.littleEndian.PutUint64, copied to avoid dependency
func lePutUint64(b []byte, v uint64) {
	_ = b[7] // early bounds check to guarantee safety of writes below
	b[0] = byte(v)
	b[1] = byte(v >> 8)
	b[2] = byte(v >> 16)
	b[3] = byte(v >> 24)
	b[4] = byte(v >> 32)
	b[5] = byte(v >> 40)
	b[6] = byte(v >> 48)
	b[7] = byte(v >> 56)
}
