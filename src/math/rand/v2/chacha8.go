// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rand

import "internal/chacha8rand"

// A ChaCha8 is a ChaCha8-based cryptographically strong
// random number generator.
type ChaCha8 struct {
	state chacha8rand.State
}

// NewChaCha8 returns a new ChaCha8 seeded with the given seed.
func NewChaCha8(seed [32]byte) *ChaCha8 {
	c := new(ChaCha8)
	c.state.Init(seed)
	return c
}

// Seed resets the ChaCha8 to behave the same way as NewChaCha8(seed).
func (c *ChaCha8) Seed(seed [32]byte) {
	c.state.Init(seed)
}

// Uint64 returns a uniformly distributed random uint64 value.
func (c *ChaCha8) Uint64() uint64 {
	for {
		x, ok := c.state.Next()
		if ok {
			return x
		}
		c.state.Refill()
	}
}

// UnmarshalBinary implements the encoding.BinaryUnmarshaler interface.
func (c *ChaCha8) UnmarshalBinary(data []byte) error {
	return chacha8rand.Unmarshal(&c.state, data)
}

// MarshalBinary implements the encoding.BinaryMarshaler interface.
func (c *ChaCha8) MarshalBinary() ([]byte, error) {
	return chacha8rand.Marshal(&c.state), nil
}
