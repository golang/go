// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rand

import (
	"errors"
	"internal/byteorder"
	"internal/chacha8rand"
)

// A ChaCha8 is a ChaCha8-based cryptographically strong
// random number generator.
type ChaCha8 struct {
	state chacha8rand.State

	// The last readLen bytes of readBuf are still to be consumed by Read.
	readBuf [8]byte
	readLen int // 0 <= readLen <= 8
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
	c.readLen = 0
	c.readBuf = [8]byte{}
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

// Read reads exactly len(p) bytes into p.
// It always returns len(p) and a nil error.
//
// If calls to Read and Uint64 are interleaved, the order in which bits are
// returned by the two is undefined, and Read may return bits generated before
// the last call to Uint64.
func (c *ChaCha8) Read(p []byte) (n int, err error) {
	if c.readLen > 0 {
		n = copy(p, c.readBuf[len(c.readBuf)-c.readLen:])
		c.readLen -= n
		p = p[n:]
	}
	for len(p) >= 8 {
		byteorder.LePutUint64(p, c.Uint64())
		p = p[8:]
		n += 8
	}
	if len(p) > 0 {
		byteorder.LePutUint64(c.readBuf[:], c.Uint64())
		n += copy(p, c.readBuf[:])
		c.readLen = 8 - len(p)
	}
	return
}

// UnmarshalBinary implements the [encoding.BinaryUnmarshaler] interface.
func (c *ChaCha8) UnmarshalBinary(data []byte) error {
	data, ok := cutPrefix(data, []byte("readbuf:"))
	if ok {
		var buf []byte
		buf, data, ok = readUint8LengthPrefixed(data)
		if !ok {
			return errors.New("invalid ChaCha8 Read buffer encoding")
		}
		c.readLen = copy(c.readBuf[len(c.readBuf)-len(buf):], buf)
	}
	return chacha8rand.Unmarshal(&c.state, data)
}

func cutPrefix(s, prefix []byte) (after []byte, found bool) {
	if len(s) < len(prefix) || string(s[:len(prefix)]) != string(prefix) {
		return s, false
	}
	return s[len(prefix):], true
}

func readUint8LengthPrefixed(b []byte) (buf, rest []byte, ok bool) {
	if len(b) == 0 || len(b) < int(1+b[0]) {
		return nil, nil, false
	}
	return b[1 : 1+b[0]], b[1+b[0]:], true
}

// AppendBinary implements the [encoding.BinaryAppender] interface.
func (c *ChaCha8) AppendBinary(b []byte) ([]byte, error) {
	if c.readLen > 0 {
		b = append(b, "readbuf:"...)
		b = append(b, uint8(c.readLen))
		b = append(b, c.readBuf[len(c.readBuf)-c.readLen:]...)
	}
	return append(b, chacha8rand.Marshal(&c.state)...), nil
}

// MarshalBinary implements the [encoding.BinaryMarshaler] interface.
func (c *ChaCha8) MarshalBinary() ([]byte, error) {
	// the maximum length of (chacha8rand.Marshal + c.readBuf + "readbuf:") is 64
	return c.AppendBinary(make([]byte, 0, 64))
}
