// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package poly1305

import "encoding/binary"

const (
	msgBlock   = uint32(1 << 24)
	finalBlock = uint32(0)
)

// sumGeneric generates an authenticator for msg using a one-time key and
// puts the 16-byte result into out. This is the generic implementation of
// Sum and should be called if no assembly implementation is available.
func sumGeneric(out *[TagSize]byte, msg []byte, key *[32]byte) {
	h := newMACGeneric(key)
	h.Write(msg)
	h.Sum(out)
}

func newMACGeneric(key *[32]byte) (h macGeneric) {
	h.r[0] = binary.LittleEndian.Uint32(key[0:]) & 0x3ffffff
	h.r[1] = (binary.LittleEndian.Uint32(key[3:]) >> 2) & 0x3ffff03
	h.r[2] = (binary.LittleEndian.Uint32(key[6:]) >> 4) & 0x3ffc0ff
	h.r[3] = (binary.LittleEndian.Uint32(key[9:]) >> 6) & 0x3f03fff
	h.r[4] = (binary.LittleEndian.Uint32(key[12:]) >> 8) & 0x00fffff

	h.s[0] = binary.LittleEndian.Uint32(key[16:])
	h.s[1] = binary.LittleEndian.Uint32(key[20:])
	h.s[2] = binary.LittleEndian.Uint32(key[24:])
	h.s[3] = binary.LittleEndian.Uint32(key[28:])
	return
}

type macGeneric struct {
	h, r [5]uint32
	s    [4]uint32

	buffer [TagSize]byte
	offset int
}

func (h *macGeneric) Write(p []byte) (n int, err error) {
	n = len(p)
	if h.offset > 0 {
		remaining := TagSize - h.offset
		if n < remaining {
			h.offset += copy(h.buffer[h.offset:], p)
			return n, nil
		}
		copy(h.buffer[h.offset:], p[:remaining])
		p = p[remaining:]
		h.offset = 0
		updateGeneric(h.buffer[:], msgBlock, &(h.h), &(h.r))
	}
	if nn := len(p) - (len(p) % TagSize); nn > 0 {
		updateGeneric(p, msgBlock, &(h.h), &(h.r))
		p = p[nn:]
	}
	if len(p) > 0 {
		h.offset += copy(h.buffer[h.offset:], p)
	}
	return n, nil
}

func (h *macGeneric) Sum(out *[16]byte) {
	H, R := h.h, h.r
	if h.offset > 0 {
		var buffer [TagSize]byte
		copy(buffer[:], h.buffer[:h.offset])
		buffer[h.offset] = 1 // invariant: h.offset < TagSize
		updateGeneric(buffer[:], finalBlock, &H, &R)
	}
	finalizeGeneric(out, &H, &(h.s))
}

func updateGeneric(msg []byte, flag uint32, h, r *[5]uint32) {
	h0, h1, h2, h3, h4 := h[0], h[1], h[2], h[3], h[4]
	r0, r1, r2, r3, r4 := uint64(r[0]), uint64(r[1]), uint64(r[2]), uint64(r[3]), uint64(r[4])
	R1, R2, R3, R4 := r1*5, r2*5, r3*5, r4*5

	for len(msg) >= TagSize {
		// h += msg
		h0 += binary.LittleEndian.Uint32(msg[0:]) & 0x3ffffff
		h1 += (binary.LittleEndian.Uint32(msg[3:]) >> 2) & 0x3ffffff
		h2 += (binary.LittleEndian.Uint32(msg[6:]) >> 4) & 0x3ffffff
		h3 += (binary.LittleEndian.Uint32(msg[9:]) >> 6) & 0x3ffffff
		h4 += (binary.LittleEndian.Uint32(msg[12:]) >> 8) | flag

		// h *= r
		d0 := (uint64(h0) * r0) + (uint64(h1) * R4) + (uint64(h2) * R3) + (uint64(h3) * R2) + (uint64(h4) * R1)
		d1 := (d0 >> 26) + (uint64(h0) * r1) + (uint64(h1) * r0) + (uint64(h2) * R4) + (uint64(h3) * R3) + (uint64(h4) * R2)
		d2 := (d1 >> 26) + (uint64(h0) * r2) + (uint64(h1) * r1) + (uint64(h2) * r0) + (uint64(h3) * R4) + (uint64(h4) * R3)
		d3 := (d2 >> 26) + (uint64(h0) * r3) + (uint64(h1) * r2) + (uint64(h2) * r1) + (uint64(h3) * r0) + (uint64(h4) * R4)
		d4 := (d3 >> 26) + (uint64(h0) * r4) + (uint64(h1) * r3) + (uint64(h2) * r2) + (uint64(h3) * r1) + (uint64(h4) * r0)

		// h %= p
		h0 = uint32(d0) & 0x3ffffff
		h1 = uint32(d1) & 0x3ffffff
		h2 = uint32(d2) & 0x3ffffff
		h3 = uint32(d3) & 0x3ffffff
		h4 = uint32(d4) & 0x3ffffff

		h0 += uint32(d4>>26) * 5
		h1 += h0 >> 26
		h0 = h0 & 0x3ffffff

		msg = msg[TagSize:]
	}

	h[0], h[1], h[2], h[3], h[4] = h0, h1, h2, h3, h4
}

func finalizeGeneric(out *[TagSize]byte, h *[5]uint32, s *[4]uint32) {
	h0, h1, h2, h3, h4 := h[0], h[1], h[2], h[3], h[4]

	// h %= p reduction
	h2 += h1 >> 26
	h1 &= 0x3ffffff
	h3 += h2 >> 26
	h2 &= 0x3ffffff
	h4 += h3 >> 26
	h3 &= 0x3ffffff
	h0 += 5 * (h4 >> 26)
	h4 &= 0x3ffffff
	h1 += h0 >> 26
	h0 &= 0x3ffffff

	// h - p
	t0 := h0 + 5
	t1 := h1 + (t0 >> 26)
	t2 := h2 + (t1 >> 26)
	t3 := h3 + (t2 >> 26)
	t4 := h4 + (t3 >> 26) - (1 << 26)
	t0 &= 0x3ffffff
	t1 &= 0x3ffffff
	t2 &= 0x3ffffff
	t3 &= 0x3ffffff

	// select h if h < p else h - p
	t_mask := (t4 >> 31) - 1
	h_mask := ^t_mask
	h0 = (h0 & h_mask) | (t0 & t_mask)
	h1 = (h1 & h_mask) | (t1 & t_mask)
	h2 = (h2 & h_mask) | (t2 & t_mask)
	h3 = (h3 & h_mask) | (t3 & t_mask)
	h4 = (h4 & h_mask) | (t4 & t_mask)

	// h %= 2^128
	h0 |= h1 << 26
	h1 = ((h1 >> 6) | (h2 << 20))
	h2 = ((h2 >> 12) | (h3 << 14))
	h3 = ((h3 >> 18) | (h4 << 8))

	// s: the s part of the key
	// tag = (h + s) % (2^128)
	t := uint64(h0) + uint64(s[0])
	h0 = uint32(t)
	t = uint64(h1) + uint64(s[1]) + (t >> 32)
	h1 = uint32(t)
	t = uint64(h2) + uint64(s[2]) + (t >> 32)
	h2 = uint32(t)
	t = uint64(h3) + uint64(s[3]) + (t >> 32)
	h3 = uint32(t)

	binary.LittleEndian.PutUint32(out[0:], h0)
	binary.LittleEndian.PutUint32(out[4:], h1)
	binary.LittleEndian.PutUint32(out[8:], h2)
	binary.LittleEndian.PutUint32(out[12:], h3)
}
