// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package json

import (
	"encoding/binary"
	"math/bits"
)

// stringCache is a cache for strings converted from a []byte.
type stringCache = [256]string // 256*unsafe.Sizeof(string("")) => 4KiB

// makeString returns the string form of b.
// It returns a pre-allocated string from c if present, otherwise
// it allocates a new string, inserts it into the cache, and returns it.
func makeString(c *stringCache, b []byte) string {
	const (
		minCachedLen = 2   // single byte strings are already interned by the runtime
		maxCachedLen = 256 // large enough for UUIDs, IPv6 addresses, SHA-256 checksums, etc.
	)
	if c == nil || len(b) < minCachedLen || len(b) > maxCachedLen {
		return string(b)
	}

	// Compute a hash from the fixed-width prefix and suffix of the string.
	// This ensures hashing a string is a constant time operation.
	var h uint32
	switch {
	case len(b) >= 8:
		lo := binary.LittleEndian.Uint64(b[:8])
		hi := binary.LittleEndian.Uint64(b[len(b)-8:])
		h = hash64(uint32(lo), uint32(lo>>32)) ^ hash64(uint32(hi), uint32(hi>>32))
	case len(b) >= 4:
		lo := binary.LittleEndian.Uint32(b[:4])
		hi := binary.LittleEndian.Uint32(b[len(b)-4:])
		h = hash64(lo, hi)
	case len(b) >= 2:
		lo := binary.LittleEndian.Uint16(b[:2])
		hi := binary.LittleEndian.Uint16(b[len(b)-2:])
		h = hash64(uint32(lo), uint32(hi))
	}

	// Check the cache for the string.
	i := h % uint32(len(*c))
	if s := (*c)[i]; s == string(b) {
		return s
	}
	s := string(b)
	(*c)[i] = s
	return s
}

// hash64 returns the hash of two uint32s as a single uint32.
func hash64(lo, hi uint32) uint32 {
	// If avalanche=true, this is identical to XXH32 hash on a 8B string:
	//	var b [8]byte
	//	binary.LittleEndian.PutUint32(b[:4], lo)
	//	binary.LittleEndian.PutUint32(b[4:], hi)
	//	return xxhash.Sum32(b[:])
	const (
		prime1 = 0x9e3779b1
		prime2 = 0x85ebca77
		prime3 = 0xc2b2ae3d
		prime4 = 0x27d4eb2f
		prime5 = 0x165667b1
	)
	h := prime5 + uint32(8)
	h += lo * prime3
	h = bits.RotateLeft32(h, 17) * prime4
	h += hi * prime3
	h = bits.RotateLeft32(h, 17) * prime4
	// Skip final mix (avalanche) step of XXH32 for performance reasons.
	// Empirical testing shows that the improvements in unbiased distribution
	// does not outweigh the extra cost in computational complexity.
	const avalanche = false
	if avalanche {
		h ^= h >> 15
		h *= prime2
		h ^= h >> 13
		h *= prime3
		h ^= h >> 16
	}
	return h
}
