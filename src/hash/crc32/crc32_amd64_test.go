// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package crc32

import (
	"math/rand"
	"testing"
)

func TestCastagnoliSSE42(t *testing.T) {
	if !sse42 {
		t.Skip("SSE42 not supported")
	}

	// Init the SSE42 tables.
	MakeTable(Castagnoli)

	// Manually init the software implementation to compare against.
	castagnoliTable = makeTable(Castagnoli)
	castagnoliTable8 = makeTable8(Castagnoli)

	// The optimized SSE4.2 implementation behaves differently for different
	// lengths (especially around multiples of K*3). Crosscheck against the
	// software implementation for various lengths.
	for _, base := range []int{castagnoliK1, castagnoliK2, castagnoliK1 + castagnoliK2} {
		for _, baseMult := range []int{2, 3, 5, 6, 9, 30} {
			for _, variation := range []int{0, 1, 2, 3, 4, 7, 10, 16, 32, 50, 128} {
				for _, varMult := range []int{-2, -1, +1, +2} {
					length := base*baseMult + variation*varMult
					p := make([]byte, length)
					_, _ = rand.Read(p)
					crcInit := uint32(rand.Int63())
					correct := updateSlicingBy8(crcInit, castagnoliTable8, p)
					result := updateCastagnoli(crcInit, p)
					if result != correct {
						t.Errorf("SSE42 implementation = 0x%x want 0x%x (buffer length %d)",
							result, correct, len(p))
					}
				}
			}
		}
	}
}
