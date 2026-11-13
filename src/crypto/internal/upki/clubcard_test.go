// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package upki

import "testing"

func TestFilterContains(t *testing.T) {
	t.Parallel()

	// Rank-zero blocks encode an empty revoked set without allocating filter
	// columns. Inversion changes that empty set into the whole issuer universe.
	t.Run("rank zero", func(t *testing.T) {
		t.Parallel()

		f := new(filter)
		if f.contains(&blockMeta{inverted: true}, [32]byte{}, nil) != true {
			t.Error("inverted rank-zero block did not contain the query")
		}
		if f.contains(&blockMeta{}, [32]byte{}, nil) != false {
			t.Error("non-inverted rank-zero block contained the query")
		}
	})

	// Exception serials are insertion failures from the exact filter. They are
	// non-members before the block's final inversion is applied.
	t.Run("exceptions", func(t *testing.T) {
		t.Parallel()

		f := &filter{x: [][]uint64{make([]uint64, 4)}, y: make([]uint64, 4)}
		for _, inverted := range []bool{false, true} {
			meta := &blockMeta{
				hModulus: 1,
				hRank:    1,
				gModulus: 1,
				exceptions: map[string]struct{}{
					"exception": {},
				},
				inverted: inverted,
			}
			if got := f.contains(meta, [32]byte{}, []byte("exception")); got != inverted {
				t.Errorf("inverted=%v exception result = %v, want %v", inverted, got, inverted)
			}
			if got := f.contains(meta, [32]byte{}, []byte("other")); got == inverted {
				t.Errorf("inverted=%v non-exception result = %v, want %v", inverted, got, !inverted)
			}
		}
	})

	// A query is rejected when its approximate-filter dot product is nonzero.
	// Pairing the query's explicit bit with its implicit low bit makes the dot
	// product zero and demonstrates the corresponding acceptance path.
	t.Run("approximate column", func(t *testing.T) {
		t.Parallel()

		var hash [32]byte
		hash[0] = 0x04 // bit 2, in addition to the implicit bit 0
		x := [][]uint64{make([]uint64, 4)}
		setFilterRowBit(x, 0, 2)
		f := &filter{x: x, y: make([]uint64, 4)}
		meta := &blockMeta{hModulus: 1, hRank: 1, gModulus: 1}
		if f.contains(meta, hash, nil) {
			t.Error("query with nonzero approximate dot product was accepted")
		}

		setFilterRowBit(x, 0, 0)
		if !f.contains(meta, hash, nil) {
			t.Error("query with zero approximate dot product was rejected")
		}
	})

	// Every approximate column up to hRank participates in the query. Planting
	// the rejecting bit in each column in turn confirms none are skipped.
	t.Run("multiple approximate columns", func(t *testing.T) {
		t.Parallel()

		var hash [32]byte
		hash[0] = 0x04
		meta := &blockMeta{hModulus: 1, hRank: 3, gModulus: 1}
		for column := range int(meta.hRank) {
			x := [][]uint64{make([]uint64, 4), make([]uint64, 4), make([]uint64, 4)}
			setFilterRowBit(x, column, 2)
			f := &filter{x: x, y: make([]uint64, 4)}
			if f.contains(meta, hash, nil) {
				t.Errorf("query rejected by column %d was accepted", column)
			}
		}
	})

	// hOffset selects the issuer block's window within a shared column. A bit
	// inside that window must affect the result, and the same bit outside it
	// must not.
	t.Run("block row offset", func(t *testing.T) {
		t.Parallel()

		var hash [32]byte
		hash[0] = 0x04
		x := [][]uint64{make([]uint64, 8)}
		setFilterRowBit(x, 0, 256+2)
		f := &filter{x: x, y: make([]uint64, 4)}
		meta := &blockMeta{hOffset: 256, hModulus: 1, hRank: 1, gModulus: 1}
		if f.contains(meta, hash, nil) {
			t.Error("query rejected inside the block window was accepted")
		}

		clear(x[0])
		setFilterRowBit(x, 0, 2)
		if !f.contains(meta, hash, nil) {
			t.Error("bit outside the block window affected the query")
		}
	})

	// The exact-filter vector is evaluated after the approximate columns. A
	// nonzero exact dot product rejects a query that passed the earlier
	// stage, and pairing the query's bit with its implicit low bit makes the
	// product zero again and accepts it.
	t.Run("exact column", func(t *testing.T) {
		t.Parallel()

		var hash [32]byte
		hash[0] = 0x04 // bit 2, in addition to the implicit bit 0
		y := make([]uint64, 4)
		setFilterRowBit([][]uint64{y}, 0, 2)
		f := &filter{x: [][]uint64{make([]uint64, 4)}, y: y}
		meta := &blockMeta{hModulus: 1, hRank: 1, gModulus: 1}
		if f.contains(meta, hash, nil) {
			t.Error("query with nonzero exact dot product was accepted")
		}

		setFilterRowBit([][]uint64{y}, 0, 0)
		if !f.contains(meta, hash, nil) {
			t.Error("query with zero exact dot product was rejected")
		}
	})

	// A zero exact modulus does not skip the exact filter: the reference
	// implementation reduces modulo max(1, m), so the row is gOffset and a
	// bit there still rejects the query.
	t.Run("exact column zero modulus", func(t *testing.T) {
		t.Parallel()

		var hash [32]byte
		hash[0] = 0x04
		y := make([]uint64, 4)
		setFilterRowBit([][]uint64{y}, 0, 100+2)
		f := &filter{x: [][]uint64{make([]uint64, 4)}, y: y}
		meta := &blockMeta{hModulus: 1, hRank: 1, gOffset: 100, gModulus: 0}
		if f.contains(meta, hash, nil) {
			t.Error("query rejected by the exact filter at gOffset was accepted")
		}
	})

	// The blocks above use a modulus of 1, which pins the probed row at the
	// block offset whatever the hash. With a larger modulus the row is
	// offset + t mod modulus, where t is the high 64 bits of the hash read
	// little-endian (§4.3 step 3), so bytes 24 through 31 select the row.
	// Plant the rejecting bit at that row, then at the row the low limb
	// would select, to pin which end of the hash t is taken from.
	t.Run("approximate row selection", func(t *testing.T) {
		t.Parallel()

		var hash [32]byte
		hash[0] = 0x04  // coefficient bit 2; the low limb is 5 once bit 0 is forced
		hash[24] = 0x0d // t = 13
		const offset, modulus = 100, 7
		selectedRow := uint64(offset + 13%modulus)
		lowLimbRow := uint64(offset + 5%modulus)
		meta := &blockMeta{hOffset: offset, hModulus: modulus, hRank: 1, gModulus: 1}
		x := [][]uint64{make([]uint64, 4)}
		setFilterRowBit(x, 0, selectedRow+2)
		f := &filter{x: x, y: make([]uint64, 4)}
		if f.contains(meta, hash, nil) {
			t.Error("query rejected at the row selected by t was accepted")
		}

		clear(x[0])
		setFilterRowBit(x, 0, lowLimbRow+2)
		if !f.contains(meta, hash, nil) {
			t.Error("bit at the row selected by the low limb affected the query")
		}
	})

	t.Run("exact row selection", func(t *testing.T) {
		t.Parallel()

		var hash [32]byte
		hash[0] = 0x04
		hash[24] = 0x0d
		const offset, modulus = 100, 7
		selectedRow := uint64(offset + 13%modulus)
		lowLimbRow := uint64(offset + 5%modulus)
		meta := &blockMeta{hModulus: 1, hRank: 1, gOffset: offset, gModulus: modulus}
		y := make([]uint64, 4)
		setFilterRowBit([][]uint64{y}, 0, selectedRow+2)
		f := &filter{x: [][]uint64{make([]uint64, 4)}, y: y}
		if f.contains(meta, hash, nil) {
			t.Error("query rejected at the exact row selected by t was accepted")
		}

		clear(y)
		setFilterRowBit([][]uint64{y}, 0, lowLimbRow+2)
		if !f.contains(meta, hash, nil) {
			t.Error("bit at the exact row selected by the low limb affected the query")
		}
	})
}

func TestBitsXorIsZero(t *testing.T) {
	t.Parallel()

	// Cover empty and nonzero products, a query window crossing a uint64
	// boundary, and the specified zero-extension beyond a short column.
	tests := []struct {
		name   string
		column []uint64
		offset uint64
		coef   [ribbonWidth]uint64
		want   bool
	}{
		{"zero", nil, 0, [ribbonWidth]uint64{1}, true},
		{"one", []uint64{1}, 0, [ribbonWidth]uint64{1}, false},
		{"crosses limb", []uint64{0, 1}, 63, [ribbonWidth]uint64{2}, false},
		{"past end", []uint64{^uint64(0)}, 64, [ribbonWidth]uint64{^uint64(0)}, true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			if got := bitsXorIsZero(test.column, test.offset, test.coef); got != test.want {
				t.Errorf("bitsXorIsZero = %v, want %v", got, test.want)
			}
		})
	}
}

func setFilterRowBit(columns [][]uint64, column int, row uint64) {
	columns[column][row/64] |= uint64(1) << (row % 64)
}
