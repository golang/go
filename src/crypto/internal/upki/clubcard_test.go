// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package upki

import (
	"bytes"
	_ "embed"
	"errors"
	"testing"

	"golang.org/x/crypto/cryptobyte"
)

const (
	nontrivialCoverageLow   timestamp = 100
	nontrivialCoverageHigh  timestamp = 200
	nontrivialRevokedSerial           = 0x2a
	nontrivialGoodSerial              = 0x2b
)

var (
	nontrivialIssuer      = issuerSPKIHash{1, 2, 3}
	nontrivialOtherIssuer = issuerSPKIHash{4, 5, 6}
	nontrivialLog         = logID{9, 9, 9}

	// nontrivialFilter covers nontrivialLog from timestamps 100 through 200
	// and enrolls two issuers whose serials are listed by nontrivialSerials.
	// Its blocks exercise nontrivial moduli and row offsets.
	//
	//go:embed testdata/nontrivial.filter
	nontrivialFilter []byte
)

// nontrivialSerials lists every serial enrolled in nontrivialFilter for
// issuer and whether it is revoked. Serials outside these lists are outside
// the filter's universe and have no deterministic verdict.
func nontrivialSerials(issuer issuerSPKIHash) (revoked, good [][]byte) {
	switch issuer {
	case nontrivialIssuer:
		revoked = append(revoked, []byte{nontrivialRevokedSerial})
		for i := range 256 {
			revoked = append(revoked, []byte{0x01, byte(i)})
		}
		for i := range 64 {
			revoked = append(revoked, []byte{0x02, byte(i)})
		}
		good = append(good, []byte{nontrivialGoodSerial}, []byte{0x2c}, []byte{0x2d}, []byte{0x2e})
		for i := range 32 {
			good = append(good, []byte{0x03, byte(i)})
		}
	case nontrivialOtherIssuer:
		for i := range 16 {
			revoked = append(revoked, []byte{0xb0, byte(i)})
		}
		for i := range 8 {
			good = append(good, []byte{0xb1, byte(i)})
		}
	}

	return revoked, good
}

func TestParseFilter(t *testing.T) {
	t.Parallel()

	// Exercise the parser and query path together using a fixture produced by
	// an independent implementation of the clubcard construction algorithm.
	parsed, err := parseFilter(nontrivialFilter)
	if err != nil {
		t.Fatalf("parseFilter: %v", err)
	}

	// Guard the fixture's shape (see nontrivialFilter): a block whose row
	// depends on t, and blocks that do not all start at row 0.
	if meta := parsed.filter.blocks[nontrivialIssuer]; meta.hModulus < 2 || meta.hRank < 2 {
		t.Fatalf("nontrivialIssuer block has hModulus %d and hRank %d; want a modulus above 1 and at least two columns",
			meta.hModulus, meta.hRank)
	}
	var hOffset, gOffset bool
	for _, meta := range parsed.filter.blocks {
		hOffset = hOffset || meta.hOffset != 0
		gOffset = gOffset || meta.gOffset != 0
	}
	if !hOffset || !gOffset {
		t.Fatalf("fixture has a nonzero hOffset: %v, gOffset: %v; want both", hOffset, gOffset)
	}

	covered := SCT{logID: nontrivialLog, timestamp: nontrivialCoverageLow}
	tests := []struct {
		name string
		key  lookupKey
		scts []SCT
		want filterStatus
	}{
		{
			name: "revoked",
			key:  newLookupKey(nontrivialIssuer, []byte{nontrivialRevokedSerial}),
			scts: []SCT{covered},
			want: statusRevoked,
		},
		{
			name: "not revoked",
			key:  newLookupKey(nontrivialIssuer, []byte{nontrivialGoodSerial}),
			scts: []SCT{covered},
			want: statusGood,
		},
		{
			name: "coverage upper bound",
			key:  newLookupKey(nontrivialIssuer, []byte{nontrivialRevokedSerial}),
			scts: []SCT{{logID: nontrivialLog, timestamp: nontrivialCoverageHigh}},
			want: statusRevoked,
		},
		{
			name: "timestamp not covered",
			key:  newLookupKey(nontrivialIssuer, []byte{nontrivialRevokedSerial}),
			scts: []SCT{{logID: nontrivialLog, timestamp: nontrivialCoverageHigh + 1}},
			want: statusNotCovered,
		},
		{
			name: "log not covered",
			key:  newLookupKey(nontrivialIssuer, []byte{nontrivialRevokedSerial}),
			scts: []SCT{{logID: logID{0xff}, timestamp: nontrivialCoverageLow}},
			want: statusNotCovered,
		},
		{
			name: "issuer not enrolled",
			key:  newLookupKey(issuerSPKIHash{0xff}, []byte{nontrivialRevokedSerial}),
			scts: []SCT{covered},
			want: statusNotEnrolled,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			if got := parsed.contains(&test.key, test.scts); got != test.want {
				t.Errorf("contains = %v, want %v", got, test.want)
			}
		})
	}

	// Every enrolled serial must get the verdict the reference builder gave
	// it. Across hundreds of serials in two blocks, a wrong row selection
	// (the wrong hash limb for t, or a mishandled block offset) cannot pass
	// by chance.
	t.Run("all enrolled serials", func(t *testing.T) {
		t.Parallel()
		for _, issuer := range []issuerSPKIHash{nontrivialIssuer, nontrivialOtherIssuer} {
			revoked, good := nontrivialSerials(issuer)
			for _, serial := range revoked {
				key := newLookupKey(issuer, serial)
				if got := parsed.contains(&key, []SCT{covered}); got != statusRevoked {
					t.Errorf("issuer %x serial %x: contains = %v, want statusRevoked", issuer[:3], serial, got)
				}
			}
			for _, serial := range good {
				key := newLookupKey(issuer, serial)
				if got := parsed.contains(&key, []SCT{covered}); got != statusGood {
					t.Errorf("issuer %x serial %x: contains = %v, want statusGood", issuer[:3], serial, got)
				}
			}
		}
	})
}

// TestParseFilterRankZero covers the compact encoding of an empty block
// (§3.1 "Rank zero blocks"): no filter columns, no exceptions, and every
// query answers not revoked, or revoked when the block is inverted. CRLite
// uses such blocks for issuers with no revocations and, inverted, for
// issuers whose certificates are all revoked, so the parser and query path
// must accept them even though the filter carries no ribbon data.
func TestParseFilterRankZero(t *testing.T) {
	t.Parallel()

	const (
		coverageLow  timestamp = 10
		coverageHigh timestamp = 20
		queryTime    timestamp = 15
	)
	issuer := issuerSPKIHash{1}
	log := logID{2}
	for _, inverted := range []bool{false, true} {
		var invertedByte uint8
		if inverted {
			invertedByte = 1
		}
		parsed, err := parseFilter(encodeTestFilter(
			[]testCoverage{{logID: log, low: coverageLow, high: coverageHigh}},
			[]testBlock{{issuer: issuer, inverted: invertedByte}},
			0,
		))
		if err != nil {
			t.Fatalf("inverted=%v: parseFilter: %v", inverted, err)
		}
		key := newLookupKey(issuer, []byte("serial"))
		want := statusGood
		if inverted {
			want = statusRevoked
		}
		if got := parsed.contains(&key, []SCT{{logID: log, timestamp: queryTime}}); got != want {
			t.Errorf("inverted=%v: contains = %v, want %v", inverted, got, want)
		}
	}
}

func TestParseFilterMalformed(t *testing.T) {
	t.Parallel()

	const (
		coverageLow  timestamp = 10
		coverageHigh timestamp = 20
	)
	issuer := issuerSPKIHash{1}
	log := logID{2}
	valid := encodeTestFilter(
		[]testCoverage{{logID: log, low: coverageLow, high: coverageHigh}},
		[]testBlock{{issuer: issuer}},
		0,
	)
	version := bytes.Clone(valid)
	version[0] = 5
	reserved := bytes.Clone(valid)
	reserved[1] = 1
	tests := []struct {
		name string
		data []byte
		want error
	}{
		{"unsupported version", version, errUnsupportedFormat},
		{"nonzero reserved byte", reserved, errDeserialize},
		{"trailing data", append(bytes.Clone(valid), 0), errDeserialize},
		{"duplicate coverage log", encodeTestFilter(
			[]testCoverage{
				{logID: log, low: coverageLow, high: coverageHigh},
				{logID: log, low: coverageHigh + 10, high: coverageHigh + 20},
			},
			nil, 0), errDeserialize},
		{"duplicate block", encodeTestFilter(nil,
			[]testBlock{{issuer: issuer}, {issuer: issuer}}, 0), errDeserialize},
		{"invalid inverted flag", encodeTestFilter(nil,
			[]testBlock{{issuer: issuer, inverted: 2}}, 0), errDeserialize},
		{"rank exceeds columns", encodeTestFilter(nil,
			[]testBlock{{issuer: issuer, rank: 1}}, 0), errDeserialize},
		{"inflated coverage count", []byte{4, 0, 0xff, 0xff}, errDeserialize},
		{"inflated block count", []byte{4, 0, 0, 0, 0xff, 0xff, 0xff, 0xff}, errDeserialize},
		{"truncated header", valid[:1], errDeserialize},
		{"truncated coverage", valid[:10], errDeserialize},
		{"truncated block", valid[:len(valid)-8], errDeserialize},
		{"truncated exact column", valid[:len(valid)-1], errDeserialize},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			if _, err := parseFilter(test.data); !errors.Is(err, test.want) {
				t.Fatalf("parseFilter error = %v, want %v", err, test.want)
			}
		})
	}
}

func FuzzParseFilter(f *testing.F) {
	f.Add(nontrivialFilter)
	f.Add(encodeTestFilter(
		[]testCoverage{{logID: logID{1}, low: 10, high: 20}},
		[]testBlock{{issuer: issuerSPKIHash{2}, inverted: 1}},
		0,
	))
	// A bare v4 version and reserved byte reaches the first required field
	// without carrying any of the following coverage or filter structure.
	f.Add([]byte{4, 0})

	f.Fuzz(func(t *testing.T, data []byte) {
		parsed, err := parseFilter(data)
		if err != nil {
			return
		}

		// Probe accepted filters using their coverage bounds, several issuers,
		// and several serial shapes so unsafe metadata triggers a panic.
		var scts []SCT
		for id, interval := range parsed.coverage {
			scts = append(scts,
				SCT{logID: id, timestamp: interval.low},
				SCT{logID: id, timestamp: interval.high})
			if len(scts) >= 8 {
				break
			}
		}
		scts = append(scts, SCT{logID: logID{0xff}})

		queried := 0
		for issuer := range parsed.filter.blocks {
			for _, serial := range [][]byte{nil, {1}, {0xab, 0xcd}} {
				key := newLookupKey(issuer, serial)
				parsed.contains(&key, scts)
			}
			queried++
			if queried == 4 {
				break
			}
		}
		key := newLookupKey(issuerSPKIHash{0x42}, []byte{1})
		parsed.contains(&key, scts)
	})
}

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

type testCoverage struct {
	logID     logID
	low, high timestamp
}

type testBlock struct {
	issuer     issuerSPKIHash
	hModulus   uint32
	rank       uint8
	hOffset    uint32
	gModulus   uint32
	gOffset    uint32
	inverted   uint8
	exceptions [][]byte
}

func encodeTestFilter(coverage []testCoverage, blocks []testBlock, columns uint8) []byte {
	// This mirrors the C2SP upki-revocation v1.0.0 Clubcard wire format,
	// restricted to the fields and zero-valued columns needed by these tests.
	var builder cryptobyte.Builder
	builder.AddUint8(4)
	builder.AddUint8(0)
	builder.AddUint16(uint16(len(coverage)))
	for _, item := range coverage {
		builder.AddBytes(item.logID[:])
		builder.AddUint64(uint64(item.low))
		builder.AddUint64(uint64(item.high))
	}
	builder.AddUint32(uint32(len(blocks)))
	for _, block := range blocks {
		builder.AddBytes(block.issuer[:])
		builder.AddUint32(block.hModulus)
		builder.AddUint8(block.rank)
		builder.AddUint32(block.hOffset)
		builder.AddUint32(block.gModulus)
		builder.AddUint32(block.gOffset)
		builder.AddUint8(block.inverted)
		builder.AddUint16(uint16(len(block.exceptions)))
		for _, serial := range block.exceptions {
			builder.AddUint8LengthPrefixed(func(builder *cryptobyte.Builder) {
				builder.AddBytes(serial)
			})
		}
	}
	builder.AddUint8(columns)
	for range columns {
		builder.AddUint32(0)
	}
	builder.AddUint32(0)

	return builder.BytesOrPanic()
}
