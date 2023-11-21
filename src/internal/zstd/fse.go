// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package zstd

import (
	"math/bits"
)

// fseEntry is one entry in an FSE table.
type fseEntry struct {
	sym  uint8  // value that this entry records
	bits uint8  // number of bits to read to determine next state
	base uint16 // add those bits to this state to get the next state
}

// readFSE reads an FSE table from data starting at off.
// maxSym is the maximum symbol value.
// maxBits is the maximum number of bits permitted for symbols in the table.
// The FSE is written into table, which must be at least 1<<maxBits in size.
// This returns the number of bits in the FSE table and the new offset.
// RFC 4.1.1.
func (r *Reader) readFSE(data block, off, maxSym, maxBits int, table []fseEntry) (tableBits, roff int, err error) {
	br := r.makeBitReader(data, off)
	if err := br.moreBits(); err != nil {
		return 0, 0, err
	}

	accuracyLog := int(br.val(4)) + 5
	if accuracyLog > maxBits {
		return 0, 0, br.makeError("FSE accuracy log too large")
	}

	// The number of remaining probabilities, plus 1.
	// This determines the number of bits to be read for the next value.
	remaining := (1 << accuracyLog) + 1

	// The current difference between small and large values,
	// which depends on the number of remaining values.
	// Small values use 1 less bit.
	threshold := 1 << accuracyLog

	// The number of bits needed to compute threshold.
	bitsNeeded := accuracyLog + 1

	// The next character value.
	sym := 0

	// Whether the last count was 0.
	prev0 := false

	var norm [256]int16

	for remaining > 1 && sym <= maxSym {
		if err := br.moreBits(); err != nil {
			return 0, 0, err
		}

		if prev0 {
			// Previous count was 0, so there is a 2-bit
			// repeat flag. If the 2-bit flag is 0b11,
			// it adds 3 and then there is another repeat flag.
			zsym := sym
			for (br.bits & 0xfff) == 0xfff {
				zsym += 3 * 6
				br.bits >>= 12
				br.cnt -= 12
				if err := br.moreBits(); err != nil {
					return 0, 0, err
				}
			}
			for (br.bits & 3) == 3 {
				zsym += 3
				br.bits >>= 2
				br.cnt -= 2
				if err := br.moreBits(); err != nil {
					return 0, 0, err
				}
			}

			// We have at least 14 bits here,
			// no need to call moreBits

			zsym += int(br.val(2))

			if zsym > maxSym {
				return 0, 0, br.makeError("FSE symbol index overflow")
			}

			for ; sym < zsym; sym++ {
				norm[uint8(sym)] = 0
			}

			prev0 = false
			continue
		}

		max := (2*threshold - 1) - remaining
		var count int
		if int(br.bits&uint32(threshold-1)) < max {
			// A small value.
			count = int(br.bits & uint32((threshold - 1)))
			br.bits >>= bitsNeeded - 1
			br.cnt -= uint32(bitsNeeded - 1)
		} else {
			// A large value.
			count = int(br.bits & uint32((2*threshold - 1)))
			if count >= threshold {
				count -= max
			}
			br.bits >>= bitsNeeded
			br.cnt -= uint32(bitsNeeded)
		}

		count--
		if count >= 0 {
			remaining -= count
		} else {
			remaining--
		}
		if sym >= 256 {
			return 0, 0, br.makeError("FSE sym overflow")
		}
		norm[uint8(sym)] = int16(count)
		sym++

		prev0 = count == 0

		for remaining < threshold {
			bitsNeeded--
			threshold >>= 1
		}
	}

	if remaining != 1 {
		return 0, 0, br.makeError("too many symbols in FSE table")
	}

	for ; sym <= maxSym; sym++ {
		norm[uint8(sym)] = 0
	}

	br.backup()

	if err := r.buildFSE(off, norm[:maxSym+1], table, accuracyLog); err != nil {
		return 0, 0, err
	}

	return accuracyLog, int(br.off), nil
}

// buildFSE builds an FSE decoding table from a list of probabilities.
// The probabilities are in norm. next is scratch space. The number of bits
// in the table is tableBits.
func (r *Reader) buildFSE(off int, norm []int16, table []fseEntry, tableBits int) error {
	tableSize := 1 << tableBits
	highThreshold := tableSize - 1

	var next [256]uint16

	for i, n := range norm {
		if n >= 0 {
			next[uint8(i)] = uint16(n)
		} else {
			table[highThreshold].sym = uint8(i)
			highThreshold--
			next[uint8(i)] = 1
		}
	}

	pos := 0
	step := (tableSize >> 1) + (tableSize >> 3) + 3
	mask := tableSize - 1
	for i, n := range norm {
		for j := 0; j < int(n); j++ {
			table[pos].sym = uint8(i)
			pos = (pos + step) & mask
			for pos > highThreshold {
				pos = (pos + step) & mask
			}
		}
	}
	if pos != 0 {
		return r.makeError(off, "FSE count error")
	}

	for i := 0; i < tableSize; i++ {
		sym := table[i].sym
		nextState := next[sym]
		next[sym]++

		if nextState == 0 {
			return r.makeError(off, "FSE state error")
		}

		highBit := 15 - bits.LeadingZeros16(nextState)

		bits := tableBits - highBit
		table[i].bits = uint8(bits)
		table[i].base = (nextState << bits) - uint16(tableSize)
	}

	return nil
}

// fseBaselineEntry is an entry in an FSE baseline table.
// We use these for literal/match/length values.
// Those require mapping the symbol to a baseline value,
// and then reading zero or more bits and adding the value to the baseline.
// Rather than looking these up in separate tables,
// we convert the FSE table to an FSE baseline table.
type fseBaselineEntry struct {
	baseline uint32 // baseline for value that this entry represents
	basebits uint8  // number of bits to read to add to baseline
	bits     uint8  // number of bits to read to determine next state
	base     uint16 // add the bits to this base to get the next state
}

// Given a literal length code, we need to read a number of bits and
// add that to a baseline. For states 0 to 15 the baseline is the
// state and the number of bits is zero. RFC 3.1.1.3.2.1.1.

const literalLengthOffset = 16

var literalLengthBase = []uint32{
	16 | (1 << 24),
	18 | (1 << 24),
	20 | (1 << 24),
	22 | (1 << 24),
	24 | (2 << 24),
	28 | (2 << 24),
	32 | (3 << 24),
	40 | (3 << 24),
	48 | (4 << 24),
	64 | (6 << 24),
	128 | (7 << 24),
	256 | (8 << 24),
	512 | (9 << 24),
	1024 | (10 << 24),
	2048 | (11 << 24),
	4096 | (12 << 24),
	8192 | (13 << 24),
	16384 | (14 << 24),
	32768 | (15 << 24),
	65536 | (16 << 24),
}

// makeLiteralBaselineFSE converts the literal length fseTable to baselineTable.
func (r *Reader) makeLiteralBaselineFSE(off int, fseTable []fseEntry, baselineTable []fseBaselineEntry) error {
	for i, e := range fseTable {
		be := fseBaselineEntry{
			bits: e.bits,
			base: e.base,
		}
		if e.sym < literalLengthOffset {
			be.baseline = uint32(e.sym)
			be.basebits = 0
		} else {
			if e.sym > 35 {
				return r.makeError(off, "FSE baseline symbol overflow")
			}
			idx := e.sym - literalLengthOffset
			basebits := literalLengthBase[idx]
			be.baseline = basebits & 0xffffff
			be.basebits = uint8(basebits >> 24)
		}
		baselineTable[i] = be
	}
	return nil
}

// makeOffsetBaselineFSE converts the offset length fseTable to baselineTable.
func (r *Reader) makeOffsetBaselineFSE(off int, fseTable []fseEntry, baselineTable []fseBaselineEntry) error {
	for i, e := range fseTable {
		be := fseBaselineEntry{
			bits: e.bits,
			base: e.base,
		}
		if e.sym > 31 {
			return r.makeError(off, "FSE offset symbol overflow")
		}

		// The simple way to write this is
		//     be.baseline = 1 << e.sym
		//     be.basebits = e.sym
		// That would give us an offset value that corresponds to
		// the one described in the RFC. However, for offsets > 3
		// we have to subtract 3. And for offset values 1, 2, 3
		// we use a repeated offset.
		//
		// The baseline is always a power of 2, and is never 0,
		// so for those low values we will see one entry that is
		// baseline 1, basebits 0, and one entry that is baseline 2,
		// basebits 1. All other entries will have baseline >= 4
		// basebits >= 2.
		//
		// So we can check for RFC offset <= 3 by checking for
		// basebits <= 1. That means that we can subtract 3 here
		// and not worry about doing it in the hot loop.

		be.baseline = 1 << e.sym
		if e.sym >= 2 {
			be.baseline -= 3
		}
		be.basebits = e.sym
		baselineTable[i] = be
	}
	return nil
}

// Given a match length code, we need to read a number of bits and add
// that to a baseline. For states 0 to 31 the baseline is state+3 and
// the number of bits is zero. RFC 3.1.1.3.2.1.1.

const matchLengthOffset = 32

var matchLengthBase = []uint32{
	35 | (1 << 24),
	37 | (1 << 24),
	39 | (1 << 24),
	41 | (1 << 24),
	43 | (2 << 24),
	47 | (2 << 24),
	51 | (3 << 24),
	59 | (3 << 24),
	67 | (4 << 24),
	83 | (4 << 24),
	99 | (5 << 24),
	131 | (7 << 24),
	259 | (8 << 24),
	515 | (9 << 24),
	1027 | (10 << 24),
	2051 | (11 << 24),
	4099 | (12 << 24),
	8195 | (13 << 24),
	16387 | (14 << 24),
	32771 | (15 << 24),
	65539 | (16 << 24),
}

// makeMatchBaselineFSE converts the match length fseTable to baselineTable.
func (r *Reader) makeMatchBaselineFSE(off int, fseTable []fseEntry, baselineTable []fseBaselineEntry) error {
	for i, e := range fseTable {
		be := fseBaselineEntry{
			bits: e.bits,
			base: e.base,
		}
		if e.sym < matchLengthOffset {
			be.baseline = uint32(e.sym) + 3
			be.basebits = 0
		} else {
			if e.sym > 52 {
				return r.makeError(off, "FSE baseline symbol overflow")
			}
			idx := e.sym - matchLengthOffset
			basebits := matchLengthBase[idx]
			be.baseline = basebits & 0xffffff
			be.basebits = uint8(basebits >> 24)
		}
		baselineTable[i] = be
	}
	return nil
}

// predefinedLiteralTable is the predefined table to use for literal lengths.
// Generated from table in RFC 3.1.1.3.2.2.1.
// Checked by TestPredefinedTables.
var predefinedLiteralTable = [...]fseBaselineEntry{
	{0, 0, 4, 0}, {0, 0, 4, 16}, {1, 0, 5, 32},
	{3, 0, 5, 0}, {4, 0, 5, 0}, {6, 0, 5, 0},
	{7, 0, 5, 0}, {9, 0, 5, 0}, {10, 0, 5, 0},
	{12, 0, 5, 0}, {14, 0, 6, 0}, {16, 1, 5, 0},
	{20, 1, 5, 0}, {22, 1, 5, 0}, {28, 2, 5, 0},
	{32, 3, 5, 0}, {48, 4, 5, 0}, {64, 6, 5, 32},
	{128, 7, 5, 0}, {256, 8, 6, 0}, {1024, 10, 6, 0},
	{4096, 12, 6, 0}, {0, 0, 4, 32}, {1, 0, 4, 0},
	{2, 0, 5, 0}, {4, 0, 5, 32}, {5, 0, 5, 0},
	{7, 0, 5, 32}, {8, 0, 5, 0}, {10, 0, 5, 32},
	{11, 0, 5, 0}, {13, 0, 6, 0}, {16, 1, 5, 32},
	{18, 1, 5, 0}, {22, 1, 5, 32}, {24, 2, 5, 0},
	{32, 3, 5, 32}, {40, 3, 5, 0}, {64, 6, 4, 0},
	{64, 6, 4, 16}, {128, 7, 5, 32}, {512, 9, 6, 0},
	{2048, 11, 6, 0}, {0, 0, 4, 48}, {1, 0, 4, 16},
	{2, 0, 5, 32}, {3, 0, 5, 32}, {5, 0, 5, 32},
	{6, 0, 5, 32}, {8, 0, 5, 32}, {9, 0, 5, 32},
	{11, 0, 5, 32}, {12, 0, 5, 32}, {15, 0, 6, 0},
	{18, 1, 5, 32}, {20, 1, 5, 32}, {24, 2, 5, 32},
	{28, 2, 5, 32}, {40, 3, 5, 32}, {48, 4, 5, 32},
	{65536, 16, 6, 0}, {32768, 15, 6, 0}, {16384, 14, 6, 0},
	{8192, 13, 6, 0},
}

// predefinedOffsetTable is the predefined table to use for offsets.
// Generated from table in RFC 3.1.1.3.2.2.3.
// Checked by TestPredefinedTables.
var predefinedOffsetTable = [...]fseBaselineEntry{
	{1, 0, 5, 0}, {61, 6, 4, 0}, {509, 9, 5, 0},
	{32765, 15, 5, 0}, {2097149, 21, 5, 0}, {5, 3, 5, 0},
	{125, 7, 4, 0}, {4093, 12, 5, 0}, {262141, 18, 5, 0},
	{8388605, 23, 5, 0}, {29, 5, 5, 0}, {253, 8, 4, 0},
	{16381, 14, 5, 0}, {1048573, 20, 5, 0}, {1, 2, 5, 0},
	{125, 7, 4, 16}, {2045, 11, 5, 0}, {131069, 17, 5, 0},
	{4194301, 22, 5, 0}, {13, 4, 5, 0}, {253, 8, 4, 16},
	{8189, 13, 5, 0}, {524285, 19, 5, 0}, {2, 1, 5, 0},
	{61, 6, 4, 16}, {1021, 10, 5, 0}, {65533, 16, 5, 0},
	{268435453, 28, 5, 0}, {134217725, 27, 5, 0}, {67108861, 26, 5, 0},
	{33554429, 25, 5, 0}, {16777213, 24, 5, 0},
}

// predefinedMatchTable is the predefined table to use for match lengths.
// Generated from table in RFC 3.1.1.3.2.2.2.
// Checked by TestPredefinedTables.
var predefinedMatchTable = [...]fseBaselineEntry{
	{3, 0, 6, 0}, {4, 0, 4, 0}, {5, 0, 5, 32},
	{6, 0, 5, 0}, {8, 0, 5, 0}, {9, 0, 5, 0},
	{11, 0, 5, 0}, {13, 0, 6, 0}, {16, 0, 6, 0},
	{19, 0, 6, 0}, {22, 0, 6, 0}, {25, 0, 6, 0},
	{28, 0, 6, 0}, {31, 0, 6, 0}, {34, 0, 6, 0},
	{37, 1, 6, 0}, {41, 1, 6, 0}, {47, 2, 6, 0},
	{59, 3, 6, 0}, {83, 4, 6, 0}, {131, 7, 6, 0},
	{515, 9, 6, 0}, {4, 0, 4, 16}, {5, 0, 4, 0},
	{6, 0, 5, 32}, {7, 0, 5, 0}, {9, 0, 5, 32},
	{10, 0, 5, 0}, {12, 0, 6, 0}, {15, 0, 6, 0},
	{18, 0, 6, 0}, {21, 0, 6, 0}, {24, 0, 6, 0},
	{27, 0, 6, 0}, {30, 0, 6, 0}, {33, 0, 6, 0},
	{35, 1, 6, 0}, {39, 1, 6, 0}, {43, 2, 6, 0},
	{51, 3, 6, 0}, {67, 4, 6, 0}, {99, 5, 6, 0},
	{259, 8, 6, 0}, {4, 0, 4, 32}, {4, 0, 4, 48},
	{5, 0, 4, 16}, {7, 0, 5, 32}, {8, 0, 5, 32},
	{10, 0, 5, 32}, {11, 0, 5, 32}, {14, 0, 6, 0},
	{17, 0, 6, 0}, {20, 0, 6, 0}, {23, 0, 6, 0},
	{26, 0, 6, 0}, {29, 0, 6, 0}, {32, 0, 6, 0},
	{65539, 16, 6, 0}, {32771, 15, 6, 0}, {16387, 14, 6, 0},
	{8195, 13, 6, 0}, {4099, 12, 6, 0}, {2051, 11, 6, 0},
	{1027, 10, 6, 0},
}
