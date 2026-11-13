// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package upki

// This file implements the in-memory query side of the clubcard membership
// test.
//
// The data structure and membership test are described in J. M. Schanck,
// "Clubcards for the WebPKI: Smaller Certificate Revocation Tests in Theory
// and Practice," 2025 IEEE Symposium on Security and Privacy (SP), San
// Francisco, CA, USA, 2025.
//
// Section references of the form §N or §N.M throughout this package refer
// to that paper, accessible at
// <https://jmschanck.info/papers/20250327-clubcard.pdf>
//
// This implementation is narrowly scoped to 32-byte block identifiers
// (matching issuer SPKI hash) and with ribbon width w = 256 (W = 4 64-bit
// limbs), matching CRLite. Construction of clubcards is out of scope.
//
// Notably the storage layout differs from the paper's row-major X matrix.
// Instead, X is stored column-major as one bit-packed []uint64 per column,
// matching the version 4 filter encoding specified by C2SP
// upki-revocation v1.0.0[1].
//
// [1]: https://c2sp.org/upki-revocation@v1.0.0#revocation-filter-files

import (
	"encoding/binary"
	"math/bits"
)

// filter is an in-memory clubcard filter.
//
// `x[c]` holds column `c` of the approximate membership query filter,
// bit-packed little-endian. e.g. bit `i` lives in `x[c][i/64]` at position
// `i % 64`.
//
// y holds the 1-bit retrieval function vector in the same packing.
//
// blocks is keyed by 32-byte issuer SPKI hash (§4.2 BlockId).
type filter struct {
	x      [][]uint64
	y      []uint64
	blocks map[[32]byte]blockMeta
}

// contains evaluates the §3 Query algorithm for an element `u` described
// by `a` = SHA256(IssuerSPKIHash || SerialNumber) along with the raw serial.
//
// Deferring the blocks lookup to the caller keeps the
// "is the issuer enrolled?" decision in the revocation filter layer.
//
// The returned bool already accounts for blockMeta.inverted.
func (f *filter) contains(meta *blockMeta, a [32]byte, serial []byte) bool {
	// Empty block (`Ri` = ∅): `u` ∉ `Ri` for every `u`. Inversion flips to
	// "every `u` is in `Ri`" (§3.1, "compact encoding of empty blocks").
	//
	// This check also short-circuits the `t % HModulus` divide-by-zero
	// below.
	if meta.hModulus == 0 {
		return meta.inverted
	}

	aLimbs := bytesToLimbs(a)
	// Force a leading 1 in the coefficient vector. Ribbon retrieval
	// requires the equation passed to InsertEquation to be aligned
	// (low bit set) for construction's Gaussian elimination to work
	// (§2.4 "aligned representative", §4.3).
	aLimbs[0] |= 1
	t := aLimbs[3] // §4.3 step 3: high 64 bits, little-endian => limb 3

	hRow := meta.hOffset + (t % meta.hModulus)
	for i := uint8(0); i < meta.hRank; i++ {
		if !bitsXorIsZero(f.x[i], hRow, aLimbs) {
			return meta.inverted
		}
	}

	// The exact filter is always evaluated, at row gOffset when gModulus is
	// 0, matching the reference implementation (clubcard-crlite's
	// CRLiteQuery::as_query reduces modulo max(1, m)). The builder never
	// emits a block with hModulus != 0 and gModulus == 0, so this only
	// keeps the two paths identical for every input.
	gRow := meta.gOffset + (t % max(1, meta.gModulus))
	if !bitsXorIsZero(f.y, gRow, aLimbs) {
		return meta.inverted
	}

	if _, isException := meta.exceptions[string(serial)]; isException {
		return meta.inverted
	}

	return !meta.inverted
}

// blockMeta is BlockMeta(i) from §3. It is the data needed to evaluate
// `Ri`-membership for one block.
//
// hOffset and hModulus parameterise the §4.3 CRLiteRibbonHash for `hi`,
// and gOffset and gModulus do the same for `gi`. The §3 per-block `δi` row
// offset is pre-folded into hOffset and gOffset by the wire format, so no
// separate bit-offset field appears here.
//
// hRank is `ki`: the number of filter columns this block's `hi` check
// probes. Per §3 "Omitting zeros", X has a staircase of zero coefficients
// in its bottom-right, so trailing columns may not extend to every
// block. Instead, hRank is the source of truth for which columns apply.
type blockMeta struct {
	hOffset  uint64
	hModulus uint64
	hRank    uint8

	gOffset  uint64
	gModulus uint64

	// exceptions is `Fi`: serials known to be insertion failures from
	// the construction of Y.
	//
	// Stored as raw serial bytes (no DER tag/length wrapper), converted to a
	// string (no []byte keys in Go maps).
	exceptions map[string]struct{}

	// inverted swaps the member/non-member returns from Query.
	// See §3.1 "Rank zero blocks".
	inverted bool
}

// bytesToLimbs reads `a` as four little-endian uint64 limbs.
//
// Limb 3 is the "high 64 bits" used by §4.3 CRLiteRibbonHash to derive `t`.
func bytesToLimbs(a [32]byte) [ribbonWidth]uint64 {
	return [ribbonWidth]uint64{
		binary.LittleEndian.Uint64(a[0:8]),
		binary.LittleEndian.Uint64(a[8:16]),
		binary.LittleEndian.Uint64(a[16:24]),
		binary.LittleEndian.Uint64(a[24:32]),
	}
}

// bitsXorIsZero reports whether the dot product of coef with the
// 256 bits of column starting at rowOffset is 0 over GF(2).
//
// Rows past the end of column read as zero.
func bitsXorIsZero(column []uint64, rowOffset uint64, coef [ribbonWidth]uint64) bool {
	limb := rowOffset / 64
	shift := rowOffset % 64

	var r uint64
	for i := limb; i < min(uint64(len(column)), limb+ribbonWidth); i++ {
		tmp := column[i] >> shift
		if shift != 0 && i+1 < uint64(len(column)) {
			tmp |= column[i+1] << (64 - shift)
		}
		r ^= tmp & coef[i-limb]
	}

	return bits.OnesCount64(r)&1 == 0
}

// ribbonWidth is the §4.3 ribbon width in 64-bit limbs (w = 256).
const ribbonWidth = 4
