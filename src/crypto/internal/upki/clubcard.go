// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package upki

import (
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"math/bits"

	"golang.org/x/crypto/cryptobyte"
)

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

var (
	errUnsupportedFormat = errors.New("upki: unsupported clubcard version")
	errDeserialize       = errors.New("upki: failed to deserialize clubcard")
)

// revocationFilter is a parsed CRLite filter.
//
// Use parseFilter to construct one and contains to look up revocation status.
type revocationFilter struct {
	coverage coverage
	filter   filter
}

// contains evaluates the revocation filter against key, trying each SCT's
// (log ID, timestamp) pair until one falls within a covered interval.
//
// If no covered interval is found, the result is statusNotCovered.
func (c *revocationFilter) contains(key *lookupKey, scts []SCT) filterStatus {
	for _, sct := range scts {
		status := c.containsAt(key, sct)
		if status == statusNotCovered {
			continue
		}

		return status
	}

	return statusNotCovered
}

// lookupKey is the (issuer, serial) pair identifying a certificate, together
// with the precomputed SHA-256(issuer || serial) used by the filter hashing.
type lookupKey struct {
	issuer           issuerSPKIHash
	serial           []byte
	issuerSerialHash [32]byte
}

// issuerSPKIHash is the SHA-256 hash of an issuer's SubjectPublicKeyInfo.
type issuerSPKIHash [32]byte

// newLookupKey combines the issuer SPKI hash and serial into a lookup key.
func newLookupKey(issuer issuerSPKIHash, serial []byte) lookupKey {
	h := sha256.New()
	h.Write(issuer[:])
	h.Write(serial)
	var sum [32]byte
	h.Sum(sum[:0])

	return lookupKey{issuer: issuer, serial: serial, issuerSerialHash: sum}
}

// filterStatus is a [revocationFilter.contains] membership result.
type filterStatus int

const (
	// statusGood indicates the certificate is not revoked.
	statusGood filterStatus = iota
	// statusNotCovered indicates none of the supplied SCTs falls within
	// a (CT log, timestamp interval) pair that this filter covers.
	//
	// Revocation status can't be determined without updated filters.
	statusNotCovered
	// statusNotEnrolled indicates the certificate's issuer is not in
	// the filter's block index.
	//
	// Revocation status can't be determined without updated filters.
	statusNotEnrolled
	// statusRevoked indicates the certificate has been revoked.
	statusRevoked
)

// containsAt is a single (log ID, timestamp) membership probe.
func (c *revocationFilter) containsAt(key *lookupKey, sct SCT) filterStatus {
	iv, ok := c.coverage[sct.logID]
	if !ok || sct.timestamp < iv.low || sct.timestamp > iv.high {
		return statusNotCovered
	}

	meta, ok := c.filter.blocks[key.issuer]
	if !ok {
		return statusNotEnrolled
	}

	if c.filter.contains(&meta, key.issuerSerialHash, key.serial) {
		return statusRevoked
	}

	return statusGood
}

// parseFilter parses a serialized CRLite clubcard.
//
// Only the V4 encoding is supported.
//
// The top-level wire format is:
//
//	struct {
//	    uint8           version;
//	    uint8           reserved0;           // MUST be 0
//	    CRLiteCoverage  coverage;
//	    ClubcardIndex   index;
//	    uint8           approx_filter_count;
//	    FilterColumn    approx_filter[approx_filter_count];
//	    FilterColumn    exact_filter;
//	} Clubcard;
//
// where `FilterColumn` is `uint64 words<count>`, a uint32 word-count prefix
// followed by that many big-endian u64 words. The exact filter is encoded
// with its own uint32 word-count prefix.
func parseFilter(data []byte) (*revocationFilter, error) {
	s := cryptobyte.String(data)

	// The version is a u8 followed by a reserved0 u8 that MUST be zero.
	var version, reserved0 uint8
	if !s.ReadUint8(&version) || !s.ReadUint8(&reserved0) {
		return nil, fmt.Errorf("%w: version", errDeserialize)
	}

	// v4 is the only encoding tag this package recognizes.
	if version != 4 {
		return nil, fmt.Errorf("%w: %d", errUnsupportedFormat, version)
	}
	// A future format version may repurpose reserved0 to signal additional
	// metadata, but v4 requires it to be zero.
	if reserved0 != 0 {
		return nil, fmt.Errorf("%w: non-zero reserved0 byte %d", errDeserialize, reserved0)
	}

	cov, err := readCoverage(&s)
	if err != nil {
		return nil, err
	}

	filter, err := readFilter(&s)
	if err != nil {
		return nil, err
	}

	if !s.Empty() {
		return nil, fmt.Errorf("%w: %d trailing bytes", errDeserialize, len(s))
	}

	return &revocationFilter{coverage: cov, filter: filter}, nil
}

// readCoverage parses the universe coverage map.
//
//	struct {
//	    LogId             log_id;
//	    TimestampInterval interval;
//	} Coverage;
//
//	Coverage coverage<count>;   // uint16 count, then `count` entries
func readCoverage(s *cryptobyte.String) (coverage, error) {
	var count uint16
	if !s.ReadUint16(&count) {
		return nil, fmt.Errorf("%w: coverage count", errDeserialize)
	}

	// Each entry is a 32-byte log id plus two u64 timestamps. Reject counts
	// that promise more data than remains before allocating.
	const coverageEntrySize = 32 + 8 + 8
	if uint64(count)*coverageEntrySize > uint64(len(*s)) {
		return nil, fmt.Errorf("%w: coverage count %d exceeds input", errDeserialize, count)
	}

	cov := make(coverage, count)
	for i := uint16(0); i < count; i++ {
		id, err := readLogId(s)
		if err != nil {
			return nil, err
		}

		iv, err := readTimestampInterval(s)
		if err != nil {
			return nil, err
		}

		// The spec requires coverage entries to have distinct log ids. A
		// duplicate would otherwise silently overwrite an earlier interval.
		if _, dup := cov[id]; dup {
			return nil, fmt.Errorf("%w: duplicate coverage log id", errDeserialize)
		}

		cov[id] = iv
	}

	return cov, nil
}

// coverage maps each CT log id to the inclusive timestamp interval the
// filter is known to cover.
type coverage map[logID]timestampInterval

// timestampInterval is a closed pair of CT timestamps.
type timestampInterval struct {
	low  timestamp
	high timestamp
}

// readLogId parses a CT log identifier.
//
//	opaque LogId[32]; // fixed-width SHA-256 digest, no length prefix
func readLogId(s *cryptobyte.String) (logID, error) {
	var id logID
	if !s.CopyBytes(id[:]) {
		return id, fmt.Errorf("%w: log id", errDeserialize)
	}

	return id, nil
}

// readTimestampInterval parses a closed pair of CT timestamps.
//
//	struct {
//	    Timestamp low;
//	    Timestamp high;
//	} TimestampInterval;
//
// Each Timestamp is a uint64, big-endian.
func readTimestampInterval(s *cryptobyte.String) (timestampInterval, error) {
	var low, high uint64
	if !s.ReadUint64(&low) || !s.ReadUint64(&high) {
		return timestampInterval{}, fmt.Errorf("%w: timestamp interval", errDeserialize)
	}

	return timestampInterval{low: timestamp(low), high: timestamp(high)}, nil
}

// readFilter parses the block index + filter columns of a V4 clubcard.
func readFilter(s *cryptobyte.String) (filter, error) {
	var f filter

	blocks, err := readBlocks(s)
	if err != nil {
		return f, err
	}
	f.blocks = blocks

	var columnCount uint8
	if !s.ReadUint8(&columnCount) {
		return f, fmt.Errorf("%w: approx filter column count", errDeserialize)
	}

	f.x = make([][]uint64, columnCount)
	for i := range f.x {
		col, err := readU64Seq(s)
		if err != nil {
			return f, fmt.Errorf("approx filter column %d/%d: %w", i, columnCount, err)
		}

		f.x[i] = col
	}

	f.y, err = readU64Seq(s)
	if err != nil {
		return f, fmt.Errorf("exact filter: %w", err)
	}

	if err := validateBlocks(&f); err != nil {
		return f, err
	}

	return f, nil
}

// readU64Seq parses a `u64 items<count>` sequence: a uint32 word-count prefix
// followed by that many big-endian u64s.
func readU64Seq(s *cryptobyte.String) ([]uint64, error) {
	var count uint32
	if !s.ReadUint32(&count) {
		return nil, fmt.Errorf("%w: u64 seq count", errDeserialize)
	}

	// Reject counts that promise more data than remains before allocating,
	// so a short malicious input can't provoke a huge allocation.
	if uint64(count)*8 > uint64(len(*s)) {
		return nil, fmt.Errorf("%w: u64 seq count %d exceeds input", errDeserialize, count)
	}

	items := make([]uint64, count)
	for i := range items {
		if !s.ReadUint64(&items[i]) {
			return nil, fmt.Errorf("%w: u64 seq item %d/%d", errDeserialize, i, count)
		}
	}

	return items, nil
}

// readBlocks parses the per-block metadata table.
//
//	struct {
//	    opaque    block_id[32];
//	    BlockMeta meta;
//	} IndexEntry;
//
//	IndexEntry index<count>;   // uint32 count, then `count` entries
//
// The wire format encodes an `approx_filter_rank` field per block: the
// number of `X` columns this block's `h` check probes. Ranks vary between
// blocks because the CRLite construction packs progressively smaller
// block ranges into the trailing columns of `X`.
func readBlocks(s *cryptobyte.String) (map[[32]byte]blockMeta, error) {
	var count uint32
	if !s.ReadUint32(&count) {
		return nil, fmt.Errorf("%w: index count", errDeserialize)
	}

	// Each entry is at least a 32-byte block id plus BlockMeta's fixed
	// fields (see readBlockMeta). Reject counts that promise more data
	// than remains before allocating.
	const minIndexEntrySize = 32 + // block_id
		4 + // approx_filter_m (u32)
		1 + // approx_filter_rank (u8)
		4 + // approx_filter_offset (u32)
		4 + // exact_filter_m (u32)
		4 + // exact_filter_offset (u32)
		1 + // inverted (u8)
		2 // exception count (u16)
	if uint64(count)*minIndexEntrySize > uint64(len(*s)) {
		return nil, fmt.Errorf("%w: index count %d exceeds input", errDeserialize, count)
	}

	blocks := make(map[[32]byte]blockMeta, count)
	for i := uint32(0); i < count; i++ {
		var id [32]byte
		if !s.CopyBytes(id[:]) {
			return nil, fmt.Errorf("%w: index block id %d/%d", errDeserialize, i, count)
		}

		meta, err := readBlockMeta(s)
		if err != nil {
			return nil, err
		}

		if _, dup := blocks[id]; dup {
			return nil, fmt.Errorf("%w: duplicate index block id", errDeserialize)
		}

		blocks[id] = meta
	}

	return blocks, nil
}

// readBlockMeta parses one block's metadata.
//
//	struct {
//	    uint8  len;
//	    opaque serial[len];
//	} Exception;                        // serial as opaque<0..2^8-1>
//
//	struct {
//	    uint32    approx_filter_m;       // -> HModulus
//	    uint8     approx_filter_rank;    // -> HRank
//	    uint32    approx_filter_offset;  // -> HOffset
//	    uint32    exact_filter_m;        // -> GModulus
//	    uint32    exact_filter_offset;   // -> GOffset
//	    uint8     inverted;
//	    Exception exceptions<count>;     // uint16 count, then `count` serials
//	} BlockMeta;
func readBlockMeta(s *cryptobyte.String) (blockMeta, error) {
	var (
		hMod, hOff, gMod, gOff uint32
		rank, inverted         uint8
	)
	if !s.ReadUint32(&hMod) ||
		!s.ReadUint8(&rank) ||
		!s.ReadUint32(&hOff) ||
		!s.ReadUint32(&gMod) ||
		!s.ReadUint32(&gOff) ||
		!s.ReadUint8(&inverted) {
		return blockMeta{}, fmt.Errorf("%w: block meta header", errDeserialize)
	}

	// inverted is a polarity flag that MUST be either 0 or 1.
	if inverted > 1 {
		return blockMeta{}, fmt.Errorf("%w: inverted flag %d not 0 or 1", errDeserialize, inverted)
	}

	var count uint16
	if !s.ReadUint16(&count) {
		return blockMeta{}, fmt.Errorf("%w: block meta exception count", errDeserialize)
	}

	// Each exception requires at least its one-byte length prefix.
	if int(count) > len(*s) {
		return blockMeta{}, fmt.Errorf("%w: block meta exception count %d exceeds input", errDeserialize, count)
	}

	// Most blocks have no exceptions. A nil map reads as empty, so only
	// allocate when there is something to put in it.
	var exceptions map[string]struct{}
	if count > 0 {
		exceptions = make(map[string]struct{}, count)
	}
	for i := uint16(0); i < count; i++ {
		var serial cryptobyte.String
		if !s.ReadUint8LengthPrefixed(&serial) {
			return blockMeta{}, fmt.Errorf("%w: block meta exception %d/%d", errDeserialize, i, count)
		}

		exceptions[string(serial)] = struct{}{}
	}

	return blockMeta{
		hOffset:    uint64(hOff),
		hModulus:   uint64(hMod),
		hRank:      rank,
		gOffset:    uint64(gOff),
		gModulus:   uint64(gMod),
		exceptions: exceptions,
		inverted:   inverted != 0,
	}, nil
}

// validateBlocks checks each block's HRank against the parsed column count
// so [filter.contains] cannot index a column that does not exist.
//
// The spec makes this unconditional: any block whose approx_rank exceeds the
// column count renders the filter malformed. An empty block (HModulus == 0)
// never indexes X at query time (Contains returns early), but the rank is
// still checked so a spec-conformant reader rejects the same files.
//
// Row offsets and moduli need no validation. Contains() treats out-of-range
// rows as zero (see bitsXorIsZero).
func validateBlocks(f *filter) error {
	for id, meta := range f.blocks {
		if int(meta.hRank) > len(f.x) {
			return fmt.Errorf("%w: block %x rank %d exceeds column count %d",
				errDeserialize, id, meta.hRank, len(f.x))
		}
	}

	return nil
}

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
