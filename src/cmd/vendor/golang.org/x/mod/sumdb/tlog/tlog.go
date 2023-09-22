// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package tlog implements a tamper-evident log
// used in the Go module go.sum database server.
//
// This package follows the design of Certificate Transparency (RFC 6962)
// and its proofs are compatible with that system.
// See TestCertificateTransparency.
package tlog

import (
	"crypto/sha256"
	"encoding/base64"
	"errors"
	"fmt"
	"math/bits"
)

// A Hash is a hash identifying a log record or tree root.
type Hash [HashSize]byte

// HashSize is the size of a Hash in bytes.
const HashSize = 32

// String returns a base64 representation of the hash for printing.
func (h Hash) String() string {
	return base64.StdEncoding.EncodeToString(h[:])
}

// MarshalJSON marshals the hash as a JSON string containing the base64-encoded hash.
func (h Hash) MarshalJSON() ([]byte, error) {
	return []byte(`"` + h.String() + `"`), nil
}

// UnmarshalJSON unmarshals a hash from JSON string containing the a base64-encoded hash.
func (h *Hash) UnmarshalJSON(data []byte) error {
	if len(data) != 1+44+1 || data[0] != '"' || data[len(data)-2] != '=' || data[len(data)-1] != '"' {
		return errors.New("cannot decode hash")
	}

	// As of Go 1.12, base64.StdEncoding.Decode insists on
	// slicing into target[33:] even when it only writes 32 bytes.
	// Since we already checked that the hash ends in = above,
	// we can use base64.RawStdEncoding with the = removed;
	// RawStdEncoding does not exhibit the same bug.
	// We decode into a temporary to avoid writing anything to *h
	// unless the entire input is well-formed.
	var tmp Hash
	n, err := base64.RawStdEncoding.Decode(tmp[:], data[1:len(data)-2])
	if err != nil || n != HashSize {
		return errors.New("cannot decode hash")
	}
	*h = tmp
	return nil
}

// ParseHash parses the base64-encoded string form of a hash.
func ParseHash(s string) (Hash, error) {
	data, err := base64.StdEncoding.DecodeString(s)
	if err != nil || len(data) != HashSize {
		return Hash{}, fmt.Errorf("malformed hash")
	}
	var h Hash
	copy(h[:], data)
	return h, nil
}

// maxpow2 returns k, the maximum power of 2 smaller than n,
// as well as l = log₂ k (so k = 1<<l).
func maxpow2(n int64) (k int64, l int) {
	l = 0
	for 1<<uint(l+1) < n {
		l++
	}
	return 1 << uint(l), l
}

var zeroPrefix = []byte{0x00}

// RecordHash returns the content hash for the given record data.
func RecordHash(data []byte) Hash {
	// SHA256(0x00 || data)
	// https://tools.ietf.org/html/rfc6962#section-2.1
	h := sha256.New()
	h.Write(zeroPrefix)
	h.Write(data)
	var h1 Hash
	h.Sum(h1[:0])
	return h1
}

// NodeHash returns the hash for an interior tree node with the given left and right hashes.
func NodeHash(left, right Hash) Hash {
	// SHA256(0x01 || left || right)
	// https://tools.ietf.org/html/rfc6962#section-2.1
	// We use a stack buffer to assemble the hash input
	// to avoid allocating a hash struct with sha256.New.
	var buf [1 + HashSize + HashSize]byte
	buf[0] = 0x01
	copy(buf[1:], left[:])
	copy(buf[1+HashSize:], right[:])
	return sha256.Sum256(buf[:])
}

// For information about the stored hash index ordering,
// see section 3.3 of Crosby and Wallach's paper
// "Efficient Data Structures for Tamper-Evident Logging".
// https://www.usenix.org/legacy/event/sec09/tech/full_papers/crosby.pdf

// StoredHashIndex maps the tree coordinates (level, n)
// to a dense linear ordering that can be used for hash storage.
// Hash storage implementations that store hashes in sequential
// storage can use this function to compute where to read or write
// a given hash.
func StoredHashIndex(level int, n int64) int64 {
	// Level L's n'th hash is written right after level L+1's 2n+1'th hash.
	// Work our way down to the level 0 ordering.
	// We'll add back the original level count at the end.
	for l := level; l > 0; l-- {
		n = 2*n + 1
	}

	// Level 0's n'th hash is written at n+n/2+n/4+... (eventually n/2ⁱ hits zero).
	i := int64(0)
	for ; n > 0; n >>= 1 {
		i += n
	}

	return i + int64(level)
}

// SplitStoredHashIndex is the inverse of [StoredHashIndex].
// That is, SplitStoredHashIndex(StoredHashIndex(level, n)) == level, n.
func SplitStoredHashIndex(index int64) (level int, n int64) {
	// Determine level 0 record before index.
	// StoredHashIndex(0, n) < 2*n,
	// so the n we want is in [index/2, index/2+log₂(index)].
	n = index / 2
	indexN := StoredHashIndex(0, n)
	if indexN > index {
		panic("bad math")
	}
	for {
		// Each new record n adds 1 + trailingZeros(n) hashes.
		x := indexN + 1 + int64(bits.TrailingZeros64(uint64(n+1)))
		if x > index {
			break
		}
		n++
		indexN = x
	}
	// The hash we want was committed with record n,
	// meaning it is one of (0, n), (1, n/2), (2, n/4), ...
	level = int(index - indexN)
	return level, n >> uint(level)
}

// StoredHashCount returns the number of stored hashes
// that are expected for a tree with n records.
func StoredHashCount(n int64) int64 {
	if n == 0 {
		return 0
	}
	// The tree will have the hashes up to the last leaf hash.
	numHash := StoredHashIndex(0, n-1) + 1
	// And it will have any hashes for subtrees completed by that leaf.
	for i := uint64(n - 1); i&1 != 0; i >>= 1 {
		numHash++
	}
	return numHash
}

// StoredHashes returns the hashes that must be stored when writing
// record n with the given data. The hashes should be stored starting
// at StoredHashIndex(0, n). The result will have at most 1 + log₂ n hashes,
// but it will average just under two per call for a sequence of calls for n=1..k.
//
// StoredHashes may read up to log n earlier hashes from r
// in order to compute hashes for completed subtrees.
func StoredHashes(n int64, data []byte, r HashReader) ([]Hash, error) {
	return StoredHashesForRecordHash(n, RecordHash(data), r)
}

// StoredHashesForRecordHash is like [StoredHashes] but takes
// as its second argument RecordHash(data) instead of data itself.
func StoredHashesForRecordHash(n int64, h Hash, r HashReader) ([]Hash, error) {
	// Start with the record hash.
	hashes := []Hash{h}

	// Build list of indexes needed for hashes for completed subtrees.
	// Each trailing 1 bit in the binary representation of n completes a subtree
	// and consumes a hash from an adjacent subtree.
	m := int(bits.TrailingZeros64(uint64(n + 1)))
	indexes := make([]int64, m)
	for i := 0; i < m; i++ {
		// We arrange indexes in sorted order.
		// Note that n>>i is always odd.
		indexes[m-1-i] = StoredHashIndex(i, n>>uint(i)-1)
	}

	// Fetch hashes.
	old, err := r.ReadHashes(indexes)
	if err != nil {
		return nil, err
	}
	if len(old) != len(indexes) {
		return nil, fmt.Errorf("tlog: ReadHashes(%d indexes) = %d hashes", len(indexes), len(old))
	}

	// Build new hashes.
	for i := 0; i < m; i++ {
		h = NodeHash(old[m-1-i], h)
		hashes = append(hashes, h)
	}
	return hashes, nil
}

// A HashReader can read hashes for nodes in the log's tree structure.
type HashReader interface {
	// ReadHashes returns the hashes with the given stored hash indexes
	// (see StoredHashIndex and SplitStoredHashIndex).
	// ReadHashes must return a slice of hashes the same length as indexes,
	// or else it must return a non-nil error.
	// ReadHashes may run faster if indexes is sorted in increasing order.
	ReadHashes(indexes []int64) ([]Hash, error)
}

// A HashReaderFunc is a function implementing [HashReader].
type HashReaderFunc func([]int64) ([]Hash, error)

func (f HashReaderFunc) ReadHashes(indexes []int64) ([]Hash, error) {
	return f(indexes)
}

// TreeHash computes the hash for the root of the tree with n records,
// using the HashReader to obtain previously stored hashes
// (those returned by StoredHashes during the writes of those n records).
// TreeHash makes a single call to ReadHash requesting at most 1 + log₂ n hashes.
// The tree of size zero is defined to have an all-zero Hash.
func TreeHash(n int64, r HashReader) (Hash, error) {
	if n == 0 {
		return Hash{}, nil
	}
	indexes := subTreeIndex(0, n, nil)
	hashes, err := r.ReadHashes(indexes)
	if err != nil {
		return Hash{}, err
	}
	if len(hashes) != len(indexes) {
		return Hash{}, fmt.Errorf("tlog: ReadHashes(%d indexes) = %d hashes", len(indexes), len(hashes))
	}
	hash, hashes := subTreeHash(0, n, hashes)
	if len(hashes) != 0 {
		panic("tlog: bad index math in TreeHash")
	}
	return hash, nil
}

// subTreeIndex returns the storage indexes needed to compute
// the hash for the subtree containing records [lo, hi),
// appending them to need and returning the result.
// See https://tools.ietf.org/html/rfc6962#section-2.1
func subTreeIndex(lo, hi int64, need []int64) []int64 {
	// See subTreeHash below for commentary.
	for lo < hi {
		k, level := maxpow2(hi - lo + 1)
		if lo&(k-1) != 0 {
			panic("tlog: bad math in subTreeIndex")
		}
		need = append(need, StoredHashIndex(level, lo>>uint(level)))
		lo += k
	}
	return need
}

// subTreeHash computes the hash for the subtree containing records [lo, hi),
// assuming that hashes are the hashes corresponding to the indexes
// returned by subTreeIndex(lo, hi).
// It returns any leftover hashes.
func subTreeHash(lo, hi int64, hashes []Hash) (Hash, []Hash) {
	// Repeatedly partition the tree into a left side with 2^level nodes,
	// for as large a level as possible, and a right side with the fringe.
	// The left hash is stored directly and can be read from storage.
	// The right side needs further computation.
	numTree := 0
	for lo < hi {
		k, _ := maxpow2(hi - lo + 1)
		if lo&(k-1) != 0 || lo >= hi {
			panic("tlog: bad math in subTreeHash")
		}
		numTree++
		lo += k
	}

	if len(hashes) < numTree {
		panic("tlog: bad index math in subTreeHash")
	}

	// Reconstruct hash.
	h := hashes[numTree-1]
	for i := numTree - 2; i >= 0; i-- {
		h = NodeHash(hashes[i], h)
	}
	return h, hashes[numTree:]
}

// A RecordProof is a verifiable proof that a particular log root contains a particular record.
// RFC 6962 calls this a “Merkle audit path.”
type RecordProof []Hash

// ProveRecord returns the proof that the tree of size t contains the record with index n.
func ProveRecord(t, n int64, r HashReader) (RecordProof, error) {
	if t < 0 || n < 0 || n >= t {
		return nil, fmt.Errorf("tlog: invalid inputs in ProveRecord")
	}
	indexes := leafProofIndex(0, t, n, nil)
	if len(indexes) == 0 {
		return RecordProof{}, nil
	}
	hashes, err := r.ReadHashes(indexes)
	if err != nil {
		return nil, err
	}
	if len(hashes) != len(indexes) {
		return nil, fmt.Errorf("tlog: ReadHashes(%d indexes) = %d hashes", len(indexes), len(hashes))
	}

	p, hashes := leafProof(0, t, n, hashes)
	if len(hashes) != 0 {
		panic("tlog: bad index math in ProveRecord")
	}
	return p, nil
}

// leafProofIndex builds the list of indexes needed to construct the proof
// that leaf n is contained in the subtree with leaves [lo, hi).
// It appends those indexes to need and returns the result.
// See https://tools.ietf.org/html/rfc6962#section-2.1.1
func leafProofIndex(lo, hi, n int64, need []int64) []int64 {
	// See leafProof below for commentary.
	if !(lo <= n && n < hi) {
		panic("tlog: bad math in leafProofIndex")
	}
	if lo+1 == hi {
		return need
	}
	if k, _ := maxpow2(hi - lo); n < lo+k {
		need = leafProofIndex(lo, lo+k, n, need)
		need = subTreeIndex(lo+k, hi, need)
	} else {
		need = subTreeIndex(lo, lo+k, need)
		need = leafProofIndex(lo+k, hi, n, need)
	}
	return need
}

// leafProof constructs the proof that leaf n is contained in the subtree with leaves [lo, hi).
// It returns any leftover hashes as well.
// See https://tools.ietf.org/html/rfc6962#section-2.1.1
func leafProof(lo, hi, n int64, hashes []Hash) (RecordProof, []Hash) {
	// We must have lo <= n < hi or else the code here has a bug.
	if !(lo <= n && n < hi) {
		panic("tlog: bad math in leafProof")
	}

	if lo+1 == hi { // n == lo
		// Reached the leaf node.
		// The verifier knows what the leaf hash is, so we don't need to send it.
		return RecordProof{}, hashes
	}

	// Walk down the tree toward n.
	// Record the hash of the path not taken (needed for verifying the proof).
	var p RecordProof
	var th Hash
	if k, _ := maxpow2(hi - lo); n < lo+k {
		// n is on left side
		p, hashes = leafProof(lo, lo+k, n, hashes)
		th, hashes = subTreeHash(lo+k, hi, hashes)
	} else {
		// n is on right side
		th, hashes = subTreeHash(lo, lo+k, hashes)
		p, hashes = leafProof(lo+k, hi, n, hashes)
	}
	return append(p, th), hashes
}

var errProofFailed = errors.New("invalid transparency proof")

// CheckRecord verifies that p is a valid proof that the tree of size t
// with hash th has an n'th record with hash h.
func CheckRecord(p RecordProof, t int64, th Hash, n int64, h Hash) error {
	if t < 0 || n < 0 || n >= t {
		return fmt.Errorf("tlog: invalid inputs in CheckRecord")
	}
	th2, err := runRecordProof(p, 0, t, n, h)
	if err != nil {
		return err
	}
	if th2 == th {
		return nil
	}
	return errProofFailed
}

// runRecordProof runs the proof p that leaf n is contained in the subtree with leaves [lo, hi).
// Running the proof means constructing and returning the implied hash of that
// subtree.
func runRecordProof(p RecordProof, lo, hi, n int64, leafHash Hash) (Hash, error) {
	// We must have lo <= n < hi or else the code here has a bug.
	if !(lo <= n && n < hi) {
		panic("tlog: bad math in runRecordProof")
	}

	if lo+1 == hi { // m == lo
		// Reached the leaf node.
		// The proof must not have any unnecessary hashes.
		if len(p) != 0 {
			return Hash{}, errProofFailed
		}
		return leafHash, nil
	}

	if len(p) == 0 {
		return Hash{}, errProofFailed
	}

	k, _ := maxpow2(hi - lo)
	if n < lo+k {
		th, err := runRecordProof(p[:len(p)-1], lo, lo+k, n, leafHash)
		if err != nil {
			return Hash{}, err
		}
		return NodeHash(th, p[len(p)-1]), nil
	} else {
		th, err := runRecordProof(p[:len(p)-1], lo+k, hi, n, leafHash)
		if err != nil {
			return Hash{}, err
		}
		return NodeHash(p[len(p)-1], th), nil
	}
}

// A TreeProof is a verifiable proof that a particular log tree contains
// as a prefix all records present in an earlier tree.
// RFC 6962 calls this a “Merkle consistency proof.”
type TreeProof []Hash

// ProveTree returns the proof that the tree of size t contains
// as a prefix all the records from the tree of smaller size n.
func ProveTree(t, n int64, h HashReader) (TreeProof, error) {
	if t < 1 || n < 1 || n > t {
		return nil, fmt.Errorf("tlog: invalid inputs in ProveTree")
	}
	indexes := treeProofIndex(0, t, n, nil)
	if len(indexes) == 0 {
		return TreeProof{}, nil
	}
	hashes, err := h.ReadHashes(indexes)
	if err != nil {
		return nil, err
	}
	if len(hashes) != len(indexes) {
		return nil, fmt.Errorf("tlog: ReadHashes(%d indexes) = %d hashes", len(indexes), len(hashes))
	}

	p, hashes := treeProof(0, t, n, hashes)
	if len(hashes) != 0 {
		panic("tlog: bad index math in ProveTree")
	}
	return p, nil
}

// treeProofIndex builds the list of indexes needed to construct
// the sub-proof related to the subtree containing records [lo, hi).
// See https://tools.ietf.org/html/rfc6962#section-2.1.2.
func treeProofIndex(lo, hi, n int64, need []int64) []int64 {
	// See treeProof below for commentary.
	if !(lo < n && n <= hi) {
		panic("tlog: bad math in treeProofIndex")
	}

	if n == hi {
		if lo == 0 {
			return need
		}
		return subTreeIndex(lo, hi, need)
	}

	if k, _ := maxpow2(hi - lo); n <= lo+k {
		need = treeProofIndex(lo, lo+k, n, need)
		need = subTreeIndex(lo+k, hi, need)
	} else {
		need = subTreeIndex(lo, lo+k, need)
		need = treeProofIndex(lo+k, hi, n, need)
	}
	return need
}

// treeProof constructs the sub-proof related to the subtree containing records [lo, hi).
// It returns any leftover hashes as well.
// See https://tools.ietf.org/html/rfc6962#section-2.1.2.
func treeProof(lo, hi, n int64, hashes []Hash) (TreeProof, []Hash) {
	// We must have lo < n <= hi or else the code here has a bug.
	if !(lo < n && n <= hi) {
		panic("tlog: bad math in treeProof")
	}

	// Reached common ground.
	if n == hi {
		if lo == 0 {
			// This subtree corresponds exactly to the old tree.
			// The verifier knows that hash, so we don't need to send it.
			return TreeProof{}, hashes
		}
		th, hashes := subTreeHash(lo, hi, hashes)
		return TreeProof{th}, hashes
	}

	// Interior node for the proof.
	// Decide whether to walk down the left or right side.
	var p TreeProof
	var th Hash
	if k, _ := maxpow2(hi - lo); n <= lo+k {
		// m is on left side
		p, hashes = treeProof(lo, lo+k, n, hashes)
		th, hashes = subTreeHash(lo+k, hi, hashes)
	} else {
		// m is on right side
		th, hashes = subTreeHash(lo, lo+k, hashes)
		p, hashes = treeProof(lo+k, hi, n, hashes)
	}
	return append(p, th), hashes
}

// CheckTree verifies that p is a valid proof that the tree of size t with hash th
// contains as a prefix the tree of size n with hash h.
func CheckTree(p TreeProof, t int64, th Hash, n int64, h Hash) error {
	if t < 1 || n < 1 || n > t {
		return fmt.Errorf("tlog: invalid inputs in CheckTree")
	}
	h2, th2, err := runTreeProof(p, 0, t, n, h)
	if err != nil {
		return err
	}
	if th2 == th && h2 == h {
		return nil
	}
	return errProofFailed
}

// runTreeProof runs the sub-proof p related to the subtree containing records [lo, hi),
// where old is the hash of the old tree with n records.
// Running the proof means constructing and returning the implied hashes of that
// subtree in both the old and new tree.
func runTreeProof(p TreeProof, lo, hi, n int64, old Hash) (Hash, Hash, error) {
	// We must have lo < n <= hi or else the code here has a bug.
	if !(lo < n && n <= hi) {
		panic("tlog: bad math in runTreeProof")
	}

	// Reached common ground.
	if n == hi {
		if lo == 0 {
			if len(p) != 0 {
				return Hash{}, Hash{}, errProofFailed
			}
			return old, old, nil
		}
		if len(p) != 1 {
			return Hash{}, Hash{}, errProofFailed
		}
		return p[0], p[0], nil
	}

	if len(p) == 0 {
		return Hash{}, Hash{}, errProofFailed
	}

	// Interior node for the proof.
	k, _ := maxpow2(hi - lo)
	if n <= lo+k {
		oh, th, err := runTreeProof(p[:len(p)-1], lo, lo+k, n, old)
		if err != nil {
			return Hash{}, Hash{}, err
		}
		return oh, NodeHash(th, p[len(p)-1]), nil
	} else {
		oh, th, err := runTreeProof(p[:len(p)-1], lo+k, hi, n, old)
		if err != nil {
			return Hash{}, Hash{}, err
		}
		return NodeHash(p[len(p)-1], oh), NodeHash(p[len(p)-1], th), nil
	}
}
