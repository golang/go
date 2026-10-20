// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rt

import (
	"iter"
	"math"
	"sync"
)

// This file holds the runtime half of M5-S4 const maps (SPEC T-Sec9).
// The compiler lays out a const map as bucketed perfect-hash tables and
// emits a lookup function that computes a base hash with the functions
// below and calls ConstTable.Get. Compiler and runtime share these
// definitions: a change here changes the layout the compiler must emit.

// finalize is the splitmix64 finalizer: a full-diffusion avalanche.
func finalize(x uint64) uint64 {
	x ^= x >> 30
	x *= 0xbf58476d1ce4e5b9
	x ^= x >> 27
	x *= 0x94d049bb133111eb
	x ^= x >> 31
	return x
}

// HashString is the base hash of a string key: FNV-1a 64.
func HashString(s string) uint64 {
	h := uint64(14695981039346656037)
	for i := range len(s) {
		h ^= uint64(s[i])
		h *= 1099511628211
	}
	return h
}

// HashUint64 is the base hash of an integer or bool key: splitmix64 of the
// 64-bit value (signed values use their two's-complement bits).
func HashUint64(x uint64) uint64 {
	return finalize(x + 0x9e3779b97f4a7c15)
}

// HashFloat64 is the base hash of a float key. Negative zero hashes as
// positive zero, so both spell one key (Go == treats them as equal).
// Float32 keys widen to float64 first, which is exact.
func HashFloat64(f float64) uint64 {
	if f == 0 {
		f = 0
	}
	return HashUint64(math.Float64bits(f))
}

// HashBool is the base hash of a bool key: HashUint64 of 1 or 0.
func HashBool(b bool) uint64 {
	if b {
		return HashUint64(1)
	}
	return HashUint64(0)
}

// HashCombine folds one field or element hash into a composite key hash.
func HashCombine(h, x uint64) uint64 {
	return finalize(h ^ (x + 0x9e3779b97f4a7c15 + h<<6 + h>>2))
}

// ConstMix is the bucket and slot mix: a full-diffusion avalanche of the
// base hash h under seed. Buckets use the outer salt as seed; slots use
// the bucket seed.
func ConstMix(seed, h uint64) uint64 {
	return finalize(h ^ (seed * 0x9e3779b97f4a7c15))
}

// ConstBucket describes one bucket of a const map table. Off and Size name
// the bucket's slot region. Seed places the bucket's keys inside it. An
// empty bucket points at the table's shared empty slot with Size 1.
type ConstBucket struct {
	Off  uint32
	Size uint32
	Seed uint64
}

// ConstSlot is one key/value slot. Occ is false for unowned slots, which a
// lookup treats as absent before it compares keys.
type ConstSlot[K comparable, V any] struct {
	Key K
	Val V
	Occ bool
}

// ConstTable is one const map's lookup table: Buckets selected by the outer
// salt, then one slot in the bucket's region.
type ConstTable[K comparable, V any] struct {
	Salt    uint64
	Buckets []ConstBucket
	Slots   []ConstSlot[K, V]
}

// Get performs the one-probe lookup for key k with base hash h: one bucket
// read, one slot probe, then full-key validation. An unoccupied slot answers
// absent before any key comparison.
func (t *ConstTable[K, V]) Get(h uint64, k K) (V, bool) {
	b := &t.Buckets[ConstMix(t.Salt, h)%uint64(len(t.Buckets))]
	s := &t.Slots[b.Off+uint32(ConstMix(b.Seed, h)%uint64(b.Size))]
	if s.Occ && s.Key == k {
		return s.Val, true
	}
	var zero V
	return zero, false
}

// ConstEntry is one entry of a const map in sorted key order, the order
// range visits.
type ConstEntry[K comparable, V any] struct {
	Key K
	Val V
}

// ConstAll returns the range iterator of a const map over its sorted
// entries.
func ConstAll[K comparable, V any](sorted []ConstEntry[K, V]) iter.Seq2[K, V] {
	return func(yield func(K, V) bool) {
		for _, e := range sorted {
			if !yield(e.Key, e.Val) {
				return
			}
		}
	}
}

// Opt converts a lookup result to the option form of a single-value read
// (SPEC T-202): present is some, absent is none.
func Opt[V any](v V, ok bool) Option[V] {
	return Option[V]{Value: v, Present: ok}
}

// Bare converts a lookup result to the bare option form, for payloads
// that keep nil as none (SPEC T-202): absent reads as the zero value.
func Bare[V any](v V, ok bool) V {
	if !ok {
		var zero V
		return zero
	}
	return v
}

// Materialize returns the function that builds a const map's regular map
// once, on first use, and returns that same map afterwards (SPEC T-Sec9).
// Concurrent first uses share one build.
func Materialize[M any](f func() M) func() M {
	return sync.OnceValue(f)
}
