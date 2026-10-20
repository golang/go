// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

// This file computes the M5-S4 layout of a const map with more than 8 keys
// (SPEC T-Sec9): B = ceil(N/8) buckets chosen by the outer salt, each
// non-empty bucket owning a power-of-two slot region with its own seed,
// and one shared empty slot for empty buckets. The lookup it feeds is
// rt.ConstTable.Get, and rt.ConstMix and the hash functions in
// logos/rt/constmap.go must match constMix and the hashes here bit for bit.

import (
	"cmp"
	"fmt"
	"math"
	"slices"
)

// constMapKey is one key of a const map as the layout sees it.
type constMapKey struct {
	hash  uint64 // base hash (rt.HashString, rt.HashUint64, rt.HashFloat64, rt.HashCombine)
	canon string // canonical key bytes: the tie-break among equal base hashes
}

// constMapBudget bounds the seed search. MaxSeeds trials in total are split
// evenly over Rounds outer salts; a round that runs out of trials discards
// its tentative layout and the next salt is tried.
type constMapBudget struct {
	MaxSeeds int
	Rounds   int
}

// defaultConstMapBudget is the SPEC T-Sec9 budget: 1,000,000 trials in 8
// rounds of 125,000 (1000000//8 per round).
var defaultConstMapBudget = constMapBudget{MaxSeeds: 1000000, Rounds: 8}

// constMapBucket is one placed bucket. An empty bucket has keys == nil and
// points at the shared empty slot, so off is the sentinel index and size is 1.
type constMapBucket struct {
	off  int
	size int
	seed uint64
	keys []int // key indices of this bucket, in placement order
}

// constMapLayout is one placed const map table.
type constMapLayout struct {
	salt     uint64
	buckets  []constMapBucket
	slots    []int // key index for each owned slot; -1 for unowned slots
	sentinel int   // index of the shared empty slot, the last slot
	spent    int   // seed trials the winning round used
}

// constFinalize is the splitmix64 finalizer; it mirrors rt's finalize.
func constFinalize(x uint64) uint64 {
	x ^= x >> 30
	x *= 0xbf58476d1ce4e5b9
	x ^= x >> 27
	x *= 0x94d049bb133111eb
	x ^= x >> 31
	return x
}

// constMix is the bucket and slot mix; it mirrors rt.ConstMix.
func constMix(seed, h uint64) uint64 {
	return constFinalize(h ^ (seed * 0x9e3779b97f4a7c15))
}

// constHashUint64 mirrors rt.HashUint64.
func constHashUint64(x uint64) uint64 {
	return constFinalize(x + 0x9e3779b97f4a7c15)
}

// constHashBool mirrors rt.HashBool.
func constHashBool(b bool) uint64 {
	if b {
		return constHashUint64(1)
	}
	return constHashUint64(0)
}

// constHashFloat64 mirrors rt.HashFloat64: negative zero hashes as positive zero.
func constHashFloat64(f float64) uint64 {
	if f == 0 {
		f = 0
	}
	return constHashUint64(math.Float64bits(f))
}

// constHashCombine mirrors rt.HashCombine.
func constHashCombine(h, x uint64) uint64 {
	return constFinalize(h ^ (x + 0x9e3779b97f4a7c15 + h<<6 + h>>2))
}

// constMapBucketSize returns M_b for a bucket of n keys: the smallest power
// of two at least ceil(1.25n), and at least 2.
func constMapBucketSize(n int) int {
	need := (5*n + 3) / 4
	size := 2
	for size < need {
		size *= 2
	}
	return size
}

// buildConstMap places keys into a const map table. It returns an error
// naming comptime::constmap_seed_not_found when every round exhausts its
// budget. keys must be non-empty and free of duplicates (the evaluator
// rejects duplicate keys before layout).
func buildConstMap(keys []constMapKey, budget constMapBudget) (*constMapLayout, error) {
	if len(keys) == 0 {
		return nil, fmt.Errorf("const map layout needs at least one key")
	}
	rounds := max(budget.Rounds, 1)
	perRound := budget.MaxSeeds / rounds
	nb := (len(keys) + 7) / 8
	for r := range rounds {
		if l, ok := placeConstMapRound(keys, nb, uint64(r), perRound); ok {
			return l, nil
		}
	}
	return nil, fmt.Errorf("[comptime::constmap_seed_not_found] const map of %d keys: no bucket layout within %d seed trials over %d salts; simplify the key set or split the map", len(keys), budget.MaxSeeds, rounds)
}

// placeConstMapRound tries one outer salt with at most trials seed trials.
// Buckets go in (size desc, index asc) order; each bucket's keys go in
// (base hash asc, canonical bytes asc) order; bucket seeds 0, 1, ... are
// tried ascending and the first fit wins. Every (bucket, seed) attempt
// spends one trial.
func placeConstMapRound(keys []constMapKey, nb int, salt uint64, trials int) (*constMapLayout, bool) {
	start := trials
	members := make([][]int, nb)
	for i, k := range keys {
		b := int(constMix(salt, k.hash) % uint64(nb))
		members[b] = append(members[b], i)
	}
	for _, ks := range members {
		slices.SortFunc(ks, func(a, b int) int {
			if c := cmp.Compare(keys[a].hash, keys[b].hash); c != 0 {
				return c
			}
			return cmp.Compare(keys[a].canon, keys[b].canon)
		})
	}
	order := make([]int, nb)
	for i := range order {
		order[i] = i
	}
	slices.SortStableFunc(order, func(a, b int) int {
		if c := len(members[b]) - len(members[a]); c != 0 {
			return c
		}
		return a - b
	})

	seeds := make([]uint64, nb)
	rel := make([][]int, nb) // slot offset within the bucket region, per member
	for _, b := range order {
		ks := members[b]
		if len(ks) == 0 {
			continue
		}
		size := constMapBucketSize(len(ks))
		var seed uint64
		for {
			if trials == 0 {
				return nil, false
			}
			trials--
			if slots, ok := constMapFit(keys, ks, seed, size); ok {
				seeds[b] = seed
				rel[b] = slots
				break
			}
			seed++
		}
	}

	l := &constMapLayout{salt: salt, buckets: make([]constMapBucket, nb), spent: start - trials}
	total := 0
	for b := range nb {
		if len(members[b]) == 0 {
			continue
		}
		size := constMapBucketSize(len(members[b]))
		l.buckets[b] = constMapBucket{off: total, size: size, seed: seeds[b], keys: members[b]}
		total += size
	}
	l.sentinel = total
	total++
	l.slots = make([]int, total)
	for i := range l.slots {
		l.slots[i] = -1
	}
	for b := range nb {
		bk := &l.buckets[b]
		if len(bk.keys) == 0 {
			l.buckets[b] = constMapBucket{off: l.sentinel, size: 1}
			continue
		}
		for j, ki := range bk.keys {
			l.slots[bk.off+rel[b][j]] = ki
		}
	}
	return l, true
}

// constMapFit reports whether the keys ks of one bucket occupy distinct
// slots of a region of the given size under seed, and returns each key's
// slot offset within the region.
func constMapFit(keys []constMapKey, ks []int, seed uint64, size int) ([]int, bool) {
	slots := make([]int, len(ks))
	for j, ki := range ks {
		slots[j] = int(constMix(seed, keys[ki].hash) % uint64(size))
		for p := range j {
			if slots[p] == slots[j] {
				return nil, false
			}
		}
	}
	return slots, true
}

// probe returns the slot a lookup of base hash h reads: one bucket
// selection, then one slot in the bucket's region. It mirrors
// rt.ConstTable.Get.
func (l *constMapLayout) probe(h uint64) int {
	b := &l.buckets[constMix(l.salt, h)%uint64(len(l.buckets))]
	return b.off + int(constMix(b.seed, h)%uint64(b.size))
}

// constHashString is the base hash of a string key; it mirrors
// rt.HashString (FNV-1a 64).
func constHashString(s string) uint64 {
	h := uint64(14695981039346656037)
	for i := range len(s) {
		h ^= uint64(s[i])
		h *= 1099511628211
	}
	return h
}
