// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"cmp"
	"fmt"
	"math"
	"slices"
)

type constMapKey struct {
	hash  uint64
	canon string
}

type constMapBudget struct {
	MaxSeeds int
	Rounds   int
}

var defaultConstMapBudget = constMapBudget{MaxSeeds: 1000000, Rounds: 8}

type constMapBucket struct {
	off  int
	size int
	seed uint64
	keys []int
}

type constMapLayout struct {
	salt     uint64
	buckets  []constMapBucket
	slots    []int
	sentinel int
	spent    int
}

func constFinalize(x uint64) uint64 {
	x ^= x >> 30
	x *= 0xbf58476d1ce4e5b9
	x ^= x >> 27
	x *= 0x94d049bb133111eb
	x ^= x >> 31
	return x
}

func constMix(seed, h uint64) uint64 {
	return constFinalize(h ^ (seed * 0x9e3779b97f4a7c15))
}

func constHashUint64(x uint64) uint64 {
	return constFinalize(x + 0x9e3779b97f4a7c15)
}

func constHashBool(b bool) uint64 {
	if b {
		return constHashUint64(1)
	}
	return constHashUint64(0)
}

func constHashFloat64(f float64) uint64 {
	if f == 0 {
		f = 0
	}
	return constHashUint64(math.Float64bits(f))
}

func constHashCombine(h, x uint64) uint64 {
	return constFinalize(h ^ (x + 0x9e3779b97f4a7c15 + h<<6 + h>>2))
}

func constMapBucketSize(n int) int {
	need := (5*n + 3) / 4
	size := 2
	for size < need {
		size *= 2
	}
	return size
}

func buildConstMap(keys []constMapKey, budget constMapBudget) (*constMapLayout, error) {
	if len(keys) == 0 {
		return nil, fmt.Errorf("const map layout needs at least one key")
	}
	rounds := max(budget.Rounds, 1)
	perRound := budget.MaxSeeds / rounds
	nb := 1
	for nb < (len(keys)+7)/8 {
		nb *= 2
	}
	for r := range rounds {
		if l, ok := placeConstMapRound(keys, nb, uint64(r), perRound); ok {
			return l, nil
		}
	}
	return nil, fmt.Errorf("[comptime::constmap_seed_not_found] const map of %d keys: no bucket layout within %d seed trials over %d salts; simplify the key set or split the map", len(keys), budget.MaxSeeds, rounds)
}

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
	rel := make([][]int, nb)
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

func (l *constMapLayout) probe(h uint64) int {
	b := &l.buckets[constMix(l.salt, h)&uint64(len(l.buckets)-1)]
	return b.off + int(constMix(b.seed, h)&uint64(b.size-1))
}

func constHashString(s string) uint64 {
	h := uint64(14695981039346656037)
	for i := range len(s) {
		h ^= uint64(s[i])
		h *= 1099511628211
	}
	return h
}
