// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"fmt"
	"math"
	"strings"
	"testing"

	rt "logos/rt"
)

func constmapKeys(n int) []constMapKey {
	keys := make([]constMapKey, n)
	for i := range keys {
		s := fmt.Sprintf("k%05d", i)
		keys[i] = constMapKey{hash: constHashString(s), canon: s}
	}
	return keys
}

func TestConstmapSeedExhausted(t *testing.T) {
	keys := constmapKeys(9)

	_, err := buildConstMap(keys, constMapBudget{MaxSeeds: 4, Rounds: 2})
	if err == nil {
		t.Fatal("buildConstMap with 4 trials succeeded; want comptime::constmap_seed_not_found")
	}
	if got := err.Error(); !strings.Contains(got, "[comptime::constmap_seed_not_found]") {
		t.Fatalf("error = %q; want [comptime::constmap_seed_not_found]", err)
	}

	if _, err := buildConstMap(keys, defaultConstMapBudget); err != nil {
		t.Fatalf("production budget failed on the same keys: %v", err)
	}
}

func TestConstmapProbeFindsEveryKey(t *testing.T) {
	for _, n := range []int{9, 1000, 10000} {
		keys := constmapKeys(n)
		l, err := buildConstMap(keys, defaultConstMapBudget)
		if err != nil {
			t.Fatalf("%d keys: %v", n, err)
		}
		for i, k := range keys {
			if got := l.slots[l.probe(k.hash)]; got != i {
				t.Fatalf("%d keys: probe of key %d read slot holding %d", n, i, got)
			}
		}
	}
}

func TestConstmapBudgetIsOneMillion(t *testing.T) {
	if defaultConstMapBudget.MaxSeeds != 1000000 {
		t.Fatalf("defaultConstMapBudget.MaxSeeds = %d, want 1000000", defaultConstMapBudget.MaxSeeds)
	}
	if defaultConstMapBudget.Rounds != 8 {
		t.Fatalf("defaultConstMapBudget.Rounds = %d, want 8", defaultConstMapBudget.Rounds)
	}
	if got := defaultConstMapBudget.MaxSeeds / defaultConstMapBudget.Rounds; got != 125000 {
		t.Fatalf("per-round trials = %d, want 125000 (1000000//8)", got)
	}
}

func TestConstmapSwitchMax(t *testing.T) {
	if constMapSwitchMax != 8 {
		t.Fatalf("constMapSwitchMax = %d, want 8", constMapSwitchMax)
	}
}

func TestConstmapProbeIsSingleShot(t *testing.T) {
	for _, n := range []int{9, 64, 4096} {
		keys := constmapKeys(n)
		l, err := buildConstMap(keys, defaultConstMapBudget)
		if err != nil {
			t.Fatalf("%d keys: %v", n, err)
		}
		oneShot := func(h uint64) int {
			b := &l.buckets[constMix(l.salt, h)%uint64(len(l.buckets))]
			return b.off + int(constMix(b.seed, h)%uint64(b.size))
		}
		for i, k := range keys {
			if got := l.probe(k.hash); got != oneShot(k.hash) {
				t.Fatalf("%d keys: probe of key %d = %d, want the one-shot slot %d", n, i, got, oneShot(k.hash))
			}
			if l.slots[l.probe(k.hash)] != i {
				t.Fatalf("%d keys: probe of key %d read slot holding %d", n, i, l.slots[l.probe(k.hash)])
			}
		}
		for i := range 20000 {
			h := constHashString(fmt.Sprintf("__probe%d__", i))
			if got := l.probe(h); got != oneShot(h) {
				t.Fatalf("%d keys: probe of hash %#x = %d, want the one-shot slot %d; a search fallback must not satisfy the contract", n, h, got, oneShot(h))
			}
		}
	}
	l := &constMapLayout{
		salt: 7,
		buckets: []constMapBucket{
			{off: 0, size: 2, seed: 11},
			{off: 4, size: 1},
			{off: 2, size: 2, seed: 13},
			{off: 4, size: 1},
		},
		slots:    []int{-1, -1, -1, -1, -1},
		sentinel: 4,
	}
	oneShotSyn := func(h uint64) int {
		b := &l.buckets[constMix(l.salt, h)%uint64(len(l.buckets))]
		return b.off + int(constMix(b.seed, h)%uint64(b.size))
	}
	for i := range 64 {
		h := uint64(i) * 0x9e3779b97f4a7c15
		if got := l.probe(h); got != oneShotSyn(h) {
			t.Fatalf("synthetic layout: probe of hash %#x = %d, want the one-shot slot %d; a search fallback must not satisfy the contract", h, got, oneShotSyn(h))
		}
	}
}

func TestConstmapBudgetBoundary(t *testing.T) {
	keys := constmapKeys(9)
	l, err := buildConstMap(keys, defaultConstMapBudget)
	if err != nil {
		t.Fatalf("production budget: %v", err)
	}
	if _, err := buildConstMap(keys, constMapBudget{MaxSeeds: l.spent, Rounds: 1}); err != nil {
		t.Fatalf("budget of the %d trials spent failed: %v", l.spent, err)
	}
	if _, err := buildConstMap(keys, constMapBudget{MaxSeeds: l.spent - 1, Rounds: 1}); err == nil {
		t.Fatalf("budget of %d trials (one fewer than spent) succeeded", l.spent-1)
	}
}

func TestConstmapHashMatchesRuntime(t *testing.T) {
	strs := []string{"", "a", "k00000", "hello, world", "const map key with spaces and unicode \u00e9"}
	for _, s := range strs {
		if got, want := constHashString(s), rt.HashString(s); got != want {
			t.Fatalf("constHashString(%q) = %#x, want rt.HashString %#x", s, got, want)
		}
	}
	uints := []uint64{0, 1, 2, 0x9e3779b97f4a7c15, 0xffffffffffffffff}
	for _, x := range uints {
		if got, want := constHashUint64(x), rt.HashUint64(x); got != want {
			t.Fatalf("constHashUint64(%#x) = %#x, want rt %#x", x, got, want)
		}
	}
	for _, b := range []bool{false, true} {
		if got, want := constHashBool(b), rt.HashBool(b); got != want {
			t.Fatalf("constHashBool(%v) = %#x, want rt %#x", b, got, want)
		}
	}
	floats := []float64{0, math.Copysign(0, -1), 1.5, -273.15, math.MaxFloat64, math.SmallestNonzeroFloat64}
	for _, f := range floats {
		if got, want := constHashFloat64(f), rt.HashFloat64(f); got != want {
			t.Fatalf("constHashFloat64(%v) = %#x, want rt %#x", f, got, want)
		}
	}
	for _, x := range uints {
		for _, y := range uints {
			if got, want := constHashCombine(x, y), rt.HashCombine(x, y); got != want {
				t.Fatalf("constHashCombine(%#x, %#x) = %#x, want rt %#x", x, y, got, want)
			}
			if got, want := constMix(x, y), rt.ConstMix(x, y); got != want {
				t.Fatalf("constMix(%#x, %#x) = %#x, want rt %#x", x, y, got, want)
			}
		}
	}
}

func constmapRuntimeTable[K comparable](t *testing.T, hashes []uint64, keys []K) *rt.ConstTable[K, int] {
	t.Helper()
	ck := make([]constMapKey, len(keys))
	for i := range keys {
		ck[i] = constMapKey{hash: hashes[i], canon: fmt.Sprintf("key-%d", i)}
	}
	l, err := buildConstMap(ck, defaultConstMapBudget)
	if err != nil {
		t.Fatalf("buildConstMap %d keys: %v", len(keys), err)
	}
	tab := &rt.ConstTable[K, int]{Salt: l.salt}
	tab.Buckets = make([]rt.ConstBucket, len(l.buckets))
	for i, b := range l.buckets {
		tab.Buckets[i] = rt.ConstBucket{Off: uint32(b.off), Size: uint32(b.size), Seed: b.seed}
	}
	tab.Slots = make([]rt.ConstSlot[K, int], len(l.slots))
	for s, ki := range l.slots {
		if ki < 0 {
			continue
		}
		tab.Slots[s] = rt.ConstSlot[K, int]{Key: keys[ki], Val: ki, Occ: true}
	}
	return tab
}

func checkConstmapRuntimeGetString(t *testing.T, n int) {
	t.Helper()
	keys := make([]string, n)
	hashes := make([]uint64, n)
	for i := range keys {
		keys[i] = fmt.Sprintf("k%05d", i)
		hashes[i] = constHashString(keys[i])
	}
	tab := constmapRuntimeTable(t, hashes, keys)
	for i, k := range keys {
		v, ok := tab.Get(rt.HashString(k), k)
		if !ok || v != i {
			t.Fatalf("size %d: Get(%q) = (%d, %v), want (%d, true)", n, k, v, ok, i)
		}
	}
	for _, k := range []string{"", "zz-absent", "k99999", keys[0] + "x"} {
		if v, ok := tab.Get(rt.HashString(k), k); ok {
			t.Fatalf("size %d: Get(%q) = (%d, true), want absent", n, k, v)
		}
	}
}

func checkConstmapRuntimeGetInt(t *testing.T, n int) {
	t.Helper()
	keys := make([]int, n)
	hashes := make([]uint64, n)
	for i := range keys {
		keys[i] = i
		hashes[i] = constHashUint64(uint64(i))
	}
	tab := constmapRuntimeTable(t, hashes, keys)
	for i, k := range keys {
		v, ok := tab.Get(rt.HashUint64(uint64(k)), k)
		if !ok || v != i {
			t.Fatalf("size %d: Get(%d) = (%d, %v), want (%d, true)", n, k, v, ok, i)
		}
	}
	for _, k := range []int{-1, n + 7} {
		if v, ok := tab.Get(rt.HashUint64(uint64(k)), k); ok {
			t.Fatalf("size %d: Get(%d) = (%d, true), want absent", n, k, v)
		}
	}
}

func TestConstmapRuntimeGetRoundTrip(t *testing.T) {
	for _, n := range []int{9, 64, 4096} {
		checkConstmapRuntimeGetString(t, n)
		checkConstmapRuntimeGetInt(t, n)
	}
}
