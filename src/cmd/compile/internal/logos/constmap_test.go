// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

// Attack tests for M5-S4 const maps (SPEC T-Sec9): the seed search has a
// fixed budget, and exhausting it reports comptime::constmap_seed_not_found
// instead of searching on. The 10,000-key acceptance and the run-time
// lookups are owned by test/logos/run/const_map.lgo.

import (
	"fmt"
	"strings"
	"testing"
)

// constmapKeys builds n string keys with their base hashes and canonical
// bytes, the inputs buildConstMap takes.
func constmapKeys(n int) []constMapKey {
	keys := make([]constMapKey, n)
	for i := range keys {
		s := fmt.Sprintf("k%05d", i)
		keys[i] = constMapKey{hash: constHashString(s), canon: s}
	}
	return keys
}

// TestConstmapSeedExhausted forces comptime::constmap_seed_not_found with
// an injected tiny budget: 4 trials split over 2 outer rounds, so each
// round gets 2 trials and the 9-key set cannot place every bucket. The
// production budget places the same keys, so the failure is the budget's
// doing, not the keys'.
func TestConstmapSeedExhausted(t *testing.T) {
	keys := constmapKeys(9)

	_, err := buildConstMap(keys, constMapBudget{MaxSeeds: 4, Rounds: 2})
	if err == nil {
		t.Fatal("buildConstMap with 4 trials succeeded; want comptime::constmap_seed_not_found")
	}
	if !strings.Contains(err.Error(), "[comptime::constmap_seed_not_found]") {
		t.Fatalf("error = %q; want [comptime::constmap_seed_not_found]", err)
	}

	if _, err := buildConstMap(keys, defaultConstMapBudget); err != nil {
		t.Fatalf("production budget failed on the same keys: %v", err)
	}
}

// TestConstmapProbeFindsEveryKey places 9, 1000, and 10,000 keys with the
// production budget and checks that each key's single probe reads its own
// slot, and that a key outside the set reads a slot holding no equal key.
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
		absent := constHashString("absent-key")
		if s := l.slots[l.probe(absent)]; s >= 0 && keys[s].canon == "absent-key" {
			t.Fatalf("%d keys: absent key reported present", n)
		}
	}
}

// TestConstmapBudgetBoundary pins the seed budget at its exact edge: the
// trials a layout spends are enough on their own, and one fewer is not. A
// bound off by one in either direction changes one of the two outcomes.
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
