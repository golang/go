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
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"runtime"
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

// TestConstmapBudgetIsOneMillion pins the SPEC T-Sec9 construction ceiling
// (#140): 1,000,000 seed-candidate trials in total per const map, split
// into 8 deterministic outer rounds of 125,000 trials each. A ceiling
// off by one in either direction changes one of the two outcomes below.
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

// TestConstmapSwitchMax pins the SPEC T-Sec9 lowering boundary (#3): a
// const map with up to 8 keys lowers to a switch lookup, and a larger
// map lowers to the static perfect-hash table. The run fixture
// test/logos/run/constmap_switch.lgo proves the emitted form end to
// end (no _tab symbol at 8 keys, a read-only _tab symbol above that);
// this pin keeps the boundary itself from drifting.
func TestConstmapSwitchMax(t *testing.T) {
	if constMapSwitchMax != 8 {
		t.Fatalf("constMapSwitchMax = %d, want 8", constMapSwitchMax)
	}
}

// TestConstmapProbeIsSingleProbe pins the SPEC T-Sec9 lookup shape
// (#140): the compiler-side probe performs one bucket selection plus
// exactly one slot read, with no loop and no logarithmic fallback. A
// loop or a binary search in this path voids the constant-time proof,
// so the test parses its own source and rejects both.
func TestConstmapProbeIsSingleProbe(t *testing.T) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	src, err := os.ReadFile(filepath.Join(filepath.Dir(file), "constmap.go"))
	if err != nil {
		t.Fatal(err)
	}
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "constmap.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}
	var probe *ast.FuncDecl
	for _, d := range f.Decls {
		if fd, ok := d.(*ast.FuncDecl); ok && fd.Name.Name == "probe" {
			probe = fd
		}
	}
	if probe == nil {
		t.Fatal("func probe not found in constmap.go")
	}
	// probe computes the single slot index the run-time Get reads: one
	// bucket selection plus exactly one slot offset, with no loop. The
	// slot read itself lives in rt.ConstTable.Get (guarded by
	// TestConstTableGetSingleProbe in src/logos/rt), and the behavioral
	// proof that the computed index reads the key's own slot is
	// TestConstmapProbeFindsEveryKey above.
	loops, buckets := 0, 0
	ast.Inspect(probe.Body, func(n ast.Node) bool {
		switch n := n.(type) {
		case *ast.ForStmt, *ast.RangeStmt:
			loops++
		case *ast.IndexExpr:
			if sel, ok := n.X.(*ast.SelectorExpr); ok && sel.Sel.Name == "buckets" {
				buckets++
			}
		}
		return true
	})
	if loops != 0 {
		t.Fatalf("probe holds %d loops, want none (one-probe lookup)", loops)
	}
	if buckets != 1 {
		t.Fatalf("probe reads %d bucket expressions, want exactly 1", buckets)
	}
	for _, name := range []string{"constmap.go", "constmap_lower.go"} {
		body, err := os.ReadFile(filepath.Join(filepath.Dir(file), name))
		if err != nil {
			t.Fatal(err)
		}
		for _, banned := range []string{"sort.Search", "sort.Find", "BinarySearch", "binarySearch"} {
			if strings.Contains(string(body), banned) {
				t.Fatalf("%s contains %q: no logarithmic fallback may satisfy the contract", name, banned)
			}
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
