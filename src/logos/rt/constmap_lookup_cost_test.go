// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rt

import (
	"fmt"
	"testing"
)

func lookupCostTable(keys []string) *ConstTable[string, int] {
	n := 1
	for n < len(keys) {
		n *= 2
	}
	groups := make([][]int, n)
	for i, k := range keys {
		j := ConstMix(0, HashString(k)) & uint64(n-1)
		groups[j] = append(groups[j], i)
	}
	tab := &ConstTable[string, int]{Buckets: make([]ConstBucket, n)}
	for j, group := range groups {
		size := 1
		for size < len(group)*len(group)*2 {
			size *= 2
		}
		slots := make([]ConstSlot[string, int], size)
		seed := uint64(0)
		for ; ; seed++ {
			clear(slots)
			collision := false
			for _, i := range group {
				p := ConstMix(seed, HashString(keys[i])) & uint64(size-1)
				if slots[p].Occ {
					collision = true
					break
				}
				slots[p] = ConstSlot[string, int]{Key: keys[i], Val: i + 1, Occ: true}
			}
			if !collision {
				break
			}
		}
		tab.Buckets[j] = ConstBucket{Off: uint32(len(tab.Slots)), Size: uint32(size), Seed: seed}
		tab.Slots = append(tab.Slots, slots...)
	}
	return tab
}

func BenchmarkConstMapLookupCost(b *testing.B) {
	for _, n := range []int{9, 64, 4096} {
		keys := make([]string, n)
		misses := make([]string, n)
		ordinary := make(map[string]int, n)
		for i := range keys {
			keys[i] = fmt.Sprintf("k%05d", i)
			misses[i] = fmt.Sprintf("m%05d", i)
			ordinary[keys[i]] = i + 1
		}
		tab := lookupCostTable(keys)
		for _, hit := range []bool{true, false} {
			probes := misses
			if hit {
				probes = keys
			}
			for _, kind := range []string{"const", "go"} {
				b.Run(fmt.Sprintf("%d/hit=%t/%s", n, hit, kind), func(b *testing.B) {
					lookup := func(k string) (int, bool) { return tab.Get(HashString(k), k) }
					if kind == "go" {
						lookup = func(k string) (int, bool) { v, ok := ordinary[k]; return v, ok }
					}
					for i, k := range probes {
						want := 0
						if hit {
							want = i + 1
						}
						if v, ok := lookup(k); v != want || ok != hit {
							b.Fatalf("T-Sec7: %d %t != %d %t", v, ok, want, hit)
						}
					}
					if a := testing.AllocsPerRun(100, func() {
						for _, k := range probes {
							v, _ := lookup(k)
							sinkConstMap = v
						}
					}); a != 0 {
						b.Fatalf("T-Sec7: allocs=%v", a)
					}
					b.ReportAllocs()
					b.ResetTimer()
					for i := range b.N {
						v, _ := lookup(probes[i%len(probes)])
						sinkConstMap = v
					}
				})
			}
		}
	}
}
