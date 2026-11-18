// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package drbg

import (
	"crypto/internal/fips140"
	"sync"
	"testing"
)

// withEmptySlots enables FIPS 140-3 mode, empties the set of slots, and unseeds
// the root DRBG for the duration of the test.
func withEmptySlots(t *testing.T) {
	if err := fips140.Supported(); err != nil {
		t.Skipf("test requires FIPS 140-3 mode: %v", err)
	}
	oldEnabled, oldSlots, oldRoot := fips140.Enabled, slots.all.Load(), root.drbg
	t.Cleanup(func() {
		fips140.Enabled = oldEnabled
		slots.all.Store(oldSlots)
		root.drbg = oldRoot
	})
	fips140.Enabled = true
	slots.all.Store(nil)
	root.drbg = nil
}

// checkSlots returns the slots, after checking that they are all unlocked.
func checkSlots(t *testing.T) []*slot {
	t.Helper()
	var all []*slot
	if p := slots.all.Load(); p != nil {
		all = *p
	}
	for i, s := range all {
		if !s.TryLock() {
			t.Errorf("slot %d is locked", i)
			continue
		}
		s.Unlock()
	}
	return all
}

func TestRootSeededAtInit(t *testing.T) {
	if !fips140.Enabled {
		t.Skip("test requires FIPS 140-3 mode to be enabled at init time")
	}
	if root.drbg == nil {
		t.Error("root DRBG was not seeded at init time")
	}
}

func TestSlots(t *testing.T) {
	withEmptySlots(t)

	var seedings int
	testingOnlyGetEntropy = func() { seedings++ }
	t.Cleanup(func() { testingOnlyGetEntropy = nil })

	// Sequential calls share a slot.
	for range 10 {
		Read(make([]byte, 32))
	}
	if all := checkSlots(t); len(all) != 1 {
		t.Errorf("got %d slots after sequential calls, want 1", len(all))
	}

	// A slot is added only when all the existing ones are in use.
	s := acquireSlot()
	Read(make([]byte, 32))
	s.Unlock()
	for range 10 {
		Read(make([]byte, 32))
	}
	if all := checkSlots(t); len(all) != 2 {
		t.Errorf("got %d slots after a call with one slot in use, want 2", len(all))
	}

	// The entropy source is only used to seed the root.
	if seedings != 1 {
		t.Errorf("the entropy source was used %d times, want 1", seedings)
	}
}

func TestReadConcurrent(t *testing.T) {
	withEmptySlots(t)

	const goroutines = 32
	var wg sync.WaitGroup
	for range goroutines {
		wg.Go(func() {
			small := make([]byte, 32)
			large := make([]byte, 2*maxRequestSize+1)
			for i := range 50 {
				if i%10 == 0 {
					Read(large)
				} else {
					Read(small)
				}
			}
		})
	}
	wg.Wait()

	// There can't be more slots than concurrent calls.
	if all := checkSlots(t); len(all) == 0 || len(all) > goroutines {
		t.Errorf("got %d slots, want 1 to %d", len(all), goroutines)
	}
}

func TestReadSeedPanic(t *testing.T) {
	withEmptySlots(t)

	const msg = "injected getEntropy failure"
	testingOnlyGetEntropy = func() { panic(msg) }
	t.Cleanup(func() { testingOnlyGetEntropy = nil })

	func() {
		defer func() {
			if r := recover(); r != msg {
				t.Errorf("Read panicked with %v, want %q", r, msg)
			}
		}()
		Read(make([]byte, 32))
	}()

	// The slot is left unlocked and unseeded, so the next call retries.
	all := checkSlots(t)
	if len(all) != 1 || all[0].drbg != nil {
		t.Fatalf("got %d slots, want 1 unseeded slot", len(all))
	}

	testingOnlyGetEntropy = nil
	Read(make([]byte, 32))
	if all := checkSlots(t); len(all) != 1 || all[0].drbg == nil {
		t.Errorf("got %d slots, want 1 seeded slot", len(all))
	}
}

func BenchmarkDBRG(b *testing.B) {
	old := fips140.Enabled
	defer func() {
		fips140.Enabled = old
	}()
	fips140.Enabled = true

	const N = 64
	b.SetBytes(N)
	b.RunParallel(func(pb *testing.PB) {
		buf := make([]byte, N)
		for pb.Next() {
			Read(buf)
		}
	})
}
