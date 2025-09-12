// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unique

import (
	"internal/abi"
	"runtime"
	"strconv"
	"sync"
	"testing"
	"unsafe"
)

func TestCanonMap(t *testing.T) {
	testCanonMap(t, func() *canonMap[string] {
		return newCanonMap[string]()
	})
}

func TestCanonMapBadHash(t *testing.T) {
	testCanonMap(t, func() *canonMap[string] {
		return newBadCanonMap[string]()
	})
}

func TestCanonMapTruncHash(t *testing.T) {
	testCanonMap(t, func() *canonMap[string] {
		// Stub out the good hash function with a different terrible one
		// (truncated hash). Everything should still work as expected.
		// This is useful to test independently to catch issues with
		// near collisions, where only the last few bits of the hash differ.
		return newTruncCanonMap[string]()
	})
}

func testCanonMap(t *testing.T, newMap func() *canonMap[string]) {
	t.Run("LoadEmpty", func(t *testing.T) {
		m := newMap()

		for _, s := range testData {
			expectMissing(t, s)(m.Load(s))
		}
	})
	t.Run("LoadOrStore", func(t *testing.T) {
		t.Run("Sequential", func(t *testing.T) {
			m := newMap()

			var refs []*string
			for _, s := range testData {
				expectMissing(t, s)(m.Load(s))
				refs = append(refs, expectPresent(t, s)(m.LoadOrStore(s)))
				expectPresent(t, s)(m.Load(s))
				expectPresent(t, s)(m.LoadOrStore(s))
			}
			drainCleanupQueue(t)
			for _, s := range testData {
				expectPresent(t, s)(m.Load(s))
				expectPresent(t, s)(m.LoadOrStore(s))
			}
			runtime.KeepAlive(refs)
			refs = nil
			drainCleanupQueue(t)
			for _, s := range testData {
				expectMissing(t, s)(m.Load(s))
				expectPresent(t, s)(m.LoadOrStore(s))
			}
		})
		t.Run("ConcurrentUnsharedKeys", func(t *testing.T) {
			makeKey := func(s string, id int) string {
				return s + "-" + strconv.Itoa(id)
			}

			// Expand and shrink the map multiple times to try to get
			// insertions and cleanups to overlap.
			m := newMap()
			gmp := runtime.GOMAXPROCS(-1)
			for try := range 3 {
				var wg sync.WaitGroup
				for i := range gmp {
					wg.Add(1)
					go func(id int) {
						defer wg.Done()

						var refs []*string
						for _, s := range testData {
							key := makeKey(s, id)
							if try == 0 {
								expectMissing(t, key)(m.Load(key))
							}
							refs = append(refs, expectPresent(t, key)(m.LoadOrStore(key)))
							expectPresent(t, key)(m.Load(key))
							expectPresent(t, key)(m.LoadOrStore(key))
						}
						for i, s := range testData {
							key := makeKey(s, id)
							expectPresent(t, key)(m.Load(key))
							if got, want := expectPresent(t, key)(m.LoadOrStore(key)), refs[i]; got != want {
								t.Errorf("canonical entry %p did not match ref %p", got, want)
							}
						}
						// N.B. We avoid trying to test entry cleanup here
						// because it's going to be very flaky, especially
						// in the bad hash cases.
					}(i)
				}
				wg.Wait()
			}

			// Run an extra GC cycle to de-flake. Sometimes the cleanups
			// fail to run in time, despite drainCleanupQueue.
			//
			// TODO(mknyszek): Figure out why the extra GC is necessary,
			// and what is transiently keeping the cleanups live.
			// * I have confirmed that they are not completely stuck, and
			//   they always eventually run.
			// * I have also confirmed it's not asynchronous preemption
			//   keeping them around (though that is a possibility).
			// * I have confirmed that they are not simply sitting on
			//   the queue, and that drainCleanupQueue is just failing
			//   to actually empty the queue.
			// * I have confirmed that it's not a write barrier that's
			//   keeping it alive, nor is it a weak pointer dereference
			//   (which shades the object during the GC).
			// The corresponding objects do seem to be transiently truly
			// reachable, but I have no idea by what path.
			runtime.GC()

			// Drain cleanups so everything is deleted.
			drainCleanupQueue(t)

			// Double-check that it's all gone.
			for id := range gmp {
				makeKey := func(s string) string {
					return s + "-" + strconv.Itoa(id)
				}
				for _, s := range testData {
					key := makeKey(s)
					expectMissing(t, key)(m.Load(key))
				}
			}
		})
	})
}

func expectMissing[T comparable](t *testing.T, key T) func(got *T) {
	t.Helper()
	return func(got *T) {
		t.Helper()

		if got != nil {
			t.Errorf("expected key %v to be missing from map, got %p", key, got)
		}
	}
}

func expectPresent[T comparable](t *testing.T, key T) func(got *T) *T {
	t.Helper()
	return func(got *T) *T {
		t.Helper()

		if got == nil {
			t.Errorf("expected key %v to be present in map, got %p", key, got)
		}
		if got != nil && *got != key {
			t.Errorf("key %v is present in map, but canonical version has the wrong value: got %v, want %v", key, *got, key)
		}
		return got
	}
}

// newBadCanonMap creates a new canonMap for the provided key type
// but with an intentionally bad hash function.
func newBadCanonMap[T comparable]() *canonMap[T] {
	// Stub out the good hash function with a terrible one.
	// Everything should still work as expected.
	m := newCanonMap[T]()
	m.hash = func(_ unsafe.Pointer, _ uintptr) uintptr {
		return 0
	}
	return m
}

// newTruncCanonMap creates a new canonMap for the provided key type
// but with an intentionally bad hash function.
func newTruncCanonMap[T comparable]() *canonMap[T] {
	// Stub out the good hash function with a terrible one.
	// Everything should still work as expected.
	m := newCanonMap[T]()
	var mx map[string]int
	mapType := abi.TypeOf(mx).MapType()
	hasher := mapType.Hasher
	m.hash = func(p unsafe.Pointer, n uintptr) uintptr {
		return hasher(p, n) & ((uintptr(1) << 4) - 1)
	}
	return m
}
