// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sync_test

import (
	"fmt"
	isync "internal/sync"
	"math"
	"runtime"
	"strconv"
	"sync"
	"testing"
	"weak"
)

func TestHashTrieMap(t *testing.T) {
	testHashTrieMap(t, func() *isync.HashTrieMap[string, int] {
		var m isync.HashTrieMap[string, int]
		return &m
	})
}

func TestHashTrieMapBadHash(t *testing.T) {
	testHashTrieMap(t, func() *isync.HashTrieMap[string, int] {
		return isync.NewBadHashTrieMap[string, int]()
	})
}

func TestHashTrieMapTruncHash(t *testing.T) {
	testHashTrieMap(t, func() *isync.HashTrieMap[string, int] {
		// Stub out the good hash function with a different terrible one
		// (truncated hash). Everything should still work as expected.
		// This is useful to test independently to catch issues with
		// near collisions, where only the last few bits of the hash differ.
		return isync.NewTruncHashTrieMap[string, int]()
	})
}

func testHashTrieMap(t *testing.T, newMap func() *isync.HashTrieMap[string, int]) {
	t.Run("LoadEmpty", func(t *testing.T) {
		m := newMap()

		for _, s := range testData {
			expectMissing(t, s, 0)(m.Load(s))
		}
	})
	t.Run("LoadOrStore", func(t *testing.T) {
		m := newMap()

		for i, s := range testData {
			expectMissing(t, s, 0)(m.Load(s))
			expectStored(t, s, i)(m.LoadOrStore(s, i))
			expectPresent(t, s, i)(m.Load(s))
			expectLoaded(t, s, i)(m.LoadOrStore(s, 0))
		}
		for i, s := range testData {
			expectPresent(t, s, i)(m.Load(s))
			expectLoaded(t, s, i)(m.LoadOrStore(s, 0))
		}
	})
	t.Run("All", func(t *testing.T) {
		m := newMap()

		testAll(t, m, testDataMap(testData[:]), func(_ string, _ int) bool {
			return true
		})
	})
	t.Run("Clear", func(t *testing.T) {
		t.Run("Simple", func(t *testing.T) {
			m := newMap()

			for i, s := range testData {
				expectMissing(t, s, 0)(m.Load(s))
				expectStored(t, s, i)(m.LoadOrStore(s, i))
				expectPresent(t, s, i)(m.Load(s))
				expectLoaded(t, s, i)(m.LoadOrStore(s, 0))
			}
			m.Clear()
			for _, s := range testData {
				expectMissing(t, s, 0)(m.Load(s))
			}
		})
		t.Run("Concurrent", func(t *testing.T) {
			m := newMap()

			// Load up the map.
			for i, s := range testData {
				expectMissing(t, s, 0)(m.Load(s))
				expectStored(t, s, i)(m.LoadOrStore(s, i))
			}
			gmp := runtime.GOMAXPROCS(-1)
			var wg sync.WaitGroup
			for i := range gmp {
				wg.Add(1)
				go func(id int) {
					defer wg.Done()

					for _, s := range testData {
						// Try a couple things to interfere with the clear.
						expectNotDeleted(t, s, math.MaxInt)(m.CompareAndDelete(s, math.MaxInt))
						m.CompareAndSwap(s, i, i+1) // May succeed or fail; we don't care.
					}
				}(i)
			}

			// Concurrently clear the map.
			runtime.Gosched()
			m.Clear()

			// Wait for workers to finish.
			wg.Wait()

			// It should all be empty now.
			for _, s := range testData {
				expectMissing(t, s, 0)(m.Load(s))
			}
		})
	})
	t.Run("CompareAndDelete", func(t *testing.T) {
		t.Run("All", func(t *testing.T) {
			m := newMap()

			for range 3 {
				for i, s := range testData {
					expectMissing(t, s, 0)(m.Load(s))
					expectStored(t, s, i)(m.LoadOrStore(s, i))
					expectPresent(t, s, i)(m.Load(s))
					expectLoaded(t, s, i)(m.LoadOrStore(s, 0))
				}
				for i, s := range testData {
					expectPresent(t, s, i)(m.Load(s))
					expectNotDeleted(t, s, math.MaxInt)(m.CompareAndDelete(s, math.MaxInt))
					expectDeleted(t, s, i)(m.CompareAndDelete(s, i))
					expectNotDeleted(t, s, i)(m.CompareAndDelete(s, i))
					expectMissing(t, s, 0)(m.Load(s))
				}
				for _, s := range testData {
					expectMissing(t, s, 0)(m.Load(s))
				}
			}
		})
		t.Run("One", func(t *testing.T) {
			m := newMap()

			for i, s := range testData {
				expectMissing(t, s, 0)(m.Load(s))
				expectStored(t, s, i)(m.LoadOrStore(s, i))
				expectPresent(t, s, i)(m.Load(s))
				expectLoaded(t, s, i)(m.LoadOrStore(s, 0))
			}
			expectNotDeleted(t, testData[15], math.MaxInt)(m.CompareAndDelete(testData[15], math.MaxInt))
			expectDeleted(t, testData[15], 15)(m.CompareAndDelete(testData[15], 15))
			expectNotDeleted(t, testData[15], 15)(m.CompareAndDelete(testData[15], 15))
			for i, s := range testData {
				if i == 15 {
					expectMissing(t, s, 0)(m.Load(s))
				} else {
					expectPresent(t, s, i)(m.Load(s))
				}
			}
		})
		t.Run("Multiple", func(t *testing.T) {
			m := newMap()

			for i, s := range testData {
				expectMissing(t, s, 0)(m.Load(s))
				expectStored(t, s, i)(m.LoadOrStore(s, i))
				expectPresent(t, s, i)(m.Load(s))
				expectLoaded(t, s, i)(m.LoadOrStore(s, 0))
			}
			for _, i := range []int{1, 105, 6, 85} {
				expectNotDeleted(t, testData[i], math.MaxInt)(m.CompareAndDelete(testData[i], math.MaxInt))
				expectDeleted(t, testData[i], i)(m.CompareAndDelete(testData[i], i))
				expectNotDeleted(t, testData[i], i)(m.CompareAndDelete(testData[i], i))
			}
			for i, s := range testData {
				if i == 1 || i == 105 || i == 6 || i == 85 {
					expectMissing(t, s, 0)(m.Load(s))
				} else {
					expectPresent(t, s, i)(m.Load(s))
				}
			}
		})
		t.Run("Iterate", func(t *testing.T) {
			m := newMap()

			testAll(t, m, testDataMap(testData[:]), func(s string, i int) bool {
				expectDeleted(t, s, i)(m.CompareAndDelete(s, i))
				return true
			})
			for _, s := range testData {
				expectMissing(t, s, 0)(m.Load(s))
			}
		})
		t.Run("ConcurrentUnsharedKeys", func(t *testing.T) {
			m := newMap()

			gmp := runtime.GOMAXPROCS(-1)
			var wg sync.WaitGroup
			for i := range gmp {
				wg.Add(1)
				go func(id int) {
					defer wg.Done()

					makeKey := func(s string) string {
						return s + "-" + strconv.Itoa(id)
					}
					for _, s := range testData {
						key := makeKey(s)
						expectMissing(t, key, 0)(m.Load(key))
						expectStored(t, key, id)(m.LoadOrStore(key, id))
						expectPresent(t, key, id)(m.Load(key))
						expectLoaded(t, key, id)(m.LoadOrStore(key, 0))
					}
					for _, s := range testData {
						key := makeKey(s)
						expectPresent(t, key, id)(m.Load(key))
						expectDeleted(t, key, id)(m.CompareAndDelete(key, id))
						expectMissing(t, key, 0)(m.Load(key))
					}
					for _, s := range testData {
						key := makeKey(s)
						expectMissing(t, key, 0)(m.Load(key))
					}
				}(i)
			}
			wg.Wait()
		})
		t.Run("ConcurrentSharedKeys", func(t *testing.T) {
			m := newMap()

			// Load up the map.
			for i, s := range testData {
				expectMissing(t, s, 0)(m.Load(s))
				expectStored(t, s, i)(m.LoadOrStore(s, i))
			}
			gmp := runtime.GOMAXPROCS(-1)
			var wg sync.WaitGroup
			for i := range gmp {
				wg.Add(1)
				go func(id int) {
					defer wg.Done()

					for i, s := range testData {
						expectNotDeleted(t, s, math.MaxInt)(m.CompareAndDelete(s, math.MaxInt))
						m.CompareAndDelete(s, i)
						expectMissing(t, s, 0)(m.Load(s))
					}
					for _, s := range testData {
						expectMissing(t, s, 0)(m.Load(s))
					}
				}(i)
			}
			wg.Wait()
		})
	})
	t.Run("CompareAndSwap", func(t *testing.T) {
		t.Run("All", func(t *testing.T) {
			m := newMap()

			for i, s := range testData {
				expectMissing(t, s, 0)(m.Load(s))
				expectStored(t, s, i)(m.LoadOrStore(s, i))
				expectPresent(t, s, i)(m.Load(s))
				expectLoaded(t, s, i)(m.LoadOrStore(s, 0))
			}
			for j := range 3 {
				for i, s := range testData {
					expectPresent(t, s, i+j)(m.Load(s))
					expectNotSwapped(t, s, math.MaxInt, i+j+1)(m.CompareAndSwap(s, math.MaxInt, i+j+1))
					expectSwapped(t, s, i, i+j+1)(m.CompareAndSwap(s, i+j, i+j+1))
					expectNotSwapped(t, s, i+j, i+j+1)(m.CompareAndSwap(s, i+j, i+j+1))
					expectPresent(t, s, i+j+1)(m.Load(s))
				}
			}
			for i, s := range testData {
				expectPresent(t, s, i+3)(m.Load(s))
			}
		})
		t.Run("One", func(t *testing.T) {
			m := newMap()

			for i, s := range testData {
				expectMissing(t, s, 0)(m.Load(s))
				expectStored(t, s, i)(m.LoadOrStore(s, i))
				expectPresent(t, s, i)(m.Load(s))
				expectLoaded(t, s, i)(m.LoadOrStore(s, 0))
			}
			expectNotSwapped(t, testData[15], math.MaxInt, 16)(m.CompareAndSwap(testData[15], math.MaxInt, 16))
			expectSwapped(t, testData[15], 15, 16)(m.CompareAndSwap(testData[15], 15, 16))
			expectNotSwapped(t, testData[15], 15, 16)(m.CompareAndSwap(testData[15], 15, 16))
			for i, s := range testData {
				if i == 15 {
					expectPresent(t, s, 16)(m.Load(s))
				} else {
					expectPresent(t, s, i)(m.Load(s))
				}
			}
		})
		t.Run("Multiple", func(t *testing.T) {
			m := newMap()

			for i, s := range testData {
				expectMissing(t, s, 0)(m.Load(s))
				expectStored(t, s, i)(m.LoadOrStore(s, i))
				expectPresent(t, s, i)(m.Load(s))
				expectLoaded(t, s, i)(m.LoadOrStore(s, 0))
			}
			for _, i := range []int{1, 105, 6, 85} {
				expectNotSwapped(t, testData[i], math.MaxInt, i+1)(m.CompareAndSwap(testData[i], math.MaxInt, i+1))
				expectSwapped(t, testData[i], i, i+1)(m.CompareAndSwap(testData[i], i, i+1))
				expectNotSwapped(t, testData[i], i, i+1)(m.CompareAndSwap(testData[i], i, i+1))
			}
			for i, s := range testData {
				if i == 1 || i == 105 || i == 6 || i == 85 {
					expectPresent(t, s, i+1)(m.Load(s))
				} else {
					expectPresent(t, s, i)(m.Load(s))
				}
			}
		})

		t.Run("ConcurrentUnsharedKeys", func(t *testing.T) {
			m := newMap()

			gmp := runtime.GOMAXPROCS(-1)
			var wg sync.WaitGroup
			for i := range gmp {
				wg.Add(1)
				go func(id int) {
					defer wg.Done()

					makeKey := func(s string) string {
						return s + "-" + strconv.Itoa(id)
					}
					for _, s := range testData {
						key := makeKey(s)
						expectMissing(t, key, 0)(m.Load(key))
						expectStored(t, key, id)(m.LoadOrStore(key, id))
						expectPresent(t, key, id)(m.Load(key))
						expectLoaded(t, key, id)(m.LoadOrStore(key, 0))
					}
					for _, s := range testData {
						key := makeKey(s)
						expectPresent(t, key, id)(m.Load(key))
						expectSwapped(t, key, id, id+1)(m.CompareAndSwap(key, id, id+1))
						expectPresent(t, key, id+1)(m.Load(key))
					}
					for _, s := range testData {
						key := makeKey(s)
						expectPresent(t, key, id+1)(m.Load(key))
					}
				}(i)
			}
			wg.Wait()
		})
		t.Run("ConcurrentUnsharedKeysWithDelete", func(t *testing.T) {
			m := newMap()

			gmp := runtime.GOMAXPROCS(-1)
			var wg sync.WaitGroup
			for i := range gmp {
				wg.Add(1)
				go func(id int) {
					defer wg.Done()

					makeKey := func(s string) string {
						return s + "-" + strconv.Itoa(id)
					}
					for _, s := range testData {
						key := makeKey(s)
						expectMissing(t, key, 0)(m.Load(key))
						expectStored(t, key, id)(m.LoadOrStore(key, id))
						expectPresent(t, key, id)(m.Load(key))
						expectLoaded(t, key, id)(m.LoadOrStore(key, 0))
					}
					for _, s := range testData {
						key := makeKey(s)
						expectPresent(t, key, id)(m.Load(key))
						expectSwapped(t, key, id, id+1)(m.CompareAndSwap(key, id, id+1))
						expectPresent(t, key, id+1)(m.Load(key))
						expectDeleted(t, key, id+1)(m.CompareAndDelete(key, id+1))
						expectNotSwapped(t, key, id+1, id+2)(m.CompareAndSwap(key, id+1, id+2))
						expectNotDeleted(t, key, id+1)(m.CompareAndDelete(key, id+1))
						expectMissing(t, key, 0)(m.Load(key))
					}
					for _, s := range testData {
						key := makeKey(s)
						expectMissing(t, key, 0)(m.Load(key))
					}
				}(i)
			}
			wg.Wait()
		})
		t.Run("ConcurrentSharedKeys", func(t *testing.T) {
			m := newMap()

			// Load up the map.
			for i, s := range testData {
				expectMissing(t, s, 0)(m.Load(s))
				expectStored(t, s, i)(m.LoadOrStore(s, i))
			}
			gmp := runtime.GOMAXPROCS(-1)
			var wg sync.WaitGroup
			for i := range gmp {
				wg.Add(1)
				go func(id int) {
					defer wg.Done()

					for i, s := range testData {
						expectNotSwapped(t, s, math.MaxInt, i+1)(m.CompareAndSwap(s, math.MaxInt, i+1))
						m.CompareAndSwap(s, i, i+1)
						expectPresent(t, s, i+1)(m.Load(s))
					}
					for i, s := range testData {
						expectPresent(t, s, i+1)(m.Load(s))
					}
				}(i)
			}
			wg.Wait()
		})
	})
	t.Run("Swap", func(t *testing.T) {
		t.Run("All", func(t *testing.T) {
			m := newMap()

			for i, s := range testData {
				expectMissing(t, s, 0)(m.Load(s))
				expectNotLoadedFromSwap(t, s, i)(m.Swap(s, i))
				expectPresent(t, s, i)(m.Load(s))
				expectLoadedFromSwap(t, s, i, i)(m.Swap(s, i))
			}
			for j := range 3 {
				for i, s := range testData {
					expectPresent(t, s, i+j)(m.Load(s))
					expectLoadedFromSwap(t, s, i+j, i+j+1)(m.Swap(s, i+j+1))
					expectPresent(t, s, i+j+1)(m.Load(s))
				}
			}
			for i, s := range testData {
				expectLoadedFromSwap(t, s, i+3, i+3)(m.Swap(s, i+3))
			}
		})
		t.Run("One", func(t *testing.T) {
			m := newMap()

			for i, s := range testData {
				expectMissing(t, s, 0)(m.Load(s))
				expectNotLoadedFromSwap(t, s, i)(m.Swap(s, i))
				expectPresent(t, s, i)(m.Load(s))
				expectLoadedFromSwap(t, s, i, i)(m.Swap(s, i))
			}
			expectLoadedFromSwap(t, testData[15], 15, 16)(m.Swap(testData[15], 16))
			for i, s := range testData {
				if i == 15 {
					expectPresent(t, s, 16)(m.Load(s))
				} else {
					expectPresent(t, s, i)(m.Load(s))
				}
			}
		})
		t.Run("Multiple", func(t *testing.T) {
			m := newMap()

			for i, s := range testData {
				expectMissing(t, s, 0)(m.Load(s))
				expectNotLoadedFromSwap(t, s, i)(m.Swap(s, i))
				expectPresent(t, s, i)(m.Load(s))
				expectLoadedFromSwap(t, s, i, i)(m.Swap(s, i))
			}
			for _, i := range []int{1, 105, 6, 85} {
				expectLoadedFromSwap(t, testData[i], i, i+1)(m.Swap(testData[i], i+1))
			}
			for i, s := range testData {
				if i == 1 || i == 105 || i == 6 || i == 85 {
					expectPresent(t, s, i+1)(m.Load(s))
				} else {
					expectPresent(t, s, i)(m.Load(s))
				}
			}
		})
		t.Run("ConcurrentUnsharedKeys", func(t *testing.T) {
			m := newMap()

			gmp := runtime.GOMAXPROCS(-1)
			var wg sync.WaitGroup
			for i := range gmp {
				wg.Add(1)
				go func(id int) {
					defer wg.Done()

					makeKey := func(s string) string {
						return s + "-" + strconv.Itoa(id)
					}
					for _, s := range testData {
						key := makeKey(s)
						expectMissing(t, key, 0)(m.Load(key))
						expectNotLoadedFromSwap(t, key, id)(m.Swap(key, id))
						expectPresent(t, key, id)(m.Load(key))
						expectLoadedFromSwap(t, key, id, id)(m.Swap(key, id))
					}
					for _, s := range testData {
						key := makeKey(s)
						expectPresent(t, key, id)(m.Load(key))
						expectLoadedFromSwap(t, key, id, id+1)(m.Swap(key, id+1))
						expectPresent(t, key, id+1)(m.Load(key))
					}
					for _, s := range testData {
						key := makeKey(s)
						expectPresent(t, key, id+1)(m.Load(key))
					}
				}(i)
			}
			wg.Wait()
		})
		t.Run("ConcurrentUnsharedKeysWithDelete", func(t *testing.T) {
			m := newMap()

			gmp := runtime.GOMAXPROCS(-1)
			var wg sync.WaitGroup
			for i := range gmp {
				wg.Add(1)
				go func(id int) {
					defer wg.Done()

					makeKey := func(s string) string {
						return s + "-" + strconv.Itoa(id)
					}
					for _, s := range testData {
						key := makeKey(s)
						expectMissing(t, key, 0)(m.Load(key))
						expectNotLoadedFromSwap(t, key, id)(m.Swap(key, id))
						expectPresent(t, key, id)(m.Load(key))
						expectLoadedFromSwap(t, key, id, id)(m.Swap(key, id))
					}
					for _, s := range testData {
						key := makeKey(s)
						expectPresent(t, key, id)(m.Load(key))
						expectLoadedFromSwap(t, key, id, id+1)(m.Swap(key, id+1))
						expectPresent(t, key, id+1)(m.Load(key))
						expectDeleted(t, key, id+1)(m.CompareAndDelete(key, id+1))
						expectNotLoadedFromSwap(t, key, id+2)(m.Swap(key, id+2))
						expectPresent(t, key, id+2)(m.Load(key))
					}
					for _, s := range testData {
						key := makeKey(s)
						expectPresent(t, key, id+2)(m.Load(key))
					}
				}(i)
			}
			wg.Wait()
		})
		t.Run("ConcurrentSharedKeys", func(t *testing.T) {
			m := newMap()

			// Load up the map.
			for i, s := range testData {
				expectMissing(t, s, 0)(m.Load(s))
				expectStored(t, s, i)(m.LoadOrStore(s, i))
			}
			gmp := runtime.GOMAXPROCS(-1)
			var wg sync.WaitGroup
			for i := range gmp {
				wg.Add(1)
				go func(id int) {
					defer wg.Done()

					for i, s := range testData {
						m.Swap(s, i+1)
						expectPresent(t, s, i+1)(m.Load(s))
					}
					for i, s := range testData {
						expectPresent(t, s, i+1)(m.Load(s))
					}
				}(i)
			}
			wg.Wait()
		})
	})
	t.Run("LoadAndDelete", func(t *testing.T) {
		t.Run("All", func(t *testing.T) {
			m := newMap()

			for range 3 {
				for i, s := range testData {
					expectMissing(t, s, 0)(m.Load(s))
					expectStored(t, s, i)(m.LoadOrStore(s, i))
					expectPresent(t, s, i)(m.Load(s))
					expectLoaded(t, s, i)(m.LoadOrStore(s, 0))
				}
				for i, s := range testData {
					expectPresent(t, s, i)(m.Load(s))
					expectLoadedFromDelete(t, s, i)(m.LoadAndDelete(s))
					expectMissing(t, s, 0)(m.Load(s))
					expectNotLoadedFromDelete(t, s, 0)(m.LoadAndDelete(s))
				}
				for _, s := range testData {
					expectMissing(t, s, 0)(m.Load(s))
				}
			}
		})
		t.Run("One", func(t *testing.T) {
			m := newMap()

			for i, s := range testData {
				expectMissing(t, s, 0)(m.Load(s))
				expectStored(t, s, i)(m.LoadOrStore(s, i))
				expectPresent(t, s, i)(m.Load(s))
				expectLoaded(t, s, i)(m.LoadOrStore(s, 0))
			}
			expectPresent(t, testData[15], 15)(m.Load(testData[15]))
			expectLoadedFromDelete(t, testData[15], 15)(m.LoadAndDelete(testData[15]))
			expectMissing(t, testData[15], 0)(m.Load(testData[15]))
			expectNotLoadedFromDelete(t, testData[15], 0)(m.LoadAndDelete(testData[15]))
			for i, s := range testData {
				if i == 15 {
					expectMissing(t, s, 0)(m.Load(s))
				} else {
					expectPresent(t, s, i)(m.Load(s))
				}
			}
		})
		t.Run("Multiple", func(t *testing.T) {
			m := newMap()

			for i, s := range testData {
				expectMissing(t, s, 0)(m.Load(s))
				expectStored(t, s, i)(m.LoadOrStore(s, i))
				expectPresent(t, s, i)(m.Load(s))
				expectLoaded(t, s, i)(m.LoadOrStore(s, 0))
			}
			for _, i := range []int{1, 105, 6, 85} {
				expectPresent(t, testData[i], i)(m.Load(testData[i]))
				expectLoadedFromDelete(t, testData[i], i)(m.LoadAndDelete(testData[i]))
				expectMissing(t, testData[i], 0)(m.Load(testData[i]))
				expectNotLoadedFromDelete(t, testData[i], 0)(m.LoadAndDelete(testData[i]))
			}
			for i, s := range testData {
				if i == 1 || i == 105 || i == 6 || i == 85 {
					expectMissing(t, s, 0)(m.Load(s))
				} else {
					expectPresent(t, s, i)(m.Load(s))
				}
			}
		})
		t.Run("Iterate", func(t *testing.T) {
			m := newMap()

			testAll(t, m, testDataMap(testData[:]), func(s string, i int) bool {
				expectLoadedFromDelete(t, s, i)(m.LoadAndDelete(s))
				return true
			})
			for _, s := range testData {
				expectMissing(t, s, 0)(m.Load(s))
			}
		})
		t.Run("ConcurrentUnsharedKeys", func(t *testing.T) {
			m := newMap()

			gmp := runtime.GOMAXPROCS(-1)
			var wg sync.WaitGroup
			for i := range gmp {
				wg.Add(1)
				go func(id int) {
					defer wg.Done()

					makeKey := func(s string) string {
						return s + "-" + strconv.Itoa(id)
					}
					for _, s := range testData {
						key := makeKey(s)
						expectMissing(t, key, 0)(m.Load(key))
						expectStored(t, key, id)(m.LoadOrStore(key, id))
						expectPresent(t, key, id)(m.Load(key))
						expectLoaded(t, key, id)(m.LoadOrStore(key, 0))
					}
					for _, s := range testData {
						key := makeKey(s)
						expectPresent(t, key, id)(m.Load(key))
						expectLoadedFromDelete(t, key, id)(m.LoadAndDelete(key))
						expectMissing(t, key, 0)(m.Load(key))
					}
					for _, s := range testData {
						key := makeKey(s)
						expectMissing(t, key, 0)(m.Load(key))
					}
				}(i)
			}
			wg.Wait()
		})
		t.Run("ConcurrentSharedKeys", func(t *testing.T) {
			m := newMap()

			// Load up the map.
			for i, s := range testData {
				expectMissing(t, s, 0)(m.Load(s))
				expectStored(t, s, i)(m.LoadOrStore(s, i))
			}
			gmp := runtime.GOMAXPROCS(-1)
			var wg sync.WaitGroup
			for i := range gmp {
				wg.Add(1)
				go func(id int) {
					defer wg.Done()

					for _, s := range testData {
						m.LoadAndDelete(s)
						expectMissing(t, s, 0)(m.Load(s))
					}
					for _, s := range testData {
						expectMissing(t, s, 0)(m.Load(s))
					}
				}(i)
			}
			wg.Wait()
		})
	})
}

func testAll[K, V comparable](t *testing.T, m *isync.HashTrieMap[K, V], testData map[K]V, yield func(K, V) bool) {
	for k, v := range testData {
		expectStored(t, k, v)(m.LoadOrStore(k, v))
	}
	visited := make(map[K]int)
	m.All()(func(key K, got V) bool {
		want, ok := testData[key]
		if !ok {
			t.Errorf("unexpected key %v in map", key)
			return false
		}
		if got != want {
			t.Errorf("expected key %v to have value %v, got %v", key, want, got)
			return false
		}
		visited[key]++
		return yield(key, got)
	})
	for key, n := range visited {
		if n > 1 {
			t.Errorf("visited key %v more than once", key)
		}
	}
}

func expectPresent[K, V comparable](t *testing.T, key K, want V) func(got V, ok bool) {
	t.Helper()
	return func(got V, ok bool) {
		t.Helper()

		if !ok {
			t.Errorf("expected key %v to be present in map", key)
		}
		if ok && got != want {
			t.Errorf("expected key %v to have value %v, got %v", key, want, got)
		}
	}
}

func expectMissing[K, V comparable](t *testing.T, key K, want V) func(got V, ok bool) {
	t.Helper()
	if want != *new(V) {
		// This is awkward, but the want argument is necessary to smooth over type inference.
		// Just make sure the want argument always looks the same.
		panic("expectMissing must always have a zero value variable")
	}
	return func(got V, ok bool) {
		t.Helper()

		if ok {
			t.Errorf("expected key %v to be missing from map, got value %v", key, got)
		}
		if !ok && got != want {
			t.Errorf("expected missing key %v to be paired with the zero value; got %v", key, got)
		}
	}
}

func expectLoaded[K, V comparable](t *testing.T, key K, want V) func(got V, loaded bool) {
	t.Helper()
	return func(got V, loaded bool) {
		t.Helper()

		if !loaded {
			t.Errorf("expected key %v to have been loaded, not stored", key)
		}
		if got != want {
			t.Errorf("expected key %v to have value %v, got %v", key, want, got)
		}
	}
}

func expectStored[K, V comparable](t *testing.T, key K, want V) func(got V, loaded bool) {
	t.Helper()
	return func(got V, loaded bool) {
		t.Helper()

		if loaded {
			t.Errorf("expected inserted key %v to have been stored, not loaded", key)
		}
		if got != want {
			t.Errorf("expected inserted key %v to have value %v, got %v", key, want, got)
		}
	}
}

func expectDeleted[K, V comparable](t *testing.T, key K, old V) func(deleted bool) {
	t.Helper()
	return func(deleted bool) {
		t.Helper()

		if !deleted {
			t.Errorf("expected key %v with value %v to be in map and deleted", key, old)
		}
	}
}

func expectNotDeleted[K, V comparable](t *testing.T, key K, old V) func(deleted bool) {
	t.Helper()
	return func(deleted bool) {
		t.Helper()

		if deleted {
			t.Errorf("expected key %v with value %v to not be in map and thus not deleted", key, old)
		}
	}
}

func expectSwapped[K, V comparable](t *testing.T, key K, old, new V) func(swapped bool) {
	t.Helper()
	return func(swapped bool) {
		t.Helper()

		if !swapped {
			t.Errorf("expected key %v with value %v to be in map and swapped for %v", key, old, new)
		}
	}
}

func expectNotSwapped[K, V comparable](t *testing.T, key K, old, new V) func(swapped bool) {
	t.Helper()
	return func(swapped bool) {
		t.Helper()

		if swapped {
			t.Errorf("expected key %v with value %v to not be in map or not swapped for %v", key, old, new)
		}
	}
}

func expectLoadedFromSwap[K, V comparable](t *testing.T, key K, want, new V) func(got V, loaded bool) {
	t.Helper()
	return func(got V, loaded bool) {
		t.Helper()

		if !loaded {
			t.Errorf("expected key %v to be in map and for %v to have been swapped for %v", key, want, new)
		} else if want != got {
			t.Errorf("key %v had its value %v swapped for %v, but expected it to have value %v", key, got, new, want)
		}
	}
}

func expectNotLoadedFromSwap[K, V comparable](t *testing.T, key K, new V) func(old V, loaded bool) {
	t.Helper()
	return func(old V, loaded bool) {
		t.Helper()

		if loaded {
			t.Errorf("expected key %v to not be in map, but found value %v for it", key, old)
		}
	}
}

func expectLoadedFromDelete[K, V comparable](t *testing.T, key K, want V) func(got V, loaded bool) {
	t.Helper()
	return func(got V, loaded bool) {
		t.Helper()

		if !loaded {
			t.Errorf("expected key %v to be in map to be deleted", key)
		} else if want != got {
			t.Errorf("key %v was deleted with value %v, but expected it to have value %v", key, got, want)
		}
	}
}

func expectNotLoadedFromDelete[K, V comparable](t *testing.T, key K, _ V) func(old V, loaded bool) {
	t.Helper()
	return func(old V, loaded bool) {
		t.Helper()

		if loaded {
			t.Errorf("expected key %v to not be in map, but found value %v for it", key, old)
		}
	}
}

func testDataMap(data []string) map[string]int {
	m := make(map[string]int)
	for i, s := range data {
		m[s] = i
	}
	return m
}

var (
	testDataSmall [8]string
	testData      [128]string
	testDataLarge [128 << 10]string
)

func init() {
	for i := range testDataSmall {
		testDataSmall[i] = fmt.Sprintf("%b", i)
	}
	for i := range testData {
		testData[i] = fmt.Sprintf("%b", i)
	}
	for i := range testDataLarge {
		testDataLarge[i] = fmt.Sprintf("%b", i)
	}
}

// TestConcurrentCache tests HashTrieMap in a scenario where it is used as
// the basis of a memory-efficient concurrent cache. We're specifically
// looking to make sure that CompareAndSwap and CompareAndDelete are
// atomic with respect to one another. When competing for the same
// key-value pair, they must not both succeed.
//
// This test is a regression test for issue #70970.
func TestConcurrentCache(t *testing.T) {
	type dummy [32]byte

	var m isync.HashTrieMap[int, weak.Pointer[dummy]]

	type cleanupArg struct {
		key   int
		value weak.Pointer[dummy]
	}
	cleanup := func(arg cleanupArg) {
		m.CompareAndDelete(arg.key, arg.value)
	}
	get := func(m *isync.HashTrieMap[int, weak.Pointer[dummy]], key int) *dummy {
		nv := new(dummy)
		nw := weak.Make(nv)
		for {
			w, loaded := m.LoadOrStore(key, nw)
			if !loaded {
				runtime.AddCleanup(nv, cleanup, cleanupArg{key, nw})
				return nv
			}
			if v := w.Value(); v != nil {
				return v
			}

			// Weak pointer was reclaimed, try to replace it with nw.
			if m.CompareAndSwap(key, w, nw) {
				runtime.AddCleanup(nv, cleanup, cleanupArg{key, nw})
				return nv
			}
		}
	}

	const N = 100_000
	const P = 5_000

	var wg sync.WaitGroup
	wg.Add(N)
	for i := range N {
		go func() {
			defer wg.Done()
			a := get(&m, i%P)
			b := get(&m, i%P)
			if a != b {
				t.Errorf("consecutive cache reads returned different values: a != b (%p vs %p)\n", a, b)
			}
		}()
	}
	wg.Wait()
}
