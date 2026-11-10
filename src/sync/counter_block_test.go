// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sync_test

import (
	"runtime"
	"sync"
	"testing"
)

func TestAllocCounters(t *testing.T) {
	t.Run("single request", func(t *testing.T) {
		ps := runtime.GOMAXPROCS(0)

		// check for pointer uniqueness
		m := make(map[*uint64]bool)
		for ptr := range sync.AllocCounters(ps) {
			if _, ok := m[ptr]; ok {
				t.Errorf("duplicate ptrs detected %+v", ptr)
				continue
			}
			m[ptr] = true
		}
		if len(m) != ps {
			t.Errorf("want %d got %d", ps, len(m))
		}
	})

	t.Run("multiple request", func(t *testing.T) {
		ps := runtime.GOMAXPROCS(0)
		var m sync.Map
		var wg sync.WaitGroup

		singleMemRequest := func() {
			// check for pointer uniqueness and count
			var got int
			for ptr := range sync.AllocCounters(ps) {
				got++
				if _, ok := m.Load(ptr); ok {
					t.Errorf("duplicate ptrs detected %+v", ptr)
					continue
				}
				m.Store(ptr, true)
			}
			if got != ps {
				t.Errorf("want %d got %d", ps, got)
			}
		}
		for range 1000 {
			wg.Go(singleMemRequest)
		}
		wg.Wait()
	})
}

func TestAllocCountersGrow(t *testing.T) {
	testAlloc := func(want int) {
		var got int
		for range sync.AllocCounters(want) {
			got++
		}
		if got != want {
			t.Errorf("want %d got %d", want, got)
		}
	}
	testAlloc(runtime.GOMAXPROCS(0))
	testAlloc(2 * runtime.GOMAXPROCS(0))
}
