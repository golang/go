// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bcache

import (
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
)

var registeredCache Cache[int, int32]

func init() {
	registeredCache.Register()
}

var seq atomic.Uint32

func next[T int | int32]() *T {
	x := new(T)
	*x = T(seq.Add(1))
	return x
}

func str[T int | int32](x *T) string {
	if x == nil {
		return "nil"
	}
	return fmt.Sprint(*x)
}

func TestCache(t *testing.T) {
	// Use unregistered cache for functionality tests,
	// to keep the runtime from clearing behind our backs.
	c := new(Cache[int, int32])

	// Create many entries.
	m := make(map[*int]*int32)
	for i := 0; i < 10000; i++ {
		k := next[int]()
		v := next[int32]()
		m[k] = v
		c.Put(k, v)
	}

	// Overwrite a random 20% of those.
	n := 0
	for k := range m {
		v := next[int32]()
		m[k] = v
		c.Put(k, v)
		if n++; n >= 2000 {
			break
		}
	}

	// Check results.
	for k, v := range m {
		if cv := c.Get(k); cv != v {
			t.Fatalf("c.Get(%v) = %v, want %v", str(k), str(cv), str(v))
		}
	}

	c.Clear()
	for k := range m {
		if cv := c.Get(k); cv != nil {
			t.Fatalf("after GC, c.Get(%v) = %v, want nil", str(k), str(cv))
		}
	}

	// Check that registered cache is cleared at GC.
	c = &registeredCache
	for k, v := range m {
		c.Put(k, v)
	}
	runtime.GC()
	for k := range m {
		if cv := c.Get(k); cv != nil {
			t.Fatalf("after Clear, c.Get(%v) = %v, want nil", str(k), str(cv))
		}
	}

	// Check that cache works for concurrent access.
	// Lists are discarded if they reach 1000 entries,
	// and there are cacheSize list heads, so we should be
	// able to do 100 * cacheSize entries with no problem at all.
	c = new(Cache[int, int32])
	var barrier, wg sync.WaitGroup
	const N = 100
	barrier.Add(N)
	wg.Add(N)
	var lost int32
	for i := 0; i < N; i++ {
		go func() {
			defer wg.Done()

			m := make(map[*int]*int32)
			for j := 0; j < cacheSize; j++ {
				k, v := next[int](), next[int32]()
				m[k] = v
				c.Put(k, v)
			}
			barrier.Done()
			barrier.Wait()

			for k, v := range m {
				if cv := c.Get(k); cv != v {
					t.Errorf("c.Get(%v) = %v, want %v", str(k), str(cv), str(v))
					atomic.AddInt32(&lost, +1)
				}
			}
		}()
	}
	wg.Wait()
	if lost != 0 {
		t.Errorf("lost %d entries", lost)
	}
}
