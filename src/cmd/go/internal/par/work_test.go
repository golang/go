// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package par

import (
	"sync/atomic"
	"testing"
	"time"
)

func TestWork(t *testing.T) {
	var w Work

	const N = 10000
	n := int32(0)
	w.Add(N)
	w.Do(100, func(x interface{}) {
		atomic.AddInt32(&n, 1)
		i := x.(int)
		if i >= 2 {
			w.Add(i - 1)
			w.Add(i - 2)
		}
		w.Add(i >> 1)
		w.Add((i >> 1) ^ 1)
	})
	if n != N+1 {
		t.Fatalf("ran %d items, expected %d", n, N+1)
	}
}

func TestWorkParallel(t *testing.T) {
	for tries := 0; tries < 10; tries++ {
		var w Work
		const N = 100
		for i := 0; i < N; i++ {
			w.Add(i)
		}
		start := time.Now()
		var n int32
		w.Do(N, func(x interface{}) {
			time.Sleep(1 * time.Millisecond)
			atomic.AddInt32(&n, +1)
		})
		if n != N {
			t.Fatalf("par.Work.Do did not do all the work")
		}
		if time.Since(start) < N/2*time.Millisecond {
			return
		}
	}
	t.Fatalf("par.Work.Do does not seem to be parallel")
}

func TestCache(t *testing.T) {
	var cache Cache

	n := 1
	v := cache.Do(1, func() interface{} { n++; return n })
	if v != 2 {
		t.Fatalf("cache.Do(1) did not run f")
	}
	v = cache.Do(1, func() interface{} { n++; return n })
	if v != 2 {
		t.Fatalf("cache.Do(1) ran f again!")
	}
	v = cache.Do(2, func() interface{} { n++; return n })
	if v != 3 {
		t.Fatalf("cache.Do(2) did not run f")
	}
	v = cache.Do(1, func() interface{} { n++; return n })
	if v != 2 {
		t.Fatalf("cache.Do(1) did not returned saved value from original cache.Do(1)")
	}
}
