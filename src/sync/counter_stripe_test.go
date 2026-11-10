// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sync_test

import (
	"runtime"
	"sync"
	"testing"
	"unsafe"
)

// TestStripeShardsOnDistinctCacheLines checks the property the stripe exists
// to provide: every P's shard of a counter is a distinct, stable pointer that
// lives on its own cache line, so Adds on different Ps don't false-share.
func TestStripeShardsOnDistinctCacheLines(t *testing.T) {
	s := sync.NewStripe()
	owner := make(map[uintptr]int) // cache line -> pID
	for pID, p := range s.Block() {
		if p == nil {
			t.Fatalf("shard %d is nil", pID)
		}
		if got := s.Get(pID); got != p {
			t.Errorf("Get(%d) = %p, want stable pointer %p", pID, got, p)
		}
		line := uintptr(unsafe.Pointer(p)) / sync.CacheLineSize
		if other, ok := owner[line]; ok {
			t.Errorf("shards for P %d and P %d share cache line %#x", other, pID, line*sync.CacheLineSize)
		}
		owner[line] = pID
	}
}

func TestStripeGrow(t *testing.T) {
	if runtime.GOARCH == "wasm" {
		t.Skip("skipping on wasm: GOMAXPROCS cannot exceed 1")
	}
	initCap := runtime.GOMAXPROCS(0)
	t.Cleanup(func() {
		runtime.GOMAXPROCS(initCap)
	})

	s := sync.NewStripe()
	if got := s.Cap(); got != initCap {
		t.Fatalf("initial cap = %d, want %d", got, initCap)
	}

	// Snapshot the initial shard pointers to verify prefix preservation on grow.
	origPtrs := append([]*uint64(nil), s.Block()...)

	for idx := range initCap {
		sync.Runtime_procPin()
		n := s.Get(idx)
		*n += 2
		sync.Runtime_procUnpin()
	}

	// Double GOMAXPROCS so that an out-of-range pID triggers real stripe growth.
	updCap := initCap * 2
	runtime.GOMAXPROCS(updCap)

	for idx := range updCap {
		sync.Runtime_procPin()
		n := s.Get(idx)
		*n += 2
		sync.Runtime_procUnpin()
	}

	if got := s.Cap(); got != updCap {
		t.Fatalf("cap after grow = %d, want %d", got, updCap)
	}

	// Verify existing shard pointers were preserved and new shards were allocated.
	grownBlock := s.Block()
	for i := range initCap {
		if grownBlock[i] != origPtrs[i] {
			t.Errorf("shard %d pointer changed across grow: got %p, want %p", i, grownBlock[i], origPtrs[i])
		}
	}
	for i := initCap; i < updCap; i++ {
		if grownBlock[i] == nil {
			t.Errorf("newly grown shard %d is nil", i)
		}
	}

	var got uint64
	for idx := range s.Cap() {
		sync.Runtime_procPin()
		got += *s.Get(idx)
		sync.Runtime_procUnpin()
	}
	want := uint64(initCap*2 + updCap*2)
	if got != want {
		t.Errorf("Total sum of values want %d, got %d", want, got)
	}

	// Verify shrinking GOMAXPROCS does not shrink the stripe.
	runtime.GOMAXPROCS(initCap)
	s.Grow()
	if got := s.Cap(); got != updCap {
		t.Errorf("cap after GOMAXPROCS shrink = %d, want %d (must not shrink)", got, updCap)
	}
}
