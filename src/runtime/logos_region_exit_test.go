// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.logosarena

package runtime_test

import (
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"
	"unsafe"
)

func TestLogosRegionExitRecyclesOuterRecords(t *testing.T) {
	h1, _ := runtime.LogosRegionOpen()
	if h1 == nil {
		t.Fatal("first open failed")
	}
	h2, _ := runtime.LogosRegionOpen()
	if h2 == nil {
		t.Fatal("second open failed")
	}
	runtime.LogosRegionExitForTest()
	a := runtime.LogosRegionTakePForTest()
	b := runtime.LogosRegionTakePForTest()
	if a == nil || b == nil {
		t.Fatalf("pool takes = %v, %v, want both records recycled", a, b)
	}
	ok := (a == h1 && b == h2) || (a == h2 && b == h1)
	if !ok {
		t.Fatalf("recycled = %v, %v, want {%v %v}", a, b, unsafe.Pointer(h1), unsafe.Pointer(h2))
	}
}

func TestLogosRegionHandleForeignUseRaceClean(t *testing.T) {
	h, gen := runtime.LogosRegionOpen()
	if h == nil {
		t.Fatal("open failed")
	}
	var stop atomic.Bool
	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			for !stop.Load() {
				_ = runtime.LogosRegionAllocNoscan(h, gen, 8, 8)
			}
		})
	}
	time.Sleep(50 * time.Millisecond)
	runtime.LogosRegionClose(h, gen)
	stop.Store(true)
	wg.Wait()
}
