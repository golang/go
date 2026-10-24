// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"fmt"
	"io"
	"runtime"
	"runtime/pprof"
	"sync"
	"testing"
)

var memProfileTestSink [2][]byte

//go:noinline
func allocateMemProfileTest(i, size int) {
	memProfileTestSink[i] = make([]byte, size)
}

func prepareMemProfile(t *testing.T) {
	oldRate := runtime.MemProfileRate
	runtime.MemProfileRate = 1
	t.Cleanup(func() {
		memProfileTestSink = [2][]byte{}
		runtime.MemProfileRate = oldRate
	})

	// Allocate enough to ensure that mcache.nextSample is updated to 1.
	for range 1024 {
		memProfileTestSink[0] = make([]byte, 1024)
	}
	allocateMemProfileTest(0, 1<<20)
	allocateMemProfileTest(1, 2<<20)
	runtime.GC()
}

func readMemProfile(inuseZero bool) ([]runtime.MemProfileRecord, error) {
	var p []runtime.MemProfileRecord
	for range 10 {
		n, ok := runtime.MemProfile(p, inuseZero)
		if ok {
			return p[:n], nil
		}
		p = make([]runtime.MemProfileRecord, n*2+1)
	}
	return nil, fmt.Errorf("memory profile size did not stabilize")
}

func checkMemProfile(p []runtime.MemProfileRecord) error {
	if len(p) == 0 {
		return fmt.Errorf("memory profile is empty")
	}
	for i, r := range p {
		if r.AllocObjects < r.FreeObjects {
			return fmt.Errorf("record %d has %d allocations and %d frees", i, r.AllocObjects, r.FreeObjects)
		}
		if len(r.Stack()) == 0 {
			return fmt.Errorf("record %d has an empty stack", i)
		}
	}
	return nil
}

func TestMemProfilePreemptible(t *testing.T) {
	prepareMemProfile(t)

	for _, inuseZero := range []bool{true, false} {
		called, preemptible := runtime.MemProfileInternalPreemptible(inuseZero)
		if !called {
			t.Errorf("inuseZero=%v: callback was not called", inuseZero)
		} else if !preemptible {
			t.Errorf("inuseZero=%v: callback ran while holding a runtime lock", inuseZero)
		}
	}
}

func TestMemProfileRecords(t *testing.T) {
	prepareMemProfile(t)

	for _, inuseZero := range []bool{true, false} {
		p, err := readMemProfile(inuseZero)
		if err != nil {
			t.Fatalf("inuseZero=%v: %v", inuseZero, err)
		}
		if err := checkMemProfile(p); err != nil {
			t.Errorf("inuseZero=%v: %v", inuseZero, err)
		}
	}
}

func TestMemProfileShortBuffer(t *testing.T) {
	prepareMemProfile(t)

	sentinel := runtime.MemProfileRecord{
		AllocBytes:   -1,
		FreeBytes:    -2,
		AllocObjects: -3,
		FreeObjects:  -4,
	}
	for i := range sentinel.Stack0 {
		sentinel.Stack0[i] = ^uintptr(i)
	}

	for _, inuseZero := range []bool{true, false} {
		p := []runtime.MemProfileRecord{sentinel}
		n, ok := runtime.MemProfile(p, inuseZero)
		if ok {
			t.Errorf("inuseZero=%v: MemProfile with one record succeeded; want buffer-too-small result (n=%d)", inuseZero, n)
		}
		if p[0] != sentinel {
			t.Errorf("inuseZero=%v: MemProfile modified a short buffer", inuseZero)
		}
	}
}

func TestMemProfileConcurrent(t *testing.T) {
	prepareMemProfile(t)

	errCh := make(chan error, 2)
	var wg sync.WaitGroup
	wg.Go(func() {
		for range 10 {
			allocateMemProfileTest(0, 1<<20)
			runtime.GC()
		}
	})
	wg.Go(func() {
		for i := range 10 {
			p, err := readMemProfile(i%2 == 0)
			if err == nil {
				err = checkMemProfile(p)
			}
			if err != nil {
				errCh <- err
				return
			}
		}
	})
	wg.Go(func() {
		for range 10 {
			if err := pprof.Lookup("heap").WriteTo(io.Discard, 0); err != nil {
				errCh <- err
				return
			}
		}
	})
	wg.Wait()
	close(errCh)
	for err := range errCh {
		t.Error(err)
	}
}

func BenchmarkBlocksampled(b *testing.B) {
	for b.Loop() {
		runtime.Blocksampled(42, 1337)
	}
}
