// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
)

func fin(v *int) {
}

func BenchmarkFinalizer(b *testing.B) {
	const CallsPerSched = 1000
	procs := runtime.GOMAXPROCS(-1)
	N := int32(b.N / CallsPerSched)
	var wg sync.WaitGroup
	wg.Add(procs)
	for p := 0; p < procs; p++ {
		go func() {
			var data [CallsPerSched]*int
			for i := 0; i < CallsPerSched; i++ {
				data[i] = new(int)
			}
			for atomic.AddInt32(&N, -1) >= 0 {
				runtime.Gosched()
				for i := 0; i < CallsPerSched; i++ {
					runtime.SetFinalizer(data[i], fin)
				}
				for i := 0; i < CallsPerSched; i++ {
					runtime.SetFinalizer(data[i], nil)
				}
			}
			wg.Done()
		}()
	}
	wg.Wait()
}

func BenchmarkFinalizerRun(b *testing.B) {
	const CallsPerSched = 1000
	procs := runtime.GOMAXPROCS(-1)
	N := int32(b.N / CallsPerSched)
	var wg sync.WaitGroup
	wg.Add(procs)
	for p := 0; p < procs; p++ {
		go func() {
			for atomic.AddInt32(&N, -1) >= 0 {
				runtime.Gosched()
				for i := 0; i < CallsPerSched; i++ {
					v := new(int)
					runtime.SetFinalizer(v, fin)
				}
				runtime.GC()
			}
			wg.Done()
		}()
	}
	wg.Wait()
}
