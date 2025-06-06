// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"internal/synctest"
	"runtime"
	"runtime/metrics"
	"sync/atomic"
	"unsafe"
)

// This program ensures system goroutines (GC workers, finalizer goroutine)
// started from within a synctest bubble do not participate in that bubble.
//
// To ensure none of these goroutines start before synctest.Run,
// it must have no dependencies on packages which may start system goroutines.
// This includes the os package, which creates finalizers at init time.

func numGCCycles() uint64 {
	samples := []metrics.Sample{{Name: "/gc/cycles/total:gc-cycles"}}
	metrics.Read(samples)
	if samples[0].Value.Kind() == metrics.KindBad {
		panic("metric not supported")
	}
	return samples[0].Value.Uint64()
}

type T struct {
	v int
	p unsafe.Pointer
}

func main() {
	// Channels created by a finalizer and cleanup func registered within the bubble.
	var (
		finalizerCh atomic.Pointer[chan struct{}]
		cleanupCh   atomic.Pointer[chan struct{}]
	)
	synctest.Run(func() {
		// Start the finalizer and cleanup goroutines.
		{
			p := new(T)
			runtime.SetFinalizer(p, func(*T) {
				ch := make(chan struct{})
				finalizerCh.Store(&ch)
			})
			runtime.AddCleanup(p, func(struct{}) {
				ch := make(chan struct{})
				cleanupCh.Store(&ch)
			}, struct{}{})
		}
		startingCycles := numGCCycles()
		ch1 := make(chan *T)
		ch2 := make(chan *T)
		defer close(ch1)
		go func() {
			for range ch1 {
				ch2 <- &T{}
			}
		}()
		const iterations = 1000
		for range iterations {
			// Make a lot of short-lived allocations to get the GC working.
			for range 1000 {
				v := new(T)
				// Set finalizers on these values, just for added stress.
				runtime.SetFinalizer(v, func(*T) {})
				ch1 <- v
				<-ch2
			}

			// If we've improperly put a GC goroutine into the synctest group,
			// this Wait is going to hang.
			synctest.Wait()

			// End the test after a couple of GC cycles have passed.
			if numGCCycles()-startingCycles > 1 && finalizerCh.Load() != nil && cleanupCh.Load() != nil {
				return
			}
		}
		println("finalizers/cleanups failed to run after", iterations, "cycles")
	})
	// Close the channels created by the finalizer and cleanup func.
	// If the funcs improperly ran inside the bubble, these channels are bubbled
	// and trying to close them will panic.
	close(*finalizerCh.Load())
	close(*cleanupCh.Load())
	println("success")
}
