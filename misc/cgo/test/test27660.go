// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Stress the interaction between the race detector and cgo in an
// attempt to reproduce the memory corruption described in #27660.
// The bug was very timing sensitive; at the time of writing this
// test would only trigger the bug about once out of every five runs.

package cgotest

// #include <unistd.h>
import "C"

import (
	"context"
	"math/rand"
	"runtime"
	"sync"
	"testing"
	"time"
)

func test27660(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	ints := make([]int, 100)
	locks := make([]sync.Mutex, 100)
	// Slowly create threads so that ThreadSanitizer is forced to
	// frequently resize its SyncClocks.
	for i := 0; i < 100; i++ {
		go func() {
			for ctx.Err() == nil {
				// Sleep in C for long enough that it is likely that the runtime
				// will retake this goroutine's currently wired P.
				C.usleep(1000 /* 1ms */)
				runtime.Gosched() // avoid starvation (see #28701)
			}
		}()
		go func() {
			// Trigger lots of synchronization and memory reads/writes to
			// increase the likelihood that the race described in #27660
			// results in corruption of ThreadSanitizer's internal state
			// and thus an assertion failure or segfault.
			for ctx.Err() == nil {
				j := rand.Intn(100)
				locks[j].Lock()
				ints[j]++
				locks[j].Unlock()
			}
		}()
		time.Sleep(time.Millisecond)
	}
}
