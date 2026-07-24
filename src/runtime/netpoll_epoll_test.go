// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux

package runtime_test

import (
	"sync"
	"syscall"
	"testing"
	"time"
)

func cpuNanos() int64 {
	var ru syscall.Rusage
	syscall.Getrusage(syscall.RUSAGE_SELF, &ru)
	return ru.Utime.Sec*1e9 + ru.Utime.Usec*1e3 +
		ru.Stime.Sec*1e9 + ru.Stime.Usec*1e3
}

// BenchmarkSpreadSubMsTimers measures the cost of waking many goroutines
// with staggered sub-millisecond sleep intervals. With epoll_wait these
// deadlines collapse into the same 1ms bucket (one wakeup); with
// epoll_pwait2 each gets its own wakeup unless coalescing absorbs them.
// Run with GODEBUG=epollpwait2=0 (default) and GODEBUG=epollpwait2=1
// to compare. cpu-ns/wakeup uses RUSAGE_SELF to measure actual process
// CPU time, not wall time.
func BenchmarkSpreadSubMsTimers(b *testing.B) {
	const nGoroutines = 50
	var wg sync.WaitGroup
	cpu0 := cpuNanos()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		wg.Add(nGoroutines)
		for j := 0; j < nGoroutines; j++ {
			d := time.Duration(20+j*20) * time.Microsecond
			go func(d time.Duration) {
				time.Sleep(d)
				wg.Done()
			}(d)
		}
		wg.Wait()
	}
	b.StopTimer()
	cpuElapsed := cpuNanos() - cpu0
	wakeups := b.N * nGoroutines
	b.ReportMetric(float64(cpuElapsed)/float64(wakeups), "cpu-ns/wakeup")
	b.ReportMetric(float64(wakeups)/b.Elapsed().Seconds(), "wakeups/s")
}
