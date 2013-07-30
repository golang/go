// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The file contains tests that can not run under race detector for some reason.
// +build !race

package runtime_test

import (
	"runtime"
	"sync/atomic"
	"testing"
)

// Syscall tests split stack between Entersyscall and Exitsyscall under race detector.
func BenchmarkSyscall(b *testing.B) {
	benchmarkSyscall(b, 0, 1)
}

func BenchmarkSyscallWork(b *testing.B) {
	benchmarkSyscall(b, 100, 1)
}

func BenchmarkSyscallExcess(b *testing.B) {
	benchmarkSyscall(b, 0, 4)
}

func BenchmarkSyscallExcessWork(b *testing.B) {
	benchmarkSyscall(b, 100, 4)
}

func benchmarkSyscall(b *testing.B, work, excess int) {
	const CallsPerSched = 1000
	procs := runtime.GOMAXPROCS(-1) * excess
	N := int32(b.N / CallsPerSched)
	c := make(chan bool, procs)
	for p := 0; p < procs; p++ {
		go func() {
			foo := 42
			for atomic.AddInt32(&N, -1) >= 0 {
				runtime.Gosched()
				for g := 0; g < CallsPerSched; g++ {
					runtime.Entersyscall()
					for i := 0; i < work; i++ {
						foo *= 2
						foo /= 2
					}
					runtime.Exitsyscall()
				}
			}
			c <- foo == 42
		}()
	}
	for p := 0; p < procs; p++ {
		<-c
	}
}
