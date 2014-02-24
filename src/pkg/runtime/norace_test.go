// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The file contains tests that can not run under race detector for some reason.
// +build !race

package runtime_test

import (
	"runtime"
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
	b.SetParallelism(excess)
	b.RunParallel(func(pb *testing.PB) {
		foo := 42
		for pb.Next() {
			runtime.Entersyscall()
			for i := 0; i < work; i++ {
				foo *= 2
				foo /= 2
			}
			runtime.Exitsyscall()
		}
		_ = foo
	})
}
