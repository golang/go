// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package atomic_test

import (
	"runtime/internal/atomic"
	"testing"
)

var sink interface{}

func BenchmarkAtomicLoad64(b *testing.B) {
	var x uint64
	sink = &x
	for i := 0; i < b.N; i++ {
		_ = atomic.Load64(&x)
	}
}

func BenchmarkAtomicStore64(b *testing.B) {
	var x uint64
	sink = &x
	for i := 0; i < b.N; i++ {
		atomic.Store64(&x, 0)
	}
}

func BenchmarkAtomicLoad(b *testing.B) {
	var x uint32
	sink = &x
	for i := 0; i < b.N; i++ {
		_ = atomic.Load(&x)
	}
}

func BenchmarkAtomicStore(b *testing.B) {
	var x uint32
	sink = &x
	for i := 0; i < b.N; i++ {
		atomic.Store(&x, 0)
	}
}

func BenchmarkAnd8(b *testing.B) {
	var x [512]uint8 // give byte its own cache line
	sink = &x
	for i := 0; i < b.N; i++ {
		atomic.And8(&x[255], uint8(i))
	}
}

func BenchmarkAnd8Parallel(b *testing.B) {
	var x [512]uint8 // give byte its own cache line
	sink = &x
	b.RunParallel(func(pb *testing.PB) {
		i := uint8(0)
		for pb.Next() {
			atomic.And8(&x[255], i)
			i++
		}
	})
}

func BenchmarkOr8(b *testing.B) {
	var x [512]uint8 // give byte its own cache line
	sink = &x
	for i := 0; i < b.N; i++ {
		atomic.Or8(&x[255], uint8(i))
	}
}

func BenchmarkOr8Parallel(b *testing.B) {
	var x [512]uint8 // give byte its own cache line
	sink = &x
	b.RunParallel(func(pb *testing.PB) {
		i := uint8(0)
		for pb.Next() {
			atomic.Or8(&x[255], i)
			i++
		}
	})
}

func BenchmarkXadd(b *testing.B) {
	var x uint32
	ptr := &x
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			atomic.Xadd(ptr, 1)
		}
	})
}

func BenchmarkXadd64(b *testing.B) {
	var x uint64
	ptr := &x
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			atomic.Xadd64(ptr, 1)
		}
	})
}
