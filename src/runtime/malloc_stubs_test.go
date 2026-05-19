// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"internal/abi"
	"runtime"
	"testing"
)

const size_ = 0

func benchmarkStubTiny(b *testing.B) {
	const size = size_
	type s struct {
		v [size]byte
	}
	b.Run("kind=new", func(b *testing.B) {
		for b.Loop() {
			runtime.Escape(new(s))
		}
	})
	typ := abi.TypeOf(s{})
	b.Run("kind=mallocgc", func(b *testing.B) {
		for b.Loop() {
			runtime.Escape(runtime.MallocGC(size, typ, false))
		}
	})
}

const noscan_ = false

func benchmarkStub(b *testing.B) {
	const size = size_
	b.Run("kind=new", func(b *testing.B) {
		for b.Loop() {
			if noscan_ {
				runtime.Escape(new(struct{ v [size / 8]uint64 }))
			}
			if !noscan_ {
				runtime.Escape(new(struct{ v [size / 8]*uint64 }))
			}
		}
	})
	var typ *abi.Type
	if noscan_ {
		typ = abi.TypeOf(struct{ v [size / 8]uint64 }{})
	}
	if !noscan_ {
		typ = abi.TypeOf(struct{ v [size / 8]*uint64 }{})
	}
	b.Run("kind=mallocgc", func(b *testing.B) {
		for b.Loop() {
			runtime.Escape(runtime.MallocGC(size, typ, true))
		}
	})
}

func benchmarkScanSliceStub(b *testing.B) {
	const size = size_
	for b.Loop() {
		runtime.Escape(make([]*uint64, size/8))
	}
}
