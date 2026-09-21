// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import "testing"

// From issue #77720.
type moveLoadBenchHandle[T any] struct {
	value *T
}

func (h moveLoadBenchHandle[T]) Value() T {
	return *h.value
}

type moveLoadBenchBig struct {
	typ        int8
	index      int64
	str        string
	pkgID      string
	dummyField [1024]byte
}

type moveLoadBenchS struct {
	h moveLoadBenchHandle[moveLoadBenchBig]
}

var moveLoadBenchSink int8

func moveLoadBenchTypViaValue(s moveLoadBenchS) int8 {
	return s.h.Value().typ
}

func moveLoadBenchTypViaPtr(s moveLoadBenchS) int8 {
	return (*s.h.value).typ
}

func benchmarkMoveLoad(b *testing.B, f func(moveLoadBenchS) int8) {
	backing := make([]moveLoadBenchBig, 1<<10)
	ss := make([]moveLoadBenchS, len(backing))
	for i := range backing {
		backing[i].typ = int8(i)
		ss[i] = moveLoadBenchS{h: moveLoadBenchHandle[moveLoadBenchBig]{&backing[i]}}
	}

	b.ResetTimer()
	var x int8
	for i := 0; i < b.N; i++ {
		x += f(ss[i&(len(ss)-1)])
	}
	moveLoadBenchSink = x
}

func BenchmarkMoveLoadTypViaValue(b *testing.B) {
	benchmarkMoveLoad(b, moveLoadBenchTypViaValue)
}

func BenchmarkMoveLoadTypViaPtr(b *testing.B) {
	benchmarkMoveLoad(b, moveLoadBenchTypViaPtr)
}
