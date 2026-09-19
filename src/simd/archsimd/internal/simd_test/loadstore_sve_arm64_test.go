// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && arm64

package simd_test

import (
	"simd/archsimd"
	"testing"
)

func mustPanic(t *testing.T, name string, f func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Errorf("%s: expected panic but did not panic", name)
		}
	}()
	f()
}

// testLoadStorePart exercises a scalable type's LoadTPart/StorePart across slice
// lengths smaller than, equal to, and larger than the vector length, plus empty.
// It checks the documented behavior — each reads/writes exactly min(len(s), Len())
// elements — and that neither reads nor writes past the slice (memory safety):
// a short load zeroes the inactive lanes rather than reading past the slice, and
// a short store leaves the trailing elements untouched.
func testLoadStorePart[T number, V any](t *testing.T, name string, load func([]T) (V, int), store func(V, []T) int, vlen int) {
	t.Helper()
	// Distinct nonzero source, longer than a full vector.
	data := make([]T, vlen+8)
	for i := range data {
		data[i] = T(i + 1)
	}

	// Load: reads n = min(len, vlen) elements (returned), zeroes the rest, and
	// never reads past the slice (a lane beyond the slice would be nonzero if it
	// had). A nil slice reads nothing.
	loadCases := []int{0, 1, vlen / 2, vlen - 1, vlen, vlen + 1, vlen + 8}
	for ci, k := range loadCases {
		if k < 0 || k > len(data) {
			continue
		}
		src := data[:k]
		if ci == 0 {
			src = nil // exercise a nil (not just empty) slice
		}
		v, gotN := load(src)
		active := min(k, vlen)
		if gotN != active {
			t.Errorf("%s Load len=%d: returned n=%d, want %d", name, k, gotN, active)
		}
		out := make([]T, vlen)
		store(v, out) // len(out)==vlen: write back all lanes
		for i := 0; i < active; i++ {
			if out[i] != data[i] {
				t.Errorf("%s Load len=%d: lane %d = %v, want %v", name, k, i, out[i], data[i])
			}
		}
		for i := active; i < vlen; i++ {
			if out[i] != 0 {
				t.Errorf("%s Load len=%d: lane %d = %v, want 0 (read past slice?)", name, k, i, out[i])
			}
		}
	}

	// Store: writes exactly n = min(len, vlen) elements (returned), leaving the
	// rest untouched. A nil slice writes nothing.
	full, _ := load(data[:vlen])
	const sentinel = 99
	storeCases := []int{0, 1, vlen / 2, vlen - 1, vlen}
	for ci, k := range storeCases {
		if k < 0 {
			continue
		}
		out := make([]T, vlen)
		for i := range out {
			out[i] = T(sentinel)
		}
		dst := out[:k]
		if ci == 0 {
			dst = nil
		}
		gotN := store(full, dst) // writes min(k, vlen) == k elements
		if gotN != k {
			t.Errorf("%s Store len=%d: returned n=%d, want %d", name, k, gotN, k)
		}
		for i := 0; i < k; i++ {
			if out[i] != data[i] {
				t.Errorf("%s Store len=%d: elem %d = %v, want %v", name, k, i, out[i], data[i])
			}
		}
		for i := k; i < vlen; i++ {
			if out[i] != T(sentinel) {
				t.Errorf("%s Store len=%d: elem %d = %v, want sentinel (wrote past len?)", name, k, i, out[i])
			}
		}
	}
}

func TestLoadStorePartSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no sve")
	}
	testLoadStorePart(t, "Int8s", archsimd.LoadInt8sPart, archsimd.Int8s.StorePart, archsimd.Int8s{}.Len())
	testLoadStorePart(t, "Uint8s", archsimd.LoadUint8sPart, archsimd.Uint8s.StorePart, archsimd.Uint8s{}.Len())
	testLoadStorePart(t, "Int16s", archsimd.LoadInt16sPart, archsimd.Int16s.StorePart, archsimd.Int16s{}.Len())
	testLoadStorePart(t, "Uint16s", archsimd.LoadUint16sPart, archsimd.Uint16s.StorePart, archsimd.Uint16s{}.Len())
	testLoadStorePart(t, "Int32s", archsimd.LoadInt32sPart, archsimd.Int32s.StorePart, archsimd.Int32s{}.Len())
	testLoadStorePart(t, "Uint32s", archsimd.LoadUint32sPart, archsimd.Uint32s.StorePart, archsimd.Uint32s{}.Len())
	testLoadStorePart(t, "Float32s", archsimd.LoadFloat32sPart, archsimd.Float32s.StorePart, archsimd.Float32s{}.Len())
	testLoadStorePart(t, "Int64s", archsimd.LoadInt64sPart, archsimd.Int64s.StorePart, archsimd.Int64s{}.Len())
	testLoadStorePart(t, "Uint64s", archsimd.LoadUint64sPart, archsimd.Uint64s.StorePart, archsimd.Uint64s{}.Len())
	testLoadStorePart(t, "Float64s", archsimd.LoadFloat64sPart, archsimd.Float64s.StorePart, archsimd.Float64s{}.Len())
}

// TestLoadStorePlainSVE checks the whole-vector Load/Store: they round-trip a
// full slice and panic when the slice is shorter than the vector.
func TestLoadStorePlainSVE(t *testing.T) {
	if !archsimd.ARM64.SVE() {
		t.Skip("no sve")
	}
	var z archsimd.Int8s
	n := z.Len()
	data := make([]int8, n)
	for i := range data {
		data[i] = int8(i + 1)
	}
	v := archsimd.LoadInt8s(data)
	out := make([]int8, n)
	v.Store(out)
	for i := range data {
		if out[i] != data[i] {
			t.Errorf("LoadInt8s/Store round-trip: lane %d = %d, want %d", i, out[i], data[i])
		}
	}
	mustPanic(t, "LoadInt8s short", func() { archsimd.LoadInt8s(make([]int8, n-1)) })
	mustPanic(t, "Int8s.Store short", func() { v.Store(make([]int8, n-1)) })
}
