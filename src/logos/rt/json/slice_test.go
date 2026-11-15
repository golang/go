// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	stdjson "encoding/json"
	"runtime"
	"runtime/debug"
	"strings"
	"testing"
	"unsafe"
)

func sliceOf[E any](elem func(*Scanner, *E)) func(*Scanner, *[]E) {
	return func(s *Scanner, dst *[]E) { Slice(s, dst, elem) }
}

var sliceInputs = []string{
	`[]`, `[ ]`, `[1]`, `[1,2,3]`, `[ 1 , 2 , 3 ]`, `null`, `[null]`, `[1,null,3]`, `["x"]`, `[1,"x",3]`, `[1.5]`, `[300]`,
	`[true]`, `[[1]]`, `[{}]`, `5`, `"s"`, `true`, `{}`, `{"b":[1]}`, `[1,[2,[3]]]`, `[-1,0,127,128]`,
}

func TestSliceParity(t *testing.T) {
	checkSingle(t, sliceInputs, sliceOf(Int[int8]))
	checkSingle(t, sliceInputs, sliceOf(Int[int]))
	checkSingle(t, sliceInputs, sliceOf(Uint[uint16]))
	checkSingle(t, sliceInputs, sliceOf(Float[float64]))
	checkSingle(t, sliceInputs, sliceOf(Bool[bool]))
	checkSingle(t, sliceInputs, sliceOf(String[string]))
}

func TestSliceNested(t *testing.T) {
	inputs := []string{`[[1],[2,3],[]]`, `[[],[[]]]`, `[null,[1],null]`, `[[1,"x"],"y",[2]]`, `[[]]`, `[1]`}
	checkSingle(t, inputs, sliceOf(sliceOf(Int[int])))
}

func TestSliceNullSetsNil(t *testing.T) {
	s := NewScanner([]byte(`{"A":null}`), (*single[[]int])(nil))
	dst := []int{1, 2, 3}
	s.Object()
	for s.Next() {
		Slice(&s, &dst, Int[int])
	}
	if err := s.End(); err != nil {
		t.Fatal(err)
	}
	if dst != nil {
		t.Fatalf("null left %#v, want nil", dst)
	}
}

func TestSliceReusesCapacity(t *testing.T) {
	backing := make([]int, 5, 16)
	for i := range backing {
		backing[i] = 99
	}
	dst := backing
	s := NewScanner([]byte(`{"A":[1,2,3]}`), (*single[[]int])(nil))
	s.Object()
	for s.Next() {
		Slice(&s, &dst, Int[int])
	}
	if err := s.End(); err != nil {
		t.Fatal(err)
	}
	if len(dst) != 3 || dst[0] != 1 || dst[1] != 2 || dst[2] != 3 {
		t.Fatalf("dst = %v", dst)
	}
	if unsafe.SliceData(dst) != unsafe.SliceData(backing) || cap(dst) != 16 {
		t.Fatal("Slice did not reuse the existing backing array")
	}
}

func TestSliceDecodesIntoExistingElements(t *testing.T) {
	type pair struct{ X, Y int }
	dst := []pair{{1, 2}, {3, 4}}
	s := NewScanner([]byte(`{"A":[{"X":9}]}`), (*single[[]pair])(nil))
	s.Object()
	for s.Next() {
		Slice(&s, &dst, func(s *Scanner, p *pair) {
			s.Skip()
			p.X = 9
		})
	}
	if len(dst) != 1 || dst[0] != (pair{X: 9, Y: 2}) {
		t.Fatalf("dst = %v, want [{9 2}]: element 0 keeps the fields the array does not set", dst)
	}
}

func TestSliceNullElementKeepsExistingValue(t *testing.T) {
	dst := []int{7, 8, 9}
	s := NewScanner([]byte(`{"A":[null,5]}`), (*single[[]int])(nil))
	s.Object()
	for s.Next() {
		Slice(&s, &dst, Int[int])
	}
	if err := s.End(); err != nil {
		t.Fatal(err)
	}
	if len(dst) != 2 || dst[0] != 7 || dst[1] != 5 {
		t.Fatalf("dst = %v, want [7 5]", dst)
	}
}

func TestSliceSlotsPastOldLengthStartZero(t *testing.T) {
	backing := []int{1, 42, 43}
	dst := backing[:1]
	s := NewScanner([]byte(`{"A":[null,null,null]}`), (*single[[]int])(nil))
	s.Object()
	for s.Next() {
		Slice(&s, &dst, Int[int])
	}
	if len(dst) != 3 || dst[0] != 1 || dst[1] != 0 || dst[2] != 0 {
		t.Fatalf("dst = %v, want [1 0 0]: no stale data past the old length", dst)
	}
}

func TestSliceEmptyKeepsExistingCapacity(t *testing.T) {
	backing := make([]int, 4, 8)
	dst := backing
	s := NewScanner([]byte(`{"A":[]}`), (*single[[]int])(nil))
	s.Object()
	for s.Next() {
		Slice(&s, &dst, Int[int])
	}
	if len(dst) != 0 || cap(dst) != 8 || unsafe.SliceData(dst) != unsafe.SliceData(backing) {
		t.Fatalf("dst len %d cap %d", len(dst), cap(dst))
	}
}

func TestSliceGrowth(t *testing.T) {
	var b strings.Builder
	b.WriteString(`{"A":[`)
	const n = 1000
	for i := range n {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString("7")
	}
	b.WriteString(`]}`)
	got, err := decodeSingle(b.String(), sliceOf(Int[int]))
	if err != nil {
		t.Fatal(err)
	}
	if len(got.A) != n {
		t.Fatalf("len = %d", len(got.A))
	}
	for _, v := range got.A {
		if v != 7 {
			t.Fatal("wrong element")
		}
	}
}

func TestSliceNoPerElementAllocation(t *testing.T) {
	data := []byte(`{"A":[` + strings.Repeat("12345,", 9999) + `1]}`)
	allocs := testing.AllocsPerRun(20, func() {
		got, err := decodeSingleBytes(data, sliceOf(Int[int]))
		if err != nil || len(got.A) != 10000 {
			t.Fatal(err)
		}
	})
	if allocs > 40 {
		t.Fatalf("decoding 10000 elements allocated %v times, want only the append growth steps", allocs)
	}
}

func allocatedBytes(f func()) uint64 {
	defer debug.SetGCPercent(debug.SetGCPercent(-1))
	runtime.GC()
	runtime.GC()
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	f()
	runtime.ReadMemStats(&after)
	return after.TotalAlloc - before.TotalAlloc
}

func TestSliceAllocationProportionalToElements(t *testing.T) {
	const n = 1_000_000
	data := []byte(`{"A":[` + strings.Repeat("1,", n-1) + `1]}`)
	var got single[[]int32]
	alloc := allocatedBytes(func() {
		var err error
		got, err = decodeSingleBytes(data, sliceOf(Int[int32]))
		if err != nil {
			t.Fatal(err)
		}
	})
	if len(got.A) != n {
		t.Fatalf("len = %d", len(got.A))
	}
	budget := 6 * 4 * uint64(n)
	t.Logf("1e6 int32 elements: %d bytes allocated (%.1f per element)", alloc, float64(alloc)/n)
	if alloc > budget {
		t.Fatalf("allocated %d bytes for %d elements, budget %d", alloc, n, budget)
	}
}

func TestHostileArrayPrefixAllocatesLittle(t *testing.T) {
	for _, in := range []string{`[1,2,3,4,5,6`, `[` + strings.Repeat("[", 10), `[0,0,0,0,0,0,`, `[` + strings.Repeat(" ", 10), `[` + strings.Repeat("9", 10)} {
		data := `{"A":` + in
		alloc := allocatedBytes(func() {
			_, err := decodeSingle(data, sliceOf(Int[int]))
			if err == nil {
				t.Errorf("%q: no error", data)
			}
		})
		if alloc > 1<<20 {
			t.Errorf("%q allocated %d bytes", data, alloc)
		}
	}
}

func TestHostileHugeArrayCostsOnArrival(t *testing.T) {
	const n = 10_000_000
	data := []byte(`{"A":[` + strings.Repeat("0,", n-1) + `0]}`)
	var length int
	alloc := allocatedBytes(func() {
		got, err := decodeSingleBytes(data, sliceOf(Uint[uint8]))
		if err != nil {
			t.Fatal(err)
		}
		length = len(got.A)
	})
	if length != n {
		t.Fatalf("len = %d", length)
	}
	t.Logf("1e7 uint8 elements: %d bytes allocated (%.1f per element)", alloc, float64(alloc)/n)
	if alloc > 8*uint64(n) {
		t.Fatalf("allocated %d bytes for %d one-byte elements", alloc, n)
	}
}

func TestSliceSingleElementPrefixClaimsNothing(t *testing.T) {
	alloc := allocatedBytes(func() {
		_, err := decodeSingle(`{"A":[1`, sliceOf(Int[int]))
		if err == nil {
			t.Error("no error")
		}
	})
	if alloc > 64<<10 {
		t.Fatalf("allocated %d bytes", alloc)
	}
}

func TestSliceSyntaxErrorsMatchStd(t *testing.T) {
	for _, in := range []string{`[`, `[1`, `[1,`, `[1,]`, `[,1]`, `[1 2]`, `[1}`, `[01]`, `[1.]`, `["a]`, `[tru]`, `[1]]`} {
		data := `{"A":` + in + `}`
		_, gotErr := decodeSingle(data, sliceOf(Int[int]))
		var raw stdjson.RawMessage
		wantErr := stdjson.Unmarshal([]byte(data), &raw)
		if wantErr == nil {
			t.Fatalf("%q unexpectedly valid", data)
		}
		if !sameError(gotErr, wantErr) {
			t.Errorf("%q: error = %#v, want %#v", data, gotErr, wantErr)
		}
	}
}

type nested []nested

func readNested(s *Scanner, dst *nested) {
	Slice(s, (*[]nested)(dst), readNested)
}

func nestedDoc(depth int) string {
	return `{"A":` + strings.Repeat("[", depth) + strings.Repeat("]", depth) + `}`
}

func TestDepthLimit(t *testing.T) {
	for _, depth := range []int{1, 100, 9999, 10000, 10001, 20000} {
		data := nestedDoc(depth)
		_, gotErr := decodeSingle(data, readNested)
		var raw stdjson.RawMessage
		wantErr := stdjson.Unmarshal([]byte(data), &raw)
		if !sameError(gotErr, wantErr) {
			t.Errorf("depth %d via Slice: error = %v, want %v", depth, gotErr, wantErr)
		}
		_, gotErr = decodeSingle(data, func(s *Scanner, dst *int) { s.Skip() })
		if !sameError(gotErr, wantErr) {
			t.Errorf("depth %d via Skip: error = %v, want %v", depth, gotErr, wantErr)
		}
		_, gotErr = decodeSingle(data, Int[int])
		if wantErr != nil && !sameError(gotErr, wantErr) {
			t.Errorf("depth %d via Int: error = %v, want %v", depth, gotErr, wantErr)
		}
	}
}
