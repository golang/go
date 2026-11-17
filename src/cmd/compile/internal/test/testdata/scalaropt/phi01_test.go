// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package scalaropt

import "testing"

type phi01Integer interface {
	~int8 | ~uint8 | ~int16 | ~uint16 | ~int32 | ~uint32 | ~int64 | ~uint64
}

//go:noinline
func phi01Sum[T phi01Integer](flag bool) T {
	var x, y T
	if flag {
		x, y = 0, 7
	} else {
		x, y = 1, 6
	}
	return x + y
}

//go:noinline
func phi01Independent[T phi01Integer](left, right bool) T {
	var x, y T
	if left {
		x = 0
	} else {
		x = 1
	}
	if right {
		y = 7
	} else {
		y = 6
	}
	return x + y
}

//go:noinline
func phi01SignedWrap(flag bool) int8 {
	var x, y int8
	if flag {
		x, y = 0, -128
	} else {
		x, y = 1, 127
	}
	return x + y
}

//go:noinline
func phi01UnsignedWrap(flag bool) uint8 {
	var x, y uint8
	if flag {
		x, y = 0, 0
	} else {
		x, y = 1, 255
	}
	return x + y
}

//go:noinline
func phi01FromPointer(flag *bool) uint64 {
	var x, y uint64
	if *flag {
		x, y = 0, 7
	} else {
		x, y = 1, 6
	}
	return x + y
}

//go:noinline
func phi01Observed(flag bool, calls *int) uint64 {
	var x, y uint64
	if flag {
		*calls += 1
		x, y = 0, 7
	} else {
		*calls += 2
		x, y = 1, 6
	}
	return x + y
}

func checkPhi01Width[T phi01Integer](t *testing.T) {
	t.Helper()
	for _, flag := range []bool{false, true} {
		if got := phi01Sum[T](flag); got != 7 {
			t.Fatalf("correlated flag %t = %v, want 7", flag, got)
		}
	}
	for _, tc := range []struct {
		left, right bool
		want        T
	}{
		{false, false, 7},
		{false, true, 8},
		{true, false, 6},
		{true, true, 7},
	} {
		if got := phi01Independent[T](tc.left, tc.right); got != tc.want {
			t.Fatalf("independent flags %t/%t = %v, want %v", tc.left, tc.right, got, tc.want)
		}
	}
}

func TestScalarPhi01Widths(t *testing.T) {
	t.Run("int8", checkPhi01Width[int8])
	t.Run("uint8", checkPhi01Width[uint8])
	t.Run("int16", checkPhi01Width[int16])
	t.Run("uint16", checkPhi01Width[uint16])
	t.Run("int32", checkPhi01Width[int32])
	t.Run("uint32", checkPhi01Width[uint32])
	t.Run("int64", checkPhi01Width[int64])
	t.Run("uint64", checkPhi01Width[uint64])
	type tag uint16
	t.Run("named_uint16", checkPhi01Width[tag])
}

func TestScalarPhi01OverflowAndEffects(t *testing.T) {
	for _, flag := range []bool{false, true} {
		if got := phi01SignedWrap(flag); got != -128 {
			t.Fatalf("signed wrap %t = %d, want -128", flag, got)
		}
		if got := phi01UnsignedWrap(flag); got != 0 {
			t.Fatalf("unsigned wrap %t = %d, want 0", flag, got)
		}
		if got := phi01FromPointer(&flag); got != 7 {
			t.Fatalf("pointer condition %t = %d, want 7", flag, got)
		}
		calls, wantCalls := 0, 2
		if flag {
			wantCalls = 1
		}
		if got := phi01Observed(flag, &calls); got != 7 || calls != wantCalls {
			t.Fatalf("observed %t = %d with %d effects, want 7 with %d", flag, got, calls, wantCalls)
		}
	}
}

func TestScalarPhi01Panics(t *testing.T) {
	for _, tc := range []struct {
		name string
		call func() uint64
	}{
		{"nil_condition", func() uint64 { return phi01FromPointer(nil) }},
		{"nil_effect_true", func() uint64 { return phi01Observed(true, nil) }},
		{"nil_effect_false", func() uint64 { return phi01Observed(false, nil) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("required nil panic was eliminated")
				}
			}()
			sink = tc.call()
		})
	}
}

type phi01Descriptor struct {
	offset, payload uint64
	format          uint8
}

// Both descriptor formats reserve seven bytes before the payload: one has a
// prefix byte and six padding bytes, while the other uses seven padding bytes.
//
//go:noinline
func phi01Descriptors(records []phi01Descriptor) uint64 {
	var checksum uint64
	for _, r := range records {
		var prefix, padding uint64
		if r.format&1 != 0 {
			prefix, padding = 0, 7
		} else {
			prefix, padding = 1, 6
		}
		checksum += (r.offset + (prefix + padding)) ^ r.payload
	}
	return checksum
}

func TestScalarPhi01Descriptors(t *testing.T) {
	values := []uint64{0, 1, 1 << 63, ^uint64(0) - 7, ^uint64(0) - 6, ^uint64(0)}
	var extremes []phi01Descriptor
	for _, offset := range values {
		for _, payload := range values {
			for _, format := range []uint8{0, 1, 2, 255} {
				extremes = append(extremes, phi01Descriptor{offset, payload, format})
			}
		}
	}
	for _, tc := range []struct {
		name    string
		records []phi01Descriptor
	}{
		{"nil", nil},
		{"empty", []phi01Descriptor{}},
		{"prefix", []phi01Descriptor{{17, 0x12345678, 0}}},
		{"padding", []phi01Descriptor{{17, 0x12345678, 1}}},
		{"extremes", extremes},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var want uint64
			for _, r := range tc.records {
				want += (r.offset + 7) ^ r.payload
			}
			if got := phi01Descriptors(tc.records); got != want {
				t.Fatalf("descriptor checksum = %#x, want %#x", got, want)
			}
		})
	}
	original := phi01Descriptor{17, 0x12345678, 0}
	want := (original.offset + 7) ^ original.payload
	for _, changed := range []phi01Descriptor{
		{original.offset + 1, original.payload, original.format},
		{original.offset, original.payload ^ 1, original.format},
	} {
		got := phi01Descriptors([]phi01Descriptor{changed})
		if expected := (changed.offset + 7) ^ changed.payload; got != expected || got == want {
			t.Fatalf("descriptor fields stopped affecting the checksum: %+v -> %#x", changed, got)
		}
	}
	original.format = 1
	if got := phi01Descriptors([]phi01Descriptor{original}); got != want {
		t.Fatalf("equivalent format changed checksum: %#x, want %#x", got, want)
	}
}

func BenchmarkPhi01Descriptors(b *testing.B) {
	var records [1024]phi01Descriptor
	for i := range records {
		x := uint64(i+1) * 0x9e3779b97f4a7c15
		records[i] = phi01Descriptor{x, ^x, uint8(x >> 32)}
	}
	b.ReportAllocs()
	b.ResetTimer()
	var checksum uint64
	for i := 0; i < b.N; i++ {
		checksum += phi01Descriptors(records[:])
	}
	sink = checksum
}
