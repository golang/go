// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rt

import (
	"strings"
	"testing"
)

func mulLenResult[T integer](t *testing.T, a, b T) (v T, panicked any) {
	t.Helper()
	defer func() { panicked = recover() }()
	return MulLen(a, b, "w * h"), nil
}

func TestMulLenBoundaries(t *testing.T) {
	t.Run("int64", func(t *testing.T) {
		boundary := []struct {
			name  string
			a, b  int64
			want  int64
			panic bool
		}{
			{"maxTimesOne", 1<<63 - 1, 1, 1<<63 - 1, false},
			{"maxTimesTwo", 1<<63 - 1, 2, 0, true},
			{"minTimesMinusOne", -1 << 63, -1, 0, true},
			{"minTimesOne", -1 << 63, 1, -1 << 63, false},
			{"minTimesZero", -1 << 63, 0, 0, false},
			{"negTwoTimesThree", -2, 3, -6, false},
			{"zeroTimesMax", 0, 1<<63 - 1, 0, false},
			{"wrapToZero", 1 << 32, 1 << 32, 0, true},
		}
		for _, tc := range boundary {
			t.Run(tc.name, func(t *testing.T) {
				v, r := mulLenResult(t, tc.a, tc.b)
				if tc.panic && r == nil {
					t.Fatalf("MulLen(%d, %d) = %d, want panic", tc.a, tc.b, v)
				}
				if !tc.panic && (r != nil || v != tc.want) {
					t.Fatalf("MulLen(%d, %d) = %d, %v; want %d, nil", tc.a, tc.b, v, r, tc.want)
				}
			})
		}
	})
	t.Run("int8", func(t *testing.T) {
		boundary := []struct {
			name  string
			a, b  int8
			want  int8
			panic bool
		}{
			{"fits", 10, 12, 120, false},
			{"negFits", -10, 12, -120, false},
			{"over", 100, 100, 0, true},
			{"minTimesMinusOne", -128, -1, 0, true},
			{"minTimesOne", -128, 1, -128, false},
		}
		for _, tc := range boundary {
			t.Run(tc.name, func(t *testing.T) {
				v, r := mulLenResult(t, tc.a, tc.b)
				if tc.panic && r == nil {
					t.Fatalf("MulLen(%d, %d) = %d, want panic", tc.a, tc.b, v)
				}
				if !tc.panic && (r != nil || v != tc.want) {
					t.Fatalf("MulLen(%d, %d) = %d, %v; want %d, nil", tc.a, tc.b, v, r, tc.want)
				}
			})
		}
	})
	t.Run("uint", func(t *testing.T) {
		boundary := []struct {
			name  string
			a, b  uint64
			want  uint64
			panic bool
		}{
			{"maxTimesOne", 1<<64 - 1, 1, 1<<64 - 1, false},
			{"maxTimesTwo", 1<<64 - 1, 2, 0, true},
			{"zeroTimesMax", 0, 1<<64 - 1, 0, false},
			{"fits", 1 << 32, (1 << 32) - 1, (1 << 64) - (1 << 32), false},
		}
		for _, tc := range boundary {
			t.Run(tc.name, func(t *testing.T) {
				v, r := mulLenResult(t, tc.a, tc.b)
				if tc.panic && r == nil {
					t.Fatalf("MulLen(%d, %d) = %d, want panic", tc.a, tc.b, v)
				}
				if !tc.panic && (r != nil || v != tc.want) {
					t.Fatalf("MulLen(%d, %d) = %d, %v; want %d, nil", tc.a, tc.b, v, r, tc.want)
				}
			})
		}
	})
	t.Run("uint8", func(t *testing.T) {
		if v, r := mulLenResult(t, uint8(15), uint8(16)); r != nil || v != 240 {
			t.Fatalf("MulLen(15, 16) = %d, %v; want 240, nil", v, r)
		}
		if _, r := mulLenResult(t, uint8(16), uint8(16)); r == nil {
			t.Fatal("MulLen(16, 16) = no panic, want panic")
		}
	})
}

func TestMulLenPanicText(t *testing.T) {
	_, r := mulLenResult(t, 1<<32, 1<<32)
	msg, ok := r.(string)
	if !ok || !strings.Contains(msg, "logos: make length overflow: w * h") {
		t.Fatalf("panic = %v, want text containing %q", r, "logos: make length overflow: w * h")
	}
}
