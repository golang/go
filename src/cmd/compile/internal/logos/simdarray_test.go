// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"strconv"
	"strings"
	"testing"
)

func lowerArrayOp(t *testing.T, pre, arr, op string) string {
	t.Helper()
	src := "package main\n\n" + pre + "func f(a, b " + arr + ") " + arr + " {\n\treturn a " + op + " b\n}\n"
	return lowerSnippetSrc(t, "arraysimd.lgo", src)
}

func TestSimdArrayVectorScalarForms(t *testing.T) {
	n := strconv.FormatInt(4, 10)
	t.Run("fixed width takes vector form", func(t *testing.T) {
		simdSliceTarget(t, "amd64", 1)
		out := lowerArrayOp(t, "", "["+n+"]float32", "+")
		if !strings.Contains(out, "archsimd") {
			t.Errorf("float32 array misses the vector form:\n%s", out)
		}
	})
	t.Run("alias takes vector form", func(t *testing.T) {
		simdSliceTarget(t, "amd64", 1)
		out := lowerArrayOp(t, "type f32 = float32\n\n", "["+n+"]f32", "+")
		if !strings.Contains(out, "archsimd") {
			t.Errorf("alias array misses the vector form:\n%s", out)
		}
	})
	t.Run("byte takes vector form amd64", func(t *testing.T) {
		simdSliceTarget(t, "amd64", 1)
		out := lowerArrayOp(t, "", "[16]byte", "+")
		if !strings.Contains(out, "archsimd") {
			t.Errorf("byte array misses the vector form:\n%s", out)
		}
	})
	t.Run("byte takes vector form arm64", func(t *testing.T) {
		simdSliceTarget(t, "arm64", 1)
		out := lowerArrayOp(t, "", "[16]byte", "+")
		if !strings.Contains(out, "archsimd") {
			t.Errorf("byte array misses the vector form on arm64:\n%s", out)
		}
		if strings.Contains(out, "X86") {
			t.Errorf("arm64 byte array carries an X86 guard:\n%s", out)
		}
	})
	t.Run("rune takes vector form amd64", func(t *testing.T) {
		simdSliceTarget(t, "amd64", 1)
		out := lowerArrayOp(t, "", "[8]rune", "+")
		if !strings.Contains(out, "archsimd") {
			t.Errorf("rune array misses the vector form:\n%s", out)
		}
	})
	t.Run("rune takes vector form arm64", func(t *testing.T) {
		simdSliceTarget(t, "arm64", 1)
		out := lowerArrayOp(t, "", "[8]rune", "+")
		if !strings.Contains(out, "archsimd") {
			t.Errorf("rune array misses the vector form on arm64:\n%s", out)
		}
		if strings.Contains(out, "X86") {
			t.Errorf("arm64 rune array carries an X86 guard:\n%s", out)
		}
	})
	t.Run("small int kinds take vector form amd64", func(t *testing.T) {
		simdSliceTarget(t, "amd64", 1)
		for _, arr := range []string{"[16]int8", "[8]int16", "[8]uint16", "[4]int64"} {
			out := lowerArrayOp(t, "", arr, "+")
			if !strings.Contains(out, "archsimd") {
				t.Errorf("%s array misses the vector form:\n%s", arr, out)
			}
		}
	})
	t.Run("small int kinds take vector form arm64", func(t *testing.T) {
		simdSliceTarget(t, "arm64", 1)
		for _, arr := range []string{"[16]int8", "[8]int16", "[8]uint16", "[4]int64"} {
			out := lowerArrayOp(t, "", arr, "+")
			if !strings.Contains(out, "archsimd") {
				t.Errorf("%s array misses the vector form:\n%s", arr, out)
			}
			if strings.Contains(out, "X86") {
				t.Errorf("arm64 %s array carries an X86 guard:\n%s", arr, out)
			}
		}
	})
	t.Run("int stays scalar", func(t *testing.T) {
		simdSliceTarget(t, "amd64", 1)
		out := lowerArrayOp(t, "", "["+n+"]int", "+")
		if strings.Contains(out, "archsimd") || strings.Contains(out, "_logos_simd") {
			t.Errorf("int array takes the vector form:\n%s", out)
		}
	})
	t.Run("uint stays scalar", func(t *testing.T) {
		simdSliceTarget(t, "amd64", 1)
		out := lowerArrayOp(t, "", "[4]uint", "+")
		if strings.Contains(out, "archsimd") || strings.Contains(out, "_logos_simd") {
			t.Errorf("uint array takes the vector form:\n%s", out)
		}
	})
	t.Run("uintptr stays scalar", func(t *testing.T) {
		simdSliceTarget(t, "amd64", 1)
		out := lowerArrayOp(t, "", "[2]uintptr", "+")
		if strings.Contains(out, "archsimd") || strings.Contains(out, "_logos_simd") {
			t.Errorf("uintptr array takes the vector form:\n%s", out)
		}
	})
	t.Run("defined type stays scalar", func(t *testing.T) {
		simdSliceTarget(t, "amd64", 1)
		out := lowerArrayOp(t, "type I int32\n\n", "["+n+"]I", "+")
		if strings.Contains(out, "archsimd") || strings.Contains(out, "_logos_simd") {
			t.Errorf("defined element type takes the vector form:\n%s", out)
		}
	})
	t.Run("no experiment scalar", func(t *testing.T) {
		scalarSliceTarget(t)
		out := lowerArrayOp(t, "", "["+n+"]float32", "+")
		if strings.Contains(out, "archsimd") || strings.Contains(out, "_logos_simd") {
			t.Errorf("experiment off takes the vector form:\n%s", out)
		}
		for _, lane := range []string{
			"_logos_a[0] + _logos_b[0]",
			"_logos_a[1] + _logos_b[1]",
			"_logos_a[2] + _logos_b[2]",
			"_logos_a[3] + _logos_b[3]",
		} {
			if !strings.Contains(out, lane) {
				t.Errorf("experiment off misses unrolled lane %q:\n%s", lane, out)
			}
		}
	})
}

func TestSimdArrayOpGuardElision(t *testing.T) {
	cases := []struct {
		arr    string
		op     string
		level  int
		guard  string
		scalar bool
	}{
		{"[4]float32", "+", 1, "_logos_simd.X86.AVX()", true},
		{"[4]float32", "+", 3, "", false},
		{"[4]float32", "+", 4, "", false},
		{"[8]uint32", "*", 1, "_logos_simd.X86.AVX2()", true},
		{"[8]uint32", "*", 3, "", false},
		{"[8]uint32", "*", 4, "", false},
		{"[4]float32", "&", 1, "_logos_simd.X86.AVX()", true},
		{"[8]float32", "&", 1, "_logos_simd.X86.AVX2()", true},
		{"[8]float32", "&", 3, "", false},
		{"[4]float64", "^", 1, "_logos_simd.X86.AVX2()", true},
		{"[2]float64", "|", 1, "_logos_simd.X86.AVX()", true},
	}
	for _, c := range cases {
		t.Run(c.arr+"/"+c.op+"/v"+strconv.Itoa(c.level), func(t *testing.T) {
			simdSliceTarget(t, "amd64", c.level)
			out := lowerArrayOp(t, "", c.arr, c.op)
			if c.guard != "" && !strings.Contains(out, "if "+c.guard+" {") {
				t.Errorf("guard is not %q:\n%s", c.guard, out)
			}
			if got := strings.Contains(out, "_logos_simd.X86."); got != (c.guard != "") {
				t.Errorf("guard present=%v, want %v:\n%s", got, c.guard != "", out)
			}
			if got := strings.Contains(out, "_logos_a[0]"); got != c.scalar {
				t.Errorf("scalar fallback present=%v, want %v:\n%s", got, c.scalar, out)
			}
		})
	}
	for _, c := range [][2]string{{"[4]float32", "+"}, {"[8]uint32", "*"}} {
		t.Run(c[0]+"/"+c[1]+"/arm64", func(t *testing.T) {
			simdSliceTarget(t, "arm64", 1)
			out := lowerArrayOp(t, "", c[0], c[1])
			if !strings.Contains(out, "archsimd") {
				t.Errorf("arm64 array misses the vector form:\n%s", out)
			}
			if strings.Contains(out, "_logos_simd.X86.") {
				t.Errorf("arm64 array carries an X86 guard:\n%s", out)
			}
		})
	}
}
