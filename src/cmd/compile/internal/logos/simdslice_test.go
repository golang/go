// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"cmd/compile/internal/base"
	"internal/buildcfg"
	"internal/testenv"
)

func simdSliceTarget(t *testing.T, arch string, level int) {
	t.Helper()
	restoreBuildAfter(t)
	buildcfg.Experiment.SIMD, buildcfg.GOARCH, buildcfg.GOAMD64 = true, arch, level
}

func scalarSliceTarget(t *testing.T) {
	t.Helper()
	restoreBuildAfter(t)
	buildcfg.Experiment.SIMD, buildcfg.GOARCH, buildcfg.GOAMD64 = false, "amd64", 1
}

func restoreBuildAfter(t *testing.T) {
	t.Helper()
	savedSIMD, savedArch, savedLevel := buildcfg.Experiment.SIMD, buildcfg.GOARCH, buildcfg.GOAMD64
	t.Cleanup(func() {
		buildcfg.Experiment.SIMD, buildcfg.GOARCH, buildcfg.GOAMD64 = savedSIMD, savedArch, savedLevel
	})
}

func lowerSliceFunc(t *testing.T, params, result, body string) string {
	t.Helper()
	src := "package main\n\ntype f32 = float32\n\nfunc f(" + params + ") " + result + " {\n\t" + body + "\n}\n"
	return lowerSnippetSrc(t, "slicesum.lgo", src)
}

func lowerSliceSum(t *testing.T, elem string) string {
	t.Helper()
	return lowerSliceFunc(t, "xs []"+elem, elem, "return xs.sum()")
}

func archsimdLoaderFeatures(t *testing.T) map[string]string {
	t.Helper()
	path := filepath.Join(testenv.GOROOT(t), "src", "simd", "archsimd", "logos_maskload_amd64.go")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("archsimd source not available: %v", err)
	}
	rx := regexp.MustCompile(`(?s)// Asm: \w+, CPU Feature: (\w+)\n//\n//go:noescape\nfunc Load(\w+)SliceMasked\(`)
	feats := make(map[string]string)
	for _, m := range rx.FindAllSubmatch(data, -1) {
		feats[string(m[2])] = string(m[1])
	}
	if len(feats) != 10 {
		t.Fatalf("found %d masked loaders in %s, want 10", len(feats), path)
	}
	return feats
}

func TestSimdSliceSumMatrix(t *testing.T) {
	feats := archsimdLoaderFeatures(t)
	cases := []struct {
		elem, vec string
		lanes     int
	}{
		{"int8", "Int8x64", 64}, {"uint8", "Uint8x64", 64},
		{"int16", "Int16x32", 32}, {"uint16", "Uint16x32", 32},
		{"int32", "Int32x8", 8}, {"uint32", "Uint32x8", 8},
		{"int64", "Int64x4", 4}, {"uint64", "Uint64x4", 4},
		{"float32", "Float32x8", 8}, {"float64", "Float64x4", 4},
	}
	simdSliceTarget(t, "amd64", 1)
	for _, c := range cases {
		t.Run(c.elem, func(t *testing.T) {
			out := lowerSliceSum(t, c.elem)
			if want := "_logos_simd.Load" + c.vec + "SliceMasked("; !strings.Contains(out, want) {
				t.Fatalf("no masked tail loader %s:\n%s", want, out)
			}
			if want := "_logos_simd.Load" + c.vec + "Array((*["; !strings.Contains(out, want) {
				t.Errorf("full vectors do not load through the array intrinsic %s:\n%s", want, out)
			}
			for _, call := range []string{"Load" + c.vec + "(", "Load" + c.vec + "Part(", ".Store(", ".StorePart("} {
				if strings.Contains(out, call) {
					t.Errorf("wrapper call %s in the vector form; a lowered package inlines no import:\n%s", call, out)
				}
			}
			if want := "_i+" + strconv.Itoa(c.lanes) + " <= len("; !strings.Contains(out, want) {
				t.Errorf("full-vector loop step %d missing:\n%s", c.lanes, out)
			}
			if c.elem != "float32" && c.elem != "float64" && !strings.Contains(out, "_acc.Add(") {
				t.Errorf("integer sum does not accumulate in vector lanes:\n%s", out)
			}
			guard := "_logos_simd.X86." + feats[c.vec] + "()"
			if feats[c.vec] == "" || !strings.Contains(out, "if "+guard+" {") {
				t.Errorf("guard is not %q (loader requires %q):\n%s", guard, feats[c.vec], out)
			}
		})
	}
}

func TestSimdSliceSumGuardElision(t *testing.T) {
	cases := []struct {
		elem   string
		level  int
		guard  bool
		scalar bool
	}{
		{"int32", 1, true, true}, {"int32", 2, true, true},
		{"int32", 3, false, false}, {"int32", 4, false, false},
		{"uint8", 1, true, true}, {"uint8", 3, true, true},
		{"uint8", 4, false, false},
		{"float32", 1, true, true}, {"float32", 3, false, false},
	}
	for _, c := range cases {
		t.Run(c.elem+"/v"+strconv.Itoa(c.level), func(t *testing.T) {
			simdSliceTarget(t, "amd64", c.level)
			out := lowerSliceSum(t, c.elem)
			if got := strings.Contains(out, "_logos_simd.X86."); got != c.guard {
				t.Errorf("guard present=%v, want %v:\n%s", got, c.guard, out)
			}
			if got := strings.Contains(out, "_e := range "); got != c.scalar {
				t.Errorf("scalar loop present=%v, want %v:\n%s", got, c.scalar, out)
			}
		})
	}
}

func TestSimdSliceSumFloatKeepsTree(t *testing.T) {
	for _, level := range []int{1, 3} {
		t.Run("v"+strconv.Itoa(level), func(t *testing.T) {
			simdSliceTarget(t, "amd64", level)
			out := lowerSliceSum(t, "float32")
			if n := strings.Count(out, "_sum("); n != 3 {
				t.Errorf("got %d helper calls, want 3 (top-level plus both halves):\n%s", n, out)
			}
			if !strings.Contains(out, "/ 2") {
				t.Errorf("halving split missing:\n%s", out)
			}
			if strings.Contains(out, "_acc") {
				t.Errorf("float sum accumulates in vector lanes, changing the order:\n%s", out)
			}
			if !strings.Contains(out, "_out = make(") {
				t.Errorf("vector form does not stage into the summed slice:\n%s", out)
			}
			if strings.Contains(out, " * ") || strings.Contains(out, "FMA") {
				t.Errorf("multiply or FMA in the float sum lowering (#67 contraction surface):\n%s", out)
			}
		})
	}
}

func TestSimdSliceSumScalarForms(t *testing.T) {
	t.Run("no experiment", func(t *testing.T) {
		scalarSliceTarget(t)
		if out := lowerSliceSum(t, "int32"); strings.Contains(out, "archsimd") || strings.Contains(out, "_logos_simd") {
			t.Errorf("experiment off still lowers through archsimd:\n%s", out)
		}
	})
	t.Run("int", func(t *testing.T) {
		simdSliceTarget(t, "amd64", 1)
		if out := lowerSliceSum(t, "int"); strings.Contains(out, "_logos_simd") {
			t.Errorf("int takes the vector form:\n%s", out)
		}
	})
	t.Run("uint", func(t *testing.T) {
		simdSliceTarget(t, "amd64", 1)
		if out := lowerSliceSum(t, "uint"); strings.Contains(out, "_logos_simd") {
			t.Errorf("uint takes the vector form:\n%s", out)
		}
	})
	t.Run("uintptr", func(t *testing.T) {
		simdSliceTarget(t, "amd64", 1)
		if out := lowerSliceSum(t, "uintptr"); strings.Contains(out, "_logos_simd") {
			t.Errorf("uintptr takes the vector form:\n%s", out)
		}
	})
	t.Run("alias element", func(t *testing.T) {
		simdSliceTarget(t, "amd64", 1)
		if out := lowerSliceSum(t, "f32"); !strings.Contains(out, "_logos_simd.LoadFloat32x8SliceMasked(") {
			t.Errorf("an alias of float32 misses the vector form:\n%s", out)
		}
	})
	t.Run("fused operation", func(t *testing.T) {
		simdSliceTarget(t, "amd64", 1)
		out := lowerSliceFunc(t, "xs []int32", "int32", "return xs.filter(x => x > 0).sum()")
		if strings.Contains(out, "_logos_simd") {
			t.Errorf("a run with a filter takes the vector form:\n%s", out)
		}
	})
	t.Run("count", func(t *testing.T) {
		simdSliceTarget(t, "amd64", 1)
		out := lowerSliceFunc(t, "xs []int32", "int", "return xs.count()")
		if strings.Contains(out, "_logos_simd") {
			t.Errorf("count takes the vector form:\n%s", out)
		}
	})
}

func TestSimdSliceSumArm64(t *testing.T) {
	simdSliceTarget(t, "arm64", 1)
	for _, c := range []struct{ elem, vec string }{
		{"int8", "Int8x16"}, {"uint16", "Uint16x8"}, {"int32", "Int32x4"},
		{"uint64", "Uint64x2"}, {"float32", "Float32x4"}, {"float64", "Float64x2"},
	} {
		out := lowerSliceSum(t, c.elem)
		if want := "_tv, _ := _logos_simd.Load" + c.vec + "Part("; !strings.Contains(out, want) {
			t.Errorf("%s: no Part tail loader %s:\n%s", c.elem, want, out)
		}
		if strings.Contains(out, "X86") || strings.Contains(out, "SliceMasked") {
			t.Errorf("%s: amd64 spelling on arm64:\n%s", c.elem, out)
		}
	}
}

func TestSimdSliceSumNoPadding(t *testing.T) {
	simdSliceTarget(t, "amd64", 1)
	exact := regexp.MustCompile(`make\(\[\]float(32|64), len\(_logos_ch\d+_\d+_rcv\)\)`)
	for _, elem := range []string{"int8", "uint16", "int32", "uint64", "float32", "float64"} {
		out := lowerSliceSum(t, elem)
		makes := strings.Count(out, "make(")
		want := 0
		if strings.HasPrefix(elem, "float") {
			want = 1
			if !exact.MatchString(out) {
				t.Errorf("%s: staging slice is not exactly len(receiver):\n%s", elem, out)
			}
		}
		if makes != want {
			t.Errorf("%s: %d allocations in the vector form, want %d:\n%s", elem, makes, want, out)
		}
	}
}

func TestSimdSliceSumNoStagingElsewhere(t *testing.T) {
	simdSliceTarget(t, "amd64", 1)
	for _, c := range []struct {
		name   string
		params string
		result string
		body   string
		makes  int
		kept   string
	}{
		{"fused", "xs []int32", "int32", "return xs.filter(x => x > 0).sum()", 0, ""},
		{"count", "xs []int32", "int", "return xs.count()", 0, ""},
		{"loop", "xs []int32", "int32", "var t int32\n\tfor _, v := range xs {\n\t\tt += v\n\t}\n\treturn t", 0, ""},
		{"explicit", "n int", "int", "out := make([]int32, n)\n\treturn len(out)", 1, "make([]int32, n)"},
	} {
		t.Run(c.name, func(t *testing.T) {
			src := "package main\n\ntype f32 = float32\n\nfunc f(" + c.params + ") " + c.result + " {\n\t" + c.body + "\n}\n\nfunc trig(xs []int32) int {\n\treturn xs.count()\n}\n"
			out := lowerSnippetSrc(t, "slicesum.lgo", src)
			for _, bad := range []string{"_logos_simd", "SliceMasked", "_mt", "archsimd"} {
				if strings.Contains(out, bad) {
					t.Errorf("%s: vector staging %q outside a bare vector sum:\n%s", c.name, bad, out)
				}
			}
			if n := strings.Count(out, "make("); n != c.makes {
				t.Errorf("%s: %d makes, want %d (only the source):\n%s", c.name, n, c.makes, out)
			}
			if !strings.Contains(out, c.kept) {
				t.Errorf("%s: source make %q lost its exact length:\n%s", c.name, c.kept, out)
			}
		})
	}
}

func ejectTarget(t *testing.T) {
	t.Helper()
	saved := base.LogosEject
	base.LogosEject = t.TempDir()
	t.Cleanup(func() { base.LogosEject = saved })
}

func TestSimdSliceSumEjectScalar(t *testing.T) {
	elems := []string{
		"int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64",
		"float32", "float64", "f32",
	}
	for _, arch := range []string{"amd64", "arm64"} {
		for _, level := range []int{1, 3, 4} {
			for _, elem := range elems {
				t.Run(arch+"/v"+strconv.Itoa(level)+"/"+elem, func(t *testing.T) {
					scalarSliceTarget(t)
					want := lowerSliceSum(t, elem)
					simdSliceTarget(t, arch, level)
					ejectTarget(t)
					got := lowerSliceSum(t, elem)
					for _, name := range []string{"archsimd", "_logos_simd", "SliceMasked"} {
						if strings.Contains(got, name) {
							t.Errorf("eject text names %s:\n%s", name, got)
						}
					}
					if got != want {
						t.Errorf("eject text under the simd experiment differs from the scalar text:\n%s\nwant:\n%s", got, want)
					}
				})
			}
		}
	}
}

func TestSimdSliceSumEjectNativeKeepsVector(t *testing.T) {
	simdSliceTarget(t, "amd64", 1)
	if out := lowerSliceSum(t, "int32"); !strings.Contains(out, "_logos_simd.LoadInt32x8SliceMasked(") {
		t.Errorf("native build lost the vector form:\n%s", out)
	}
	ejectTarget(t)
	if out := lowerSliceSum(t, "int32"); strings.Contains(out, "SliceMasked") {
		t.Errorf("eject build keeps the fork-only loader:\n%s", out)
	}
}

func TestSimdSliceEjectKeepsArrayForms(t *testing.T) {
	simdSliceTarget(t, "amd64", 1)
	ejectTarget(t)
	src := `package main

func f(a, b [8]float32) (float32, [8]float32) {
	c := a + b
	return c.sum(), a * b
}
`
	out := lowerSnippetSrc(t, "arrays.lgo", src)
	if !strings.Contains(out, "_logos_simd.LoadFloat32x8Array(") {
		t.Errorf("eject lost the array vector form:\n%s", out)
	}
	if strings.Contains(out, "SliceMasked") {
		t.Errorf("fork-only loader in eject text:\n%s", out)
	}
}

func TestSimdSliceLoadersOnlyInSliceSum(t *testing.T) {
	simdSliceTarget(t, "amd64", 1)
	src := `package main

func f(a, b [8]float32, xs []int32) (float32, int32, [8]float32) {
	c := a + b
	return c.sum(), xs.map(x => x + 1).sum(), a * b
}
`
	out := lowerSnippetSrc(t, "scope.lgo", src)
	if strings.Contains(out, "SliceMasked") {
		t.Errorf("masked loader outside a bare slice sum:\n%s", out)
	}
	if !strings.Contains(out, "_logos_simd.LoadFloat32x8Array(") {
		t.Errorf("array operator lost its vector form:\n%s", out)
	}
}
