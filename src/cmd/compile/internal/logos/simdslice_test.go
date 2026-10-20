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

// simdSliceTarget builds for arch at GOAMD64 level with the simd
// experiment on, and restores the build when the test ends.
func simdSliceTarget(t *testing.T, arch string, level int) {
	t.Helper()
	restoreBuildAfter(t)
	buildcfg.Experiment.SIMD, buildcfg.GOARCH, buildcfg.GOAMD64 = true, arch, level
}

// scalarSliceTarget builds without the simd experiment.
func scalarSliceTarget(t *testing.T) {
	t.Helper()
	restoreBuildAfter(t)
	buildcfg.Experiment.SIMD, buildcfg.GOARCH, buildcfg.GOAMD64 = false, "amd64", 1
}

// restoreBuildAfter restores the build the lowering reads when t ends.
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

// archsimdLoaderFeatures reads the `CPU Feature:` line of every
// Load<T>SliceMasked loader in logos_maskload_amd64.go, keyed by vector
// type.
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

// TestSimdSliceSumMatrix is the #22 lowering attack: on amd64 every
// exact element kind sums through a full-vector loop and the masked tail
// loader of its width, and the feature the guard checks is the `CPU
// Feature:` line of that loader. The tail never goes through the
// full-width Part loader, which reads past the slice.
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
			guard := "_logos_simd.X86." + feats[c.vec] + "()"
			if feats[c.vec] == "" || !strings.Contains(out, "if "+guard+" {") {
				t.Errorf("guard is not %q (loader requires %q):\n%s", guard, feats[c.vec], out)
			}
		})
	}
}

// TestSimdSliceSumGuardElision pins the guard against the GOAMD64 level:
// the check is absent exactly when the level guarantees the loader's
// feature, and then the scalar loop is not emitted at all.
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

// TestSimdSliceSumFloatKeepsTree pins the float contract: the vector
// form only stages the elements, and the pairwise recursion over halves
// that the scalar form runs is the one that sums, spelled once whether
// or not a feature check guards the vector form, so the bits match.
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

// TestSimdSliceSumScalarForms pins the shapes that keep the scalar
// form: no experiment, a fused operation, and `int`, whose pointer is
// not the *[N]int64 the archsimd loaders take; an alias of a fixed-width
// kind is that kind and keeps the vector form.
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

// TestSimdSliceSumArm64 pins the arm64 form: 128-bit vectors, the
// upstream Part loader for the tail (NEON has no masked load, and the
// loader inserts only the elements present), and no feature check.
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

// TestSimdSliceSumNoPadding is the #22 padding attack: the build adds no
// guard padding for a vector tail. The integer form allocates nothing,
// and the float form stages into a slice of exactly len(receiver)
// elements, so no length rounds up to a whole vector and every masked
// load alone keeps the tail inside the slice.
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

// ejectTarget marks the compile as an eject (-logos.eject set) and
// restores the flag when the test ends.
func ejectTarget(t *testing.T) {
	t.Helper()
	saved := base.LogosEject
	base.LogosEject = t.TempDir()
	t.Cleanup(func() { base.LogosEject = saved })
}

// TestSimdSliceSumEjectScalar pins T-0.3 for the slice sum: a compile
// that writes eject text takes the scalar form for every element kind on
// both architectures, at every GOAMD64 level, so the text names no
// archsimd function (the Load<T>SliceMasked loaders exist only in this
// toolchain's archsimd) and equals the text of a build without the simd
// experiment byte for byte, which is the form whose float bits the
// fixed pairwise order pins.
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

// TestSimdSliceSumEjectNativeKeepsVector pins the other half: only the
// eject compile takes the scalar form. The same source lowered with no
// eject target still takes the vector form.
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

// TestSimdSliceEjectKeepsArrayForms pins the scope of the eject rule:
// array operators and array reductions load through the archsimd array
// loaders an upstream archsimd has, so eject keeps their vector form.
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

// TestSimdSliceLoadersOnlyInSliceSum pins the 15.2 scope sentence: the
// masked loaders appear only in the loop a bare slice sum produces. An
// array operator, an array reduction, and a slice run with a fused
// operation carry none.
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
