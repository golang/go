// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"testing"

	"cmd/compile/internal/syntax"
	"internal/buildcfg"
	"internal/testenv"
)

// archsimdFeatureLines reads the `CPU Feature:` line of every
// element-wise method in src/simd/archsimd/ops_amd64.go, keyed by
// "<vector type>.<method>". The file is the ground truth for what each
// emitted call needs from the CPU.
func archsimdFeatureLines(t *testing.T) map[string]string {
	t.Helper()
	path := filepath.Join(testenv.GOROOT(t), "src", "simd", "archsimd", "ops_amd64.go")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("archsimd source not available: %v", err)
	}
	rx := regexp.MustCompile(`// Asm: \w+, CPU Feature: (\w+)\nfunc \(x (\w+)\) (Add|Sub|Mul|Div|And|Or|Xor)\(`)
	feats := make(map[string]string)
	for _, m := range rx.FindAllSubmatch(data, -1) {
		feats[string(m[2])+"."+string(m[3])] = string(m[1])
	}
	if len(feats) == 0 {
		t.Fatalf("no CPU Feature lines found in %s", path)
	}
	return feats
}

var simdOpsByName = map[string]syntax.Operator{
	"Add": syntax.Add, "Sub": syntax.Sub, "Mul": syntax.Mul, "Div": syntax.Div,
	"And": syntax.And, "Or": syntax.Or, "Xor": syntax.Xor,
}

var simdAttackLanes = []simdLane{
	{"Int8", 1, false}, {"Uint8", 1, false},
	{"Int16", 2, false}, {"Uint16", 2, false},
	{"Int32", 4, false}, {"Uint32", 4, false},
	{"Int64", 8, false}, {"Uint64", 8, false},
	{"Float32", 4, true}, {"Float64", 8, true},
}

// TestSimdAmd64FeatureMatchesArchsimd is the M14-S2 attack: for every
// (vector type, operator) the amd64 vector form can emit, the feature
// the guard checks must be exactly the `CPU Feature:` of that archsimd
// method. A narrower guard faults on a CPU lacking the feature; a wider
// one needlessly takes the scalar form.
func TestSimdAmd64FeatureMatchesArchsimd(t *testing.T) {
	feats := archsimdFeatureLines(t)
	checked := 0
	for _, lane := range simdAttackLanes {
		for _, total := range []int64{16, 32, 64} {
			vec := lane.name + "x" + strconv.FormatInt(total/lane.width, 10)
			for name, op := range simdOpsByName {
				want, ok := feats[vec+"."+name]
				if !ok {
					continue
				}
				got := amd64Feature(total, lane, op)
				if got.check != want {
					t.Errorf("%s.%s: guard checks %s, archsimd requires %s", vec, name, got.check, want)
				}
				checked++
			}
		}
	}
	if checked < 100 {
		t.Fatalf("matrix checked only %d (vector, operator) pairs", checked)
	}
}

// TestSimdFragmentBlocksNormalized is the M14-S2 scope attack: the guard
// puts nested blocks in a simd fragment, and the DWARF scope marker pops
// each block at its closing brace and fatals when that position sorts
// before the block start. A fragment parsed from "logos-simd" sorts
// before any user file, so every closing brace must carry the
// replacement position, not the fragment coordinates.
func TestSimdFragmentBlocksNormalized(t *testing.T) {
	pos := syntax.MakePos(syntax.NewFileBase("main.lgo"), 7, 3)
	e := parseSimdExprFrag("func() int { if true { return 1 } else { return 2 } }()", pos)
	blocks := 0
	syntax.Inspect(e, func(n syntax.Node) bool {
		if b, ok := n.(*syntax.BlockStmt); ok {
			blocks++
			if b.Rbrace != pos {
				t.Errorf("block %d: closing brace at %v, want %v", blocks, b.Rbrace, pos)
			}
		}
		return true
	})
	if blocks != 3 {
		t.Fatalf("fragment has %d blocks, want function body, then, and else", blocks)
	}
}

// TestSimdGuardElision is the M14-S2 level attack: the guard is elided
// exactly when the GOAMD64 level guarantees the feature, the baseline
// feature (every arm64 form) is never guarded at any level, and the
// check spells the archsimd X86 method.
func TestSimdGuardElision(t *testing.T) {
	cases := []struct {
		level   int
		feat    simdFeature
		wantCnd string
	}{
		{1, simdFeatAVX, "_logos_simd.X86.AVX()"},
		{2, simdFeatAVX, "_logos_simd.X86.AVX()"},
		{3, simdFeatAVX, ""},
		{4, simdFeatAVX, ""},
		{1, simdFeatAVX2, "_logos_simd.X86.AVX2()"},
		{2, simdFeatAVX2, "_logos_simd.X86.AVX2()"},
		{3, simdFeatAVX2, ""},
		{4, simdFeatAVX2, ""},
		{1, simdFeatAVX512, "_logos_simd.X86.AVX512()"},
		{2, simdFeatAVX512, "_logos_simd.X86.AVX512()"},
		{3, simdFeatAVX512, "_logos_simd.X86.AVX512()"},
		{4, simdFeatAVX512, ""},
		{1, simdFeatBaseline, ""},
		{4, simdFeatBaseline, ""},
	}
	saved := buildcfg.GOAMD64
	defer func() { buildcfg.GOAMD64 = saved }()
	fl := new(fileLower)
	for _, c := range cases {
		buildcfg.GOAMD64 = c.level
		if got := fl.simdGuardCond(c.feat); got != c.wantCnd {
			t.Errorf("v%d %q: guard %q, want %q", c.level, c.feat.check, got, c.wantCnd)
		}
	}
}
