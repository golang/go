// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sgutil

import (
	"fmt"
	"go/parser"
	"go/token"
	"slices"
	"testing"
)

func TestParseBuildConstraints(t *testing.T) {
	tests := []struct {
		filename, build string
		phase           Phase
		stableArchOnly  bool
	}{
		// Phases.
		{"x.go", "", 0, false},
		{"x.go", "//go:build amd64", 0, true},
		{"x.go", "//go:build goexperiment.simdstable", PhaseStable, false},
		{"x.go", "//go:build goexperiment.simd", PhaseExperiment, false},
		{"x.go", "//go:build !goexperiment.simd", 0, false},
		{"x.go", "//go:build !goexperiment.simdstable", 0, false},
		{"x.go", "//go:build goexperiment.simd || goexperiment.simdstable", PhaseStable, false},
		{"x.go", "//go:build goexperiment.simdstable && !goexperiment.simd", PhaseStable, false},
		{"x.go", "//go:build goexperiment.simdstable && (goexperiment.simd || amd64)", PhaseStable, true},

		// Stable archsimd GOARCHes.
		{"ops_amd64.go", "//go:build goexperiment.simdstable", PhaseStable, true},
		{"ops_amd64.go", "", 0, true},
		{"slicepart_wasm_test.go", "//go:build goexperiment.simdstable", PhaseStable, true},
		{"ops_sve.go", "//go:build goexperiment.simd && arm64", PhaseExperiment, true},
		{"maskrange_128.go", "//go:build goexperiment.simdstable && (amd64 || arm64 || wasm)", PhaseStable, true},
		{"x.go", "//go:build goexperiment.simdstable && (amd64 || mips)", PhaseStable, false},
		{"x.go", "//go:build goexperiment.simdstable && amd64 && !mips", PhaseStable, true},
		{"x.go", "//go:build goexperiment.simdstable && (amd64 || arm64) && linux", PhaseStable, true},
		{"x.go", "//go:build goexperiment.simdstable && !amd64", PhaseStable, false},
		{"ops_riscv64.go", "//go:build goexperiment.simdstable", PhaseStable, false},
		{"ops_riscv64.go", "//go:build goexperiment.simd && riscv64", PhaseExperiment, false},
		{"amd64.go", "//go:build goexperiment.simdstable", PhaseStable, false}, // no _ before the GOARCH
		{"x_linux_amd64.go", "//go:build goexperiment.simdstable", PhaseStable, true},
		{"x_arm64_test.go", "//go:build goexperiment.simdstable", PhaseStable, true},
		{"x_linux.go", "//go:build goexperiment.simdstable", PhaseStable, false},
		{"cpu.go", "//go:build goexperiment.simdstable", PhaseStable, false},
		{"tofrom_emulated.go", "//go:build goexperiment.simdstable && !(amd64 || wasm || arm64)", PhaseStable, false},
		// Only stable archsimd GOARCHes in the experimental build, but not in the
		// stable one, which determines the phase.
		{"x.go", "//go:build goexperiment.simdstable && (amd64 || !goexperiment.simd)", PhaseStable, false},
	}
	for _, tt := range tests {
		fset := token.NewFileSet()
		f, err := parser.ParseFile(fset, tt.filename, tt.build+"\n\npackage p\n", parser.ParseComments)
		if err != nil {
			t.Fatal(err)
		}
		c, err := ParseBuildConstraints(fset, f)
		if err != nil {
			t.Errorf("ParseBuildConstraints(%q, %q): %v", tt.filename, tt.build, err)
			continue
		}
		if got := c.Phase(); got != tt.phase {
			t.Errorf("ParseBuildConstraints(%q, %q).Phase() = %v, want %v", tt.filename, tt.build, int(got), int(tt.phase))
		}
		if got := c.StableArchOnly(); got != tt.stableArchOnly {
			t.Errorf("ParseBuildConstraints(%q, %q).StableArchOnly() = %v, want %v", tt.filename, tt.build, got, tt.stableArchOnly)
		}
	}

	// Errors.
	tooMany := "//go:build goexperiment.simdstable && (t0"
	for i := 1; i <= maxBuildTags; i++ {
		tooMany += fmt.Sprintf(" || t%d", i)
	}
	tooMany += ")"
	for _, build := range []string{
		tooMany,
		"//go:build goexperiment.simd && !goexperiment.simdstable", // never builds
		"//go:build amd64 && !amd64",
		"//go:build goexperiment.simdstable && amd64 && !(arm64 || amd64)",
	} {
		fset := token.NewFileSet()
		f, err := parser.ParseFile(fset, "x.go", build+"\n\npackage p\n", parser.ParseComments)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := ParseBuildConstraints(fset, f); err == nil {
			t.Errorf("ParseBuildConstraints(%q): want error", build)
		}
	}
}

// TestParseBuildConstraintsFixed checks that tags that must be set or unset
// for a file to build aren't enumerated.
func TestParseBuildConstraintsFixed(t *testing.T) {
	tests := []struct {
		filename, build string
		free            []string
	}{
		{"x.go", "//go:build goexperiment.simdstable && amd64", nil},
		{"ops_amd64.go", "//go:build goexperiment.simdstable", nil},
		{"x.go", "//go:build goexperiment.simd && arm64 && !mips", nil},
		{"x.go", "//go:build goexperiment.simdstable && (amd64 || mips)", []string{"amd64", "mips"}},
		{"x.go", "//go:build goexperiment.simdstable && linux && (amd64 || arm64)", []string{"amd64", "arm64"}},
		{"x.go", "//go:build goexperiment.simdstable && !mips && (amd64 || mips)", []string{"amd64"}},
		{"x.go", "//go:build goexperiment.simdstable && !(mips || s390x) && (amd64 || mips || s390x)", []string{"amd64"}},
	}
	for _, tt := range tests {
		fset := token.NewFileSet()
		f, err := parser.ParseFile(fset, tt.filename, tt.build+"\n\npackage p\n", parser.ParseComments)
		if err != nil {
			t.Fatal(err)
		}
		c, err := ParseBuildConstraints(fset, f)
		if err != nil {
			t.Errorf("ParseBuildConstraints(%q, %q): %v", tt.filename, tt.build, err)
			continue
		}
		slices.Sort(c.free)
		if !slices.Equal(c.free, tt.free) {
			t.Errorf("ParseBuildConstraints(%q, %q) tried free tags %q, want %q", tt.filename, tt.build, c.free, tt.free)
		}
	}
}

func TestNameGOARCH(t *testing.T) {
	for name, want := range map[string]string{
		"x.go":             "",
		"x_amd64.go":       "amd64",
		"x_linux_arm64.go": "arm64",
		"x_wasm.go":        "wasm",
		"x_wasm_test.go":   "wasm",
		"x_linux.go":       "",
		"x_riscv64.go":     "",
		"amd64.go":         "",
	} {
		if got := nameGOARCH(name); got != want {
			t.Errorf("nameGOARCH(%q) = %q, want %q", name, got, want)
		}
	}
}
