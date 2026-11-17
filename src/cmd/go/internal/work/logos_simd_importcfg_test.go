// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package work

import (
	"bytes"
	"internal/buildcfg"
	"slices"
	"strings"
	"testing"

	"cmd/go/internal/cfg"
)

func TestSIMDSynthImports(t *testing.T) {
	oldArch, oldExperiment, oldDryRun := cfg.Goarch, cfg.Experiment, cfg.BuildN
	t.Cleanup(func() {
		cfg.Goarch, cfg.Experiment, cfg.BuildN = oldArch, oldExperiment, oldDryRun
	})
	for _, tc := range []struct {
		name    string
		arch    string
		simd    bool
		nilExp  bool
		dryRun  bool
		direct  bool
		unbuilt bool
		want    bool
	}{
		{name: "amd64", arch: "amd64", simd: true, want: true},
		{name: "arm64", arch: "arm64", simd: true, want: true},
		{name: "wasm", arch: "wasm", simd: true, want: true},
		{name: "unsupported_arch", arch: "riscv64", simd: true},
		{name: "experiment_off", arch: "amd64"},
		{name: "missing_experiment", arch: "amd64", nilExp: true},
		{name: "dry_run", arch: "amd64", simd: true, dryRun: true},
		{name: "already_direct", arch: "amd64", simd: true, direct: true},
		{name: "unbuilt_archives", arch: "amd64", simd: true, unbuilt: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg.Goarch, cfg.BuildN = tc.arch, tc.dryRun
			cfg.Experiment = &buildcfg.ExperimentFlags{}
			cfg.Experiment.SIMD = tc.simd
			if tc.nilExp {
				cfg.Experiment = nil
			}
			bridge := ownerTestAction("simd/internal/bridge", "bridge.go", "bridge.a")
			simd := ownerTestAction("simd", "simd.go", "simd.a", bridge)
			other := ownerTestAction("fmt", "print.go", "fmt.a")
			for _, action := range []*Action{bridge, simd, other} {
				action.Package.Standard = true
			}
			if tc.unbuilt {
				bridge.built, simd.built = "", "DO NOT USE cached marker"
			}
			producer := ownerTestAction("example/producer", "producer.go", "producer.a", simd, bridge, other)
			root := ownerTestAction("example/consumer", "consumer.go", "consumer.a", producer)
			if tc.direct {
				root.Deps = append(root.Deps, simd, bridge)
			}
			var want []string
			if tc.want {
				want = []string{"simd", "simd/internal/bridge"}
			}
			if got := ownerTestPaths(logosSynthImports(root)); !slices.Equal(got, want) {
				t.Logf("synthetic archive paths = %v, want %v", got, want)
				t.Fatal("SIMD synthetic archive set differs from the required set")
			}
		})
	}
}

func TestSIMDSynthImportActionID(t *testing.T) {
	oldArch, oldExperiment, oldDryRun := cfg.Goarch, cfg.Experiment, cfg.BuildN
	t.Cleanup(func() {
		cfg.Goarch, cfg.Experiment, cfg.BuildN = oldArch, oldExperiment, oldDryRun
	})
	cfg.Goarch, cfg.BuildN = "amd64", false
	cfg.Experiment = &buildcfg.ExperimentFlags{}
	cfg.Experiment.SIMD = true
	bridge := ownerTestAction("simd/internal/bridge", "bridge.go", "bridge.a")
	simd := ownerTestAction("simd", "simd.go", "simd.a", bridge)
	bridge.Package.Standard, simd.Package.Standard = true, true
	bridge.buildID, simd.buildID = "action/bridge-before", "action/simd-content"
	producer := ownerTestAction("example/producer", "producer.go", "producer.a", simd)
	root := ownerTestAction("example/consumer", "consumer.go", "consumer.a", producer)
	var imports, before, after bytes.Buffer
	logosSynthImportCfg(&imports, root)
	if got, want := imports.String(), "packagefile simd=simd.a\npackagefile simd/internal/bridge=bridge.a\n"; got != want {
		t.Fatalf("importcfg = %q, want %q", got, want)
	}
	logosSynthActionID(&before, root)
	bridge.buildID = "action/bridge-after"
	logosSynthActionID(&after, root)
	if before.String() == after.String() || !strings.Contains(after.String(), "simd/internal/bridge bridge-after\n") {
		t.Fatalf("implicit bridge content omitted from action identity: before=%q after=%q", before.String(), after.String())
	}
}
