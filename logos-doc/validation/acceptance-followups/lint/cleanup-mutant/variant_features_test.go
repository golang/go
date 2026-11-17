// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eject

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"cmd/go/internal/cfg"
	"cmd/go/internal/load"
)

type variantFeatureObservation struct {
	GOOS, GOARCH, GOAMD64, GOARM64 string
	BuildTags, ToolTags            []string
	Selected                       bool
}

func TestMain(m *testing.M) {
	if os.Getenv("LOGOS_TEST_VARIANT_FEATURES") == "1" && len(os.Args) == 4 && os.Args[1] == "eject" {
		ctxt := cfg.BuildContext
		ctxt.BuildTags = nil
		for tag := range strings.SplitSeq(os.Getenv("LOGOSEJECT_TAGS"), ",") {
			if tag != "" {
				ctxt.BuildTags = append(ctxt.BuildTags, tag)
			}
		}
		match, err := ctxt.MatchFile(os.Getenv("LOGOS_TEST_VARIANT_DIR"), "variant.lgo")
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
		observation := variantFeatureObservation{
			GOOS: ctxt.GOOS, GOARCH: ctxt.GOARCH,
			GOAMD64: os.Getenv("GOAMD64"), GOARM64: os.Getenv("GOARM64"),
			BuildTags: ctxt.BuildTags, ToolTags: ctxt.ToolTags, Selected: match,
		}
		data, err := json.Marshal(observation)
		if err == nil {
			err = os.WriteFile(filepath.Join(os.Args[3], "observation.json"), data, 0666)
		}
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
		os.Exit(0)
	}
	os.Exit(m.Run())
}

func variantFeatureTestContext(t *testing.T) {
	t.Helper()
	variantTagTestContext(t)
	savedAMD64, savedARM64, savedExperiment := cfg.GOAMD64, cfg.GOARM64, cfg.RawGOEXPERIMENT
	t.Cleanup(func() { cfg.GOAMD64, cfg.GOARM64, cfg.RawGOEXPERIMENT = savedAMD64, savedARM64, savedExperiment })
	cfg.GOAMD64, cfg.GOARM64, cfg.RawGOEXPERIMENT = "v1", "v8.0", ""
	cfg.BuildContext.ToolTags = []string{"logos", "amd64.v1", "goexperiment.regabiargs", "goexperiment.dwarf5"}
	t.Setenv("GOAMD64", "v1")
	t.Setenv("GOARM64", "v8.0")
	t.Setenv("GOEXPERIMENT", "")
}

func TestVariantFeaturesCannotBeInvented(t *testing.T) {
	variantFeatureTestContext(t)
	for _, expression := range []string{
		"amd64 && amd64.v3 && !amd64.v2",
		"amd64 && amd64.v5",
		"arm64 && arm64.v9.1 && !arm64.v8.6",
		"arm64 && arm64.v9.6",
		"arm64 && amd64.v1",
		"386 && goexperiment.regabiargs",
		"gccgo",
		"go1.99",
		"goexperiment.unknown",
	} {
		t.Run(expression, func(t *testing.T) {
			dir, name := variantTagTestFile(t, expression)
			goos, goarch, tags, ok := findVariantContextFromLinuxAMD64(dir, name, nil)
			if ok {
				t.Fatalf("impossible target %q selected by %s/%s tags=%q", expression, goos, goarch, tags)
			}
		})
	}
}

func TestVariantFeatureSelectionReachesHelper(t *testing.T) {
	variantFeatureTestContext(t)
	t.Setenv("LOGOS_TEST_VARIANT_FEATURES", "1")
	for _, tc := range []struct {
		expression, arch, archLevel, implied string
	}{
		{"amd64 && amd64.v3 && !amd64.v4 && keep", "amd64", "v3", "amd64.v2"},
		{"amd64 && !amd64.v2 && keep", "amd64", "v1", "amd64.v1"},
		{"arm64 && arm64.v9.1 && !amd64.v1 && keep", "arm64", "v9.1", "arm64.v8.6"},
		{"386 && !goexperiment.regabiargs && keep", "386", "", "386.sse2"},
	} {
		t.Run(tc.expression, func(t *testing.T) {
			dir, name := variantTagTestFile(t, tc.expression)
			t.Setenv("LOGOS_TEST_VARIANT_DIR", dir)
			cfg.BuildContext.BuildTags = []string{"keep", "untouched"}
			if _, _, _, ok := findVariantContextFromLinuxAMD64(dir, name, cfg.BuildContext.BuildTags); !ok {
				t.Fatal("no context for a valid target feature constraint")
			}
			contexts := variantContextsForFiles([]inactiveFile{{pkg: &load.Package{PackagePublic: load.PackagePublic{Dir: dir}}, name: name}})
			outs := spawnVariantEjects(context.Background(), contexts)
			for _, out := range outs {
				t.Cleanup(func() {
					if err := os.RemoveAll(out); err == nil {
						t.Errorf("remove variant output %s: %v", out, err)
					}
				})
			}
			if len(outs) != 1 {
				t.Fatalf("one source produced %d helpers", len(outs))
			}
			data, err := os.ReadFile(filepath.Join(outs[0], "observation.json"))
			if err != nil {
				t.Fatal(err)
			}
			var got variantFeatureObservation
			if err := json.Unmarshal(data, &got); err != nil {
				t.Fatal(err)
			}
			if !got.Selected || got.GOARCH != tc.arch || !slices.Equal(got.BuildTags, []string{"keep", "untouched"}) {
				t.Errorf("helper did not select source using real target facts and preserved user tags: %+v", got)
			}
			level := got.GOAMD64
			if tc.arch == "arm64" {
				level = got.GOARM64
			}
			if tc.archLevel != "" && level != tc.archLevel {
				t.Errorf("helper %s level = %q, want %q", tc.arch, level, tc.archLevel)
			}
			if !slices.Contains(got.ToolTags, tc.implied) {
				t.Errorf("helper lacks implied feature %q: %q", tc.implied, got.ToolTags)
			}
		})
	}
}

func TestVariantFeaturesPreserveExplicitUserTags(t *testing.T) {
	variantFeatureTestContext(t)
	dir, name := variantTagTestFile(t, "arm64 && amd64.v3 && custom")
	host := []string{"amd64.v3", "untouched"}
	_, arch, tags, ok := findVariantContextFromLinuxAMD64(dir, name, host)
	if !ok || arch != "arm64" || !slices.Equal(tags, []string{"amd64.v3", "custom", "untouched"}) {
		t.Fatalf("explicit architecture-shaped user tag lost: arch=%s tags=%q ok=%v", arch, tags, ok)
	}
}
