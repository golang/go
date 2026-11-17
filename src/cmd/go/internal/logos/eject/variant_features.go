// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eject

import (
	"go/build"
	"internal/buildcfg"
	"slices"
	"strings"

	"cmd/go/internal/cfg"
)

func architectureFeatureTag(tag string) bool {
	arch, _, ok := strings.Cut(tag, ".")
	return ok && slices.Contains(ownedGoarchTags, arch)
}

// variantArchEnv reads the settings already resolved by cmd/go, including
// logos.toml and explicit environment overrides. No process global changes.
func variantArchEnv(arch string) (name, value string) {
	switch arch {
	case "386":
		return "GO386", cfg.GO386
	case "amd64":
		return "GOAMD64", cfg.GOAMD64
	case "arm":
		return "GOARM", cfg.GOARM
	case "arm64":
		return "GOARM64", cfg.GOARM64
	case "mips", "mipsle":
		return "GOMIPS", cfg.GOMIPS
	case "mips64", "mips64le":
		return "GOMIPS64", cfg.GOMIPS64
	case "ppc64", "ppc64le":
		return "GOPPC64", cfg.GOPPC64
	case "riscv64":
		return "GORISCV64", cfg.GORISCV64
	case "wasm":
		return "GOWASM", cfg.GOWASM
	}
	return "", ""
}

// variantArchLevels tries the current setting first, then levels named by
// the constraint and its implied lower levels. A negated feature may require
// a lower level or the other floating-point mode. Only buildcfg-valid settings
// reach the matcher; feature-shaped tags are never synthesized as user tags.
func variantArchLevels(arch string, mentioned []string) (string, []string) {
	name, current := variantArchEnv(arch)
	levels := []string{current}
	if name == "" {
		return name, levels
	}
	features, err := buildcfg.GOGOARCHTags(arch, current)
	if err != nil {
		return name, nil
	}
	_, suffix, extended := strings.Cut(current, ",")
	for _, tag := range append(slices.Clone(mentioned), features...) {
		if level, ok := strings.CutPrefix(tag, arch+"."); ok {
			if extended && (arch == "arm" || arch == "arm64") {
				level += "," + suffix
			}
			if _, err := buildcfg.GOGOARCHTags(arch, level); err == nil && !slices.Contains(levels, level) {
				levels = append(levels, level)
			}
		}
	}
	switch arch {
	case "386":
		levels = append(levels, "sse2", "softfloat")
	case "mips", "mipsle", "mips64", "mips64le":
		levels = append(levels, "hardfloat", "softfloat")
	}
	return name, levels
}

// variantBuildContext derives fixed facts from the target configuration that
// the helper will receive. It retains compiler/release tags and unrelated tool
// tags but recomputes architecture and experiment facts for this target.
func variantBuildContext(vc variantCtx) (build.Context, bool) {
	ctxt := cfg.BuildContext
	ctxt.GOOS, ctxt.GOARCH = vc.goos, vc.goarch
	features, err := buildcfg.GOGOARCHTags(vc.goarch, vc.archValue)
	if err != nil {
		return build.Context{}, false
	}
	experiments, err := buildcfg.ParseGOEXPERIMENT(vc.goos, vc.goarch, cfg.RawGOEXPERIMENT)
	if err != nil {
		return build.Context{}, false
	}
	ctxt.ToolTags = nil
	for _, tag := range cfg.BuildContext.ToolTags {
		if !architectureFeatureTag(tag) && !strings.HasPrefix(tag, "goexperiment.") {
			ctxt.ToolTags = append(ctxt.ToolTags, tag)
		}
	}
	ctxt.ToolTags = append(ctxt.ToolTags, features...)
	for _, experiment := range experiments.Enabled() {
		ctxt.ToolTags = append(ctxt.ToolTags, "goexperiment."+experiment)
	}
	if !slices.Contains(ctxt.ToolTags, "logos") {
		ctxt.ToolTags = append(ctxt.ToolTags, "logos")
	}
	return ctxt, true
}
