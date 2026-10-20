// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package work

import (
	"fmt"
	"io"
	"slices"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
)

// logosWasmPlainDefault is the G3 switch of the WASM profile (M16-S1):
// plain go builds the profile only on request until run.bash passes with
// it, then this becomes true. logos builds it by default.
const logosWasmPlainDefault = false

// logosWasmFeatureNames lists the values --wasm-features accepts.
var logosWasmFeatureNames = []string{"stack-switching"}

// wasmFeaturesFlag implements --wasm-features: it validates each entry
// where the flag is parsed and stores the list in cfg.WasmFeatures.
type wasmFeaturesFlag struct{}

func (wasmFeaturesFlag) String() string { return cfg.WasmFeatures }

func (wasmFeaturesFlag) Set(s string) error {
	for f := range strings.SplitSeq(s, ",") {
		if !slices.Contains(logosWasmFeatureNames, f) {
			return fmt.Errorf("unsupported wasm feature %q (want %s)", f, strings.Join(logosWasmFeatureNames, ", "))
		}
	}
	cfg.WasmFeatures = s
	return nil
}

// logosAddWasmFeaturesFlag registers --wasm-features on logos build and
// go build. It has no help text, so go help build stays upstream.
func logosAddWasmFeaturesFlag(cmd *base.Command) {
	cmd.Flag.Var(wasmFeaturesFlag{}, "wasm-features", "")
}

// logosWasmProfile reports whether compile actions get the WASM profile:
// GOARCH=wasm builds of logos, builds that name a wasm feature, and every
// build once logosWasmPlainDefault flips.
func logosWasmProfile() bool {
	if cfg.Goarch != "wasm" {
		return false
	}
	return cfg.Logos || cfg.WasmFeatures != "" || logosWasmPlainDefault
}

// logosWasmFlags returns the compiler flags of the WASM profile.
func logosWasmFlags() []string {
	if !logosWasmProfile() {
		return nil
	}
	flags := []string{"-logos.wasmprofile"}
	if cfg.WasmFeatures != "" {
		flags = append(flags, "-logos.wasmfeatures="+cfg.WasmFeatures)
	}
	return flags
}

// logosWasmActionID adds the profile flags to a compile action ID, so a
// switch of profile or feature recompiles and never reuses cached output.
// It is the hook line in buildActionID.
func logosWasmActionID(h io.Writer) {
	if flags := logosWasmFlags(); len(flags) > 0 {
		fmt.Fprintf(h, "logos wasm %q\n", flags)
	}
}
