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
	"cmd/go/internal/logos/settings"
)

const logosWasmPlainDefault = false

func logosAddWasmFeaturesFlag(cmd *base.Command) {
	cmd.Flag.Var(settings.WasmFeaturesFlag(&settings.CommandLine), "wasm-features", "")
}

func logosWasmProfile() bool {
	if cfg.Goarch != "wasm" {
		return false
	}
	return cfg.Logos || settings.Current().WasmFeatures != "" || logosWasmPlainDefault
}

func logosWasmFlags() []string {
	if !logosWasmProfile() {
		return nil
	}
	flags := []string{"-logos.wasmprofile"}
	if settings.Current().WasmFeatures != "" {
		flags = append(flags, "-logos.wasmfeatures="+settings.Current().WasmFeatures)
	}
	return flags
}

func logosWasmAsmFlags() []string {
	if cfg.Goarch != "wasm" || !slices.Contains(strings.Split(settings.Current().WasmFeatures, ","), "stack-switching") {
		return nil
	}
	return []string{"-logos.wasmstackswitch"}
}

func logosWasmActionID(h io.Writer) {
	if flags := logosWasmFlags(); len(flags) > 0 {
		fmt.Fprintf(h, "logos wasm %q\n", flags)
	}
}
