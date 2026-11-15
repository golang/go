// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package base

import (
	"flag"
	"internal/buildcfg"
	"log"
	"slices"
	"strings"
)

var LogosWasmProfile bool

var LogosWasmFeatures string

var LogosWasmAssume string

var logosWasmFeatureNames = []string{"stack-switching"}

func init() {
	flag.BoolVar(&LogosWasmProfile, "logos.wasmprofile", false, "compile wasm with selective resume points")
	flag.StringVar(&LogosWasmFeatures, "logos.wasmfeatures", "", "enable wasm `features` (stack-switching)")
	flag.StringVar(&LogosWasmAssume, "logos.wasmassume", "", "treat the comma-separated `symbols` as MayYield=0 (fault injection for tests)")
}

func LogosWasmAssumed(name string) bool {
	return LogosWasmAssume != "" && slices.Contains(strings.Split(LogosWasmAssume, ","), name)
}

func logosWasmApply() {
	if buildcfg.GOARCH != "wasm" {
		return
	}
	if LogosWasmFeatures != "" {
		logosWasmCheckFeatures()
		LogosWasmProfile = true
	}
	Ctxt.WasmProfile = LogosWasmProfile
	Ctxt.KeepProgs = LogosWasmProfile
}

func logosWasmCheckFeatures() {
	for f := range strings.SplitSeq(LogosWasmFeatures, ",") {
		if !slices.Contains(logosWasmFeatureNames, f) {
			log.Fatalf("unsupported -logos.wasmfeatures entry %q (want stack-switching)", f)
		}
		Ctxt.WasmStackSwitch = true
	}
}
