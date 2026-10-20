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

// LogosWasmProfile is the -logos.wasmprofile flag (M16-S1): on GOARCH=wasm
// the backend compiles calls whose callee cannot unwind without a resume
// point and drops the resume dispatch of functions that need none. The go
// command passes it for every package of a logos build that targets wasm.
// Plain go keeps it off until the G3 flip.
var LogosWasmProfile bool

// LogosWasmFeatures is the -logos.wasmfeatures flag: a comma-separated list
// of optional wasm features. The only name the compiler knows is
// stack-switching, and any entry implies the profile.
var LogosWasmFeatures string

// LogosWasmAssume is the -logos.wasmassume flag: a comma-separated list of
// symbol names whose callers treat them as MayYield=0 whatever their bodies
// do. It injects a wrong fact so a test can prove that a wrong MayYield=0
// traps instead of hanging (SPEC Sec 14.1).
var LogosWasmAssume string

var logosWasmFeatureNames = []string{"stack-switching"}

func init() {
	flag.BoolVar(&LogosWasmProfile, "logos.wasmprofile", false, "compile wasm with selective resume points")
	flag.StringVar(&LogosWasmFeatures, "logos.wasmfeatures", "", "enable wasm `features` (stack-switching)")
	flag.StringVar(&LogosWasmAssume, "logos.wasmassume", "", "treat the comma-separated `symbols` as MayYield=0 (fault injection for tests)")
}

// LogosWasmAssumed reports whether name is listed in -logos.wasmassume.
func LogosWasmAssumed(name string) bool {
	return LogosWasmAssume != "" && slices.Contains(strings.Split(LogosWasmAssume, ","), name)
}

// logosWasmApply copies the wasm profile flags into the link context. It
// runs once after flag parsing; a non-wasm target ignores every flag.
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

// logosWasmCheckFeatures rejects an unknown feature name and, for a known
// one, reports that the backend for it does not exist: the compiler never
// falls back to resume points when the build asked for stack switching.
func logosWasmCheckFeatures() {
	for f := range strings.SplitSeq(LogosWasmFeatures, ",") {
		if !slices.Contains(logosWasmFeatureNames, f) {
			log.Fatalf("unsupported -logos.wasmfeatures entry %q (want stack-switching)", f)
		}
	}
	log.Fatalf("-logos.wasmfeatures=%s: the wasm stack-switching backend (cont.new, resume) is not implemented", LogosWasmFeatures)
}
