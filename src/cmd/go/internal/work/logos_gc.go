// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package work

import (
	"slices"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/load"
	"cmd/go/internal/logos/config"
	"cmd/go/internal/modload"
)

// logosGCPendingPhase maps a collector mode that is not finished to the
// moving-collector phase that must pass its gate before the mode opens
// (SPEC Sec 19, M19-S5). Selecting such a mode is an error naming the phase;
// passing a phase deletes its entry.
var logosGCPendingPhase = map[string]string{"immix": "phase 1", "zgc": "phase 2"}

// gcModeFlag is the -gc flag value (SPEC Sec 19): it records the text and
// that the flag appeared, and initLogosGC checks it. The flag has no help
// text under plain go; logos documents it.
type gcModeFlag struct{}

// String returns the current value.
func (gcModeFlag) String() string { return cfg.BuildGC }

// Set records the value and that the flag was given.
func (gcModeFlag) Set(s string) error {
	cfg.BuildGC, cfg.BuildGCSet = s, true
	return nil
}

// addLogosGCFlag registers the -gc flag on cmd.
func addLogosGCFlag(cmd *base.Command) {
	cmd.Flag.Var(gcModeFlag{}, "gc", "")
}

// logosGCProgram names the command in -gc messages.
func logosGCProgram() string {
	if cfg.Logos {
		return "logos"
	}
	return "go"
}

// selectLogosGC returns the collector mode of this build and whether one is
// selected: the -gc flag, else under logos the [build] gc key of the main
// module's logos.toml, else the logos default (greentea, or wasm on
// GOARCH=wasm). Plain go with no flag selects none and keeps the upstream
// collector and pacing until run.bash passes with a changed default.
func selectLogosGC(ld *modload.Loader) (mode string, selected bool) {
	if cfg.BuildGCSet {
		return cfg.BuildGC, true
	}
	if !cfg.Logos {
		return "", false
	}
	if mode, set := load.LogosBuildGC(ld); set {
		return mode, true
	}
	if cfg.Goarch == "wasm" {
		return "wasm", true
	}
	return "greentea", true
}

// initLogosGC resolves and checks the collector mode in BuildInit. An
// unknown mode and a mode whose phase is pending are fatal, with the prefix of
// the running command. The mode reaches the linker as
// -X=runtime.logosGCMode=MODE through forcedLdflags, which enter the link
// action ID, and reaches the compiler through load's gcflags hook, which
// reads cfg.LogosGC.
func initLogosGC(ld *modload.Loader) {
	mode, selected := selectLogosGC(ld)
	if !selected {
		return
	}
	if !slices.Contains(config.GCModes, mode) {
		base.Fatalf("%s build: unknown -gc mode %q (want %s)", logosGCProgram(), mode, strings.Join(config.GCModes, ", "))
	}
	if phase, pending := logosGCPendingPhase[mode]; pending {
		base.Fatalf("%s build: -gc=%s requires GOEXPERIMENT=logosgcmoving %s, which is not complete", logosGCProgram(), mode, phase)
	}
	cfg.LogosGC = mode
	forcedLdflags = append(forcedLdflags, "-X=runtime.logosGCMode="+mode)
}
