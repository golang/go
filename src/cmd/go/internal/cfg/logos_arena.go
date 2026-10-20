// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cfg

import (
	"os"
	"strings"

	"cmd/go/internal/logos/config"
)

// logosArenaExperiment is the GOEXPERIMENT name that logos.toml [build] arena
// selects.
const logosArenaExperiment = "logosarena"

// logosInitGOEXPERIMENT snapshots the user GOEXPERIMENT at process startup,
// so RefreshLogosArena derives the effective value from the user's setting
// on every call instead of stacking its own addition.
var logosInitGOEXPERIMENT = RawGOEXPERIMENT

// RefreshLogosArena sets GOEXPERIMENT=logosarena for the build when the
// logos.toml of any main module root has [build] arena = true. GOEXPERIMENT
// is fixed per build, so the experiment tags and the clean value that keys
// the build cache are recomputed here, before package loading, and the
// standard library is rebuilt with the experiment under its own cache keys.
// A root without the key, a broken logos.toml, or plain go leaves the user's
// GOEXPERIMENT untouched, so GOEXPERIMENT=logosarena in the environment
// still enables the arena. With no roots the module above the working
// directory decides. Callers: envcmd MkEnv runs it after the -C chdir and
// before MkEnv bakes GOEXPERIMENT into the process environment; work
// BuildInit re-runs it once the full workspace member list is known.
func RefreshLogosArena(mainRoots []string) {
	if !Logos {
		return
	}
	roots := mainRoots
	if len(roots) == 0 {
		if root := cwdModuleRoot(); root != "" {
			roots = []string{root}
		}
	}
	raw := logosInitGOEXPERIMENT
	if logosAnyRootArena(roots) {
		raw = withLogosArena(raw)
	}
	if raw == RawGOEXPERIMENT {
		return
	}
	RawGOEXPERIMENT = raw
	recomputeExperiment()
	os.Setenv("GOEXPERIMENT", RawGOEXPERIMENT)
	for i := range CmdEnv {
		if CmdEnv[i].Name == "GOEXPERIMENT" {
			CmdEnv[i].Value = RawGOEXPERIMENT
		}
	}
}

func logosAnyRootArena(roots []string) bool {
	for _, root := range roots {
		cf, err := config.Load(root)
		if err == nil && cf != nil && cf.Build.Arena {
			return true
		}
	}
	return false
}

// withLogosArena appends the arena experiment to a GOEXPERIMENT value. The
// later entry wins over an earlier nologosarena.
func withLogosArena(raw string) string {
	if raw == "" {
		return logosArenaExperiment
	}
	return raw + "," + logosArenaExperiment
}

// recomputeExperiment drops the experiment tool tags computed for the
// previous GOEXPERIMENT and computes them again.
func recomputeExperiment() {
	kept := BuildContext.ToolTags[:0:0]
	for _, tag := range BuildContext.ToolTags {
		if !strings.HasPrefix(tag, "goexperiment.") {
			kept = append(kept, tag)
		}
	}
	BuildContext.ToolTags = kept
	computeExperiment()
}
