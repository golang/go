// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cfg

import (
	"go/build"
	"internal/buildcfg"
	"os"
	"runtime"
	"slices"
	"strings"
	"testing"

	"cmd/go/internal/logos/settings"
)

func arenaTagCount() int {
	n := 0
	for _, tag := range BuildContext.ToolTags {
		if tag == "goexperiment.logosarena" {
			n++
		}
	}
	return n
}

func withArenaState(t *testing.T, logos bool, userExperiment string) {
	t.Helper()
	oldLogos, oldRaw, oldInit := Logos, RawGOEXPERIMENT, logosInitGOEXPERIMENT
	oldChanged := CGOChanged
	oldTags := slices.Clone(BuildContext.ToolTags)
	t.Setenv("GOEXPERIMENT", userExperiment)
	t.Cleanup(func() {
		Logos, RawGOEXPERIMENT, logosInitGOEXPERIMENT = oldLogos, oldRaw, oldInit
		CGOChanged = oldChanged
		BuildContext.ToolTags = oldTags
		recomputeExperiment()
	})
	Logos = logos
	RawGOEXPERIMENT = userExperiment
	logosInitGOEXPERIMENT = userExperiment
	recomputeExperiment()
}

func TestApplyLogosEnvArena(t *testing.T) {
	withArenaState(t, true, "")
	ApplyLogosEnv(&settings.Settings{Logos: true, Experiment: "logosarena"})
	if !Experiment.LogosArena || CleanGOEXPERIMENT == "" {
		t.Fatalf("after apply: LogosArena=%v Raw=%q Clean=%q, want the experiment on", Experiment.LogosArena, RawGOEXPERIMENT, CleanGOEXPERIMENT)
	}
	if n := arenaTagCount(); n != 1 {
		t.Fatalf("goexperiment.logosarena tool tag appears %d times, want once", n)
	}
	publishedExp, err := buildcfg.ParseGOEXPERIMENT(Goos, Goarch, os.Getenv("GOEXPERIMENT"))
	if err != nil {
		t.Fatalf("parse process GOEXPERIMENT %q: %v", os.Getenv("GOEXPERIMENT"), err)
	}
	if !publishedExp.LogosArena {
		t.Fatalf("process GOEXPERIMENT = %q, want logosarena enabled", os.Getenv("GOEXPERIMENT"))
	}
}

func TestApplyLogosEnvArenaOff(t *testing.T) {
	withArenaState(t, true, "")
	ApplyLogosEnv(&settings.Settings{Logos: true, Experiment: "logosarena"})
	ApplyLogosEnv(&settings.Settings{Logos: true, Experiment: ""})
	if Experiment.LogosArena || arenaTagCount() != 0 {
		t.Errorf("after apply off: LogosArena=%v tags=%d, want the startup experiment back", Experiment.LogosArena, arenaTagCount())
	}
	publishedExp, err := buildcfg.ParseGOEXPERIMENT(Goos, Goarch, os.Getenv("GOEXPERIMENT"))
	if err != nil {
		t.Fatalf("parse process GOEXPERIMENT %q: %v", os.Getenv("GOEXPERIMENT"), err)
	}
	if publishedExp.LogosArena {
		t.Errorf("process GOEXPERIMENT = %q, want logosarena disabled", os.Getenv("GOEXPERIMENT"))
	}
}

func TestApplyLogosEnvKeepsOtherExperiments(t *testing.T) {
	withArenaState(t, true, "nojsonv2")
	if Experiment.JSONv2 {
		t.Fatal("nojsonv2 left jsonv2 on")
	}
	ApplyLogosEnv(&settings.Settings{Logos: true, Experiment: "nojsonv2,logosarena"})
	if !Experiment.LogosArena || Experiment.JSONv2 {
		t.Errorf("LogosArena=%v JSONv2=%v Raw=%q, want the arena added to the user's nojsonv2", Experiment.LogosArena, Experiment.JSONv2, RawGOEXPERIMENT)
	}
	publishedExp, err := buildcfg.ParseGOEXPERIMENT(Goos, Goarch, os.Getenv("GOEXPERIMENT"))
	if err != nil {
		t.Fatalf("parse process GOEXPERIMENT %q: %v", os.Getenv("GOEXPERIMENT"), err)
	}
	if !publishedExp.LogosArena || publishedExp.JSONv2 {
		t.Errorf("published GOEXPERIMENT=%q: LogosArena=%v JSONv2=%v, want arena enabled and jsonv2 disabled",
			os.Getenv("GOEXPERIMENT"), publishedExp.LogosArena, publishedExp.JSONv2)
	}
}

func TestApplyLogosEnvIgnoresPlainGo(t *testing.T) {
	withArenaState(t, false, "")
	ApplyLogosEnv(&settings.Settings{Logos: false, Experiment: "logosarena"})
	if Experiment.LogosArena || RawGOEXPERIMENT != "" || arenaTagCount() != 0 {
		t.Errorf("plain go applied settings: LogosArena=%v Raw=%q", Experiment.LogosArena, RawGOEXPERIMENT)
	}
}

func TestApplyLogosEnvCgoOpinion(t *testing.T) {
	withArenaState(t, true, "")
	oldCgo := BuildContext.CgoEnabled
	t.Cleanup(func() { BuildContext.CgoEnabled = oldCgo })
	ApplyLogosEnv(&settings.Settings{Logos: true, Experiment: RawGOEXPERIMENT, Cgo: settings.Cgo{Enabled: true, Opinion: true, Changed: true}})
	if !BuildContext.CgoEnabled || !CGOChanged {
		t.Errorf("after opinionated apply: CgoEnabled=%v CGOChanged=%v, want true true", BuildContext.CgoEnabled, CGOChanged)
	}
	BuildContext.CgoEnabled = oldCgo
	ApplyLogosEnv(&settings.Settings{Logos: true, Experiment: RawGOEXPERIMENT})
	if BuildContext.CgoEnabled != oldCgo || CGOChanged {
		t.Errorf("after silent apply: CgoEnabled=%v CGOChanged=%v, want %v false", BuildContext.CgoEnabled, CGOChanged, oldCgo)
	}
}

func TestApplyLogosEnvWasmDefault(t *testing.T) {
	withArenaState(t, true, "")
	oldGoarch := Goarch
	Goarch = "wasm"
	t.Cleanup(func() { Goarch = oldGoarch; recomputeExperiment() })
	recomputeExperiment()
	ApplyLogosEnv(&settings.Settings{Logos: true})
	if !Experiment.PreemptibleLoops || preemptibleTagCount() != 1 {
		t.Fatalf("after apply: PreemptibleLoops=%v Raw=%q tags=%d, want the wasm default on",
			Experiment.PreemptibleLoops, RawGOEXPERIMENT, preemptibleTagCount())
	}
}

func withPreemptibleState(t *testing.T, logos bool, goarch, userExperiment string) {
	t.Helper()
	oldLogos, oldRaw, oldInit, oldGoarch := Logos, RawGOEXPERIMENT, logosInitGOEXPERIMENT, Goarch
	oldTags := slices.Clone(BuildContext.ToolTags)
	oldEnv := os.Getenv("GOEXPERIMENT")
	t.Cleanup(func() {
		Logos, RawGOEXPERIMENT, logosInitGOEXPERIMENT, Goarch = oldLogos, oldRaw, oldInit, oldGoarch
		BuildContext.ToolTags = oldTags
		os.Setenv("GOEXPERIMENT", oldEnv)
		recomputeExperiment()
	})
	Logos = logos
	Goarch = goarch
	RawGOEXPERIMENT = userExperiment
	logosInitGOEXPERIMENT = userExperiment
	t.Setenv("GOEXPERIMENT", userExperiment)
	recomputeExperiment()
}

func preemptibleTagCount() int {
	n := 0
	for _, tag := range BuildContext.ToolTags {
		if tag == "goexperiment.preemptibleloops" {
			n++
		}
	}
	return n
}

func TestRefreshLogosPreemptibleDefaultsOnWasm(t *testing.T) {
	withPreemptibleState(t, true, "wasm", "")
	if Experiment.PreemptibleLoops {
		t.Fatal("experiment on before refresh with empty GOEXPERIMENT")
	}
	for range 2 {
		RefreshLogosPreemptible()
		if !Experiment.PreemptibleLoops || RawGOEXPERIMENT != "preemptibleloops" {
			t.Fatalf("after refresh: PreemptibleLoops=%v Raw=%q, want the wasm default on",
				Experiment.PreemptibleLoops, RawGOEXPERIMENT)
		}
		if !strings.Contains(CleanGOEXPERIMENT, "preemptibleloops") {
			t.Fatalf("CleanGOEXPERIMENT=%q, want the default in the cache key", CleanGOEXPERIMENT)
		}
		if n := preemptibleTagCount(); n != 1 {
			t.Fatalf("goexperiment.preemptibleloops tool tag appears %d times, want once", n)
		}
		if got := os.Getenv("GOEXPERIMENT"); got != "preemptibleloops" {
			t.Fatalf("process GOEXPERIMENT = %q, want preemptibleloops", got)
		}
	}
}

func TestRefreshLogosPreemptiblePlainGoUnchanged(t *testing.T) {
	withPreemptibleState(t, false, "wasm", "")
	RefreshLogosPreemptible()
	if Experiment.PreemptibleLoops || RawGOEXPERIMENT != "" || preemptibleTagCount() != 0 {
		t.Errorf("plain go wasm build: PreemptibleLoops=%v Raw=%q tags=%d, want unchanged until G3",
			Experiment.PreemptibleLoops, RawGOEXPERIMENT, preemptibleTagCount())
	}
}

func TestRefreshLogosPreemptibleNonWasmUnchanged(t *testing.T) {
	withPreemptibleState(t, true, "amd64", "")
	RefreshLogosPreemptible()
	if Experiment.PreemptibleLoops || RawGOEXPERIMENT != "" || preemptibleTagCount() != 0 {
		t.Errorf("logos amd64 build: PreemptibleLoops=%v Raw=%q tags=%d, want the default off",
			Experiment.PreemptibleLoops, RawGOEXPERIMENT, preemptibleTagCount())
	}
}

func TestRefreshLogosPreemptibleRespectsUserOptOut(t *testing.T) {
	for _, raw := range []string{
		"nopreemptibleloops",
		"none",
		"preemptibleloops,none",
		"nojsonv2,nopreemptibleloops",
	} {
		withPreemptibleState(t, true, "wasm", raw)
		RefreshLogosPreemptible()
		if Experiment.PreemptibleLoops {
			t.Errorf("GOEXPERIMENT=%q: default overrode an explicit user opt-out", raw)
		}
		if RawGOEXPERIMENT != raw {
			t.Errorf("GOEXPERIMENT=%q: Raw=%q, want untouched", raw, RawGOEXPERIMENT)
		}
	}
}

func TestRefreshLogosPreemptibleKeepsExplicitEnable(t *testing.T) {
	withPreemptibleState(t, true, "wasm", "preemptibleloops")
	RefreshLogosPreemptible()
	if !Experiment.PreemptibleLoops || RawGOEXPERIMENT != "preemptibleloops" {
		t.Errorf("explicit preemptibleloops: PreemptibleLoops=%v Raw=%q, want kept without duplication",
			Experiment.PreemptibleLoops, RawGOEXPERIMENT)
	}
	withPreemptibleState(t, true, "amd64", "preemptibleloops")
	RefreshLogosPreemptible()
	if !Experiment.PreemptibleLoops || RawGOEXPERIMENT != "preemptibleloops" {
		t.Errorf("explicit preemptibleloops on amd64: PreemptibleLoops=%v Raw=%q, want kept",
			Experiment.PreemptibleLoops, RawGOEXPERIMENT)
	}
}

func TestRefreshLogosPreemptibleKeepsOtherExperiments(t *testing.T) {
	withPreemptibleState(t, true, "wasm", "nojsonv2")
	RefreshLogosPreemptible()
	if !Experiment.PreemptibleLoops || Experiment.JSONv2 || RawGOEXPERIMENT != "nojsonv2,preemptibleloops" {
		t.Errorf("PreemptibleLoops=%v JSONv2=%v Raw=%q, want the default added to the user's nojsonv2",
			Experiment.PreemptibleLoops, Experiment.JSONv2, RawGOEXPERIMENT)
	}
}

func TestRefreshLogosPreemptibleStripsOnRetarget(t *testing.T) {
	withPreemptibleState(t, true, "wasm", "")
	RefreshLogosPreemptible()
	if !Experiment.PreemptibleLoops {
		t.Fatal("wasm default did not apply")
	}
	Goarch = "amd64"
	RefreshLogosPreemptible()
	if Experiment.PreemptibleLoops || RawGOEXPERIMENT != "" || preemptibleTagCount() != 0 {
		t.Errorf("after retarget: PreemptibleLoops=%v Raw=%q tags=%d, want the wasm default stripped",
			Experiment.PreemptibleLoops, RawGOEXPERIMENT, preemptibleTagCount())
	}
}

func withForceHostGlobals(t *testing.T) {
	t.Helper()
	oldGoos, oldGoarch, oldExe := Goos, Goarch, ExeSuffix
	old386, oldAmd64, oldArm, oldArm64 := GO386, GOAMD64, GOARM, GOARM64
	oldMips, oldMips64, oldPpc64, oldRiscv64, oldWasm := GOMIPS, GOMIPS64, GOPPC64, GORISCV64, GOWASM
	oldCtx, oldToolDir := BuildContext, build.ToolDir
	oldGoroot, oldBin, oldPkg, oldSrc := GOROOT, GOROOTbin, GOROOTpkg, GOROOTsrc
	t.Cleanup(func() {
		Goos, Goarch, ExeSuffix = oldGoos, oldGoarch, oldExe
		GO386, GOAMD64, GOARM, GOARM64 = old386, oldAmd64, oldArm, oldArm64
		GOMIPS, GOMIPS64, GOPPC64, GORISCV64, GOWASM = oldMips, oldMips64, oldPpc64, oldRiscv64, oldWasm
		BuildContext, build.ToolDir = oldCtx, oldToolDir
		GOROOT, GOROOTbin, GOROOTpkg, GOROOTsrc = oldGoroot, oldBin, oldPkg, oldSrc
		recomputeExperiment()
	})
}

func TestForceHostDropsWasmDefault(t *testing.T) {
	if runtime.GOARCH == "wasm" {
		t.Skip("host is wasm: ForceHost cannot drop the default")
	}
	withPreemptibleState(t, true, "wasm", "")
	withForceHostGlobals(t)
	RefreshLogosPreemptible()
	if !Experiment.PreemptibleLoops {
		t.Fatal("wasm default did not apply")
	}

	ForceHost()
	if Experiment.PreemptibleLoops || RawGOEXPERIMENT != "" || preemptibleTagCount() != 0 {
		t.Errorf("after ForceHost: PreemptibleLoops=%v Raw=%q tags=%d, want the wasm default gone",
			Experiment.PreemptibleLoops, RawGOEXPERIMENT, preemptibleTagCount())
	}
}

func TestForceHostKeepsExplicitPreemptible(t *testing.T) {
	if runtime.GOARCH == "wasm" {
		t.Skip("host is wasm: nothing to strip")
	}
	withPreemptibleState(t, true, "wasm", "preemptibleloops")
	withForceHostGlobals(t)

	ForceHost()
	if !Experiment.PreemptibleLoops || RawGOEXPERIMENT != "preemptibleloops" {
		t.Errorf("after ForceHost: PreemptibleLoops=%v Raw=%q, want the explicit user experiment kept",
			Experiment.PreemptibleLoops, RawGOEXPERIMENT)
	}
}
