// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cfg

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

func arenaModule(t *testing.T, toml string) string {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "logos.toml"), []byte(toml), 0o666); err != nil {
		t.Fatal(err)
	}
	return dir
}

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
	oldTags := slices.Clone(BuildContext.ToolTags)
	t.Setenv("GOEXPERIMENT", userExperiment)
	t.Cleanup(func() {
		Logos, RawGOEXPERIMENT, logosInitGOEXPERIMENT = oldLogos, oldRaw, oldInit
		BuildContext.ToolTags = oldTags
		recomputeExperiment()
	})
	Logos = logos
	RawGOEXPERIMENT = userExperiment
	logosInitGOEXPERIMENT = userExperiment
	recomputeExperiment()
}

func TestRefreshLogosArenaEnablesFromLogosToml(t *testing.T) {
	withArenaState(t, true, "")
	root := arenaModule(t, "[build]\narena = true\n")
	for range 2 {
		RefreshLogosArena([]string{root})
		if !Experiment.LogosArena || RawGOEXPERIMENT != "logosarena" || CleanGOEXPERIMENT == "" {
			t.Fatalf("after refresh: LogosArena=%v Raw=%q Clean=%q, want the experiment on", Experiment.LogosArena, RawGOEXPERIMENT, CleanGOEXPERIMENT)
		}
		if n := arenaTagCount(); n != 1 {
			t.Fatalf("goexperiment.logosarena tool tag appears %d times, want once", n)
		}
		if got := os.Getenv("GOEXPERIMENT"); got != "logosarena" {
			t.Fatalf("process GOEXPERIMENT = %q, want logosarena", got)
		}
	}
}

func TestRefreshLogosArenaFollowsTheCurrentRoots(t *testing.T) {
	withArenaState(t, true, "")
	on := arenaModule(t, "[build]\narena = true\n")
	off := arenaModule(t, "[build]\narena = false\n")
	RefreshLogosArena([]string{on})
	RefreshLogosArena([]string{off})
	if Experiment.LogosArena || RawGOEXPERIMENT != "" || arenaTagCount() != 0 {
		t.Errorf("arena = false after arena = true: LogosArena=%v Raw=%q tags=%d, want the startup experiment back", Experiment.LogosArena, RawGOEXPERIMENT, arenaTagCount())
	}
	RefreshLogosArena([]string{off, on})
	if !Experiment.LogosArena {
		t.Error("one workspace root with arena = true did not enable the arena")
	}
}

func TestRefreshLogosArenaKeepsOtherExperiments(t *testing.T) {
	withArenaState(t, true, "nojsonv2")
	if Experiment.JSONv2 {
		t.Fatal("nojsonv2 left jsonv2 on")
	}
	root := arenaModule(t, "[build]\narena = true\n")
	RefreshLogosArena([]string{root})
	if !Experiment.LogosArena || Experiment.JSONv2 || RawGOEXPERIMENT != "nojsonv2,logosarena" {
		t.Errorf("LogosArena=%v JSONv2=%v Raw=%q, want the arena added to the user's nojsonv2", Experiment.LogosArena, Experiment.JSONv2, RawGOEXPERIMENT)
	}
}

func TestRefreshLogosArenaDefaultIsOff(t *testing.T) {
	withArenaState(t, true, "")
	absent := arenaModule(t, "[build]\ncgo = \"forbid\"\n")
	RefreshLogosArena([]string{absent})
	broken := arenaModule(t, "[build]\narena = maybe\n")
	RefreshLogosArena([]string{broken})
	if Experiment.LogosArena || arenaTagCount() != 0 {
		t.Errorf("LogosArena=%v tags=%d with the key absent or the file broken, want off", Experiment.LogosArena, arenaTagCount())
	}
}

func TestRefreshLogosArenaKeepsEnvironmentExperiment(t *testing.T) {
	withArenaState(t, true, "logosarena")
	absent := arenaModule(t, "[build]\ncgo = \"forbid\"\n")
	RefreshLogosArena([]string{absent})
	if !Experiment.LogosArena || arenaTagCount() != 1 {
		t.Errorf("LogosArena=%v tags=%d, want GOEXPERIMENT=logosarena from the environment to stay on", Experiment.LogosArena, arenaTagCount())
	}
}

func TestRefreshLogosArenaIgnoredByPlainGo(t *testing.T) {
	withArenaState(t, false, "")
	root := arenaModule(t, "[build]\narena = true\n")
	RefreshLogosArena([]string{root})
	if Experiment.LogosArena || RawGOEXPERIMENT != "" || arenaTagCount() != 0 {
		t.Errorf("plain go read logos.toml: LogosArena=%v Raw=%q", Experiment.LogosArena, RawGOEXPERIMENT)
	}
}
