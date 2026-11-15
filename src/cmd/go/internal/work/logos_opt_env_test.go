// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package work

import (
	"os"
	"slices"
	"testing"

	"cmd/go/internal/cfg"
	"cmd/go/internal/logos/settings"
)

func resetArchEnvState(t *testing.T, goarch string) {
	t.Helper()
	oldLogos, oldGoarch := cfg.Logos, cfg.Goarch
	oldAMD64, oldARM64 := cfg.GOAMD64, cfg.GOARM64
	oldTargetArchEnv := logosTargetArchEnv
	oldCfgChangedEnv := cfgChangedEnv
	oldEnvAMD64, hasAMD64 := os.LookupEnv("GOAMD64")
	oldEnvARM64, hasARM64 := os.LookupEnv("GOARM64")
	t.Cleanup(func() {
		cfg.Logos, cfg.Goarch = oldLogos, oldGoarch
		cfg.GOAMD64, cfg.GOARM64 = oldAMD64, oldARM64
		settings.SetCurrent(&settings.Settings{})
		logosTargetArchEnv = oldTargetArchEnv
		cfgChangedEnv = oldCfgChangedEnv
		if hasAMD64 {
			os.Setenv("GOAMD64", oldEnvAMD64)
		} else {
			os.Unsetenv("GOAMD64")
		}
		if hasARM64 {
			os.Setenv("GOARM64", oldEnvARM64)
		} else {
			os.Unsetenv("GOARM64")
		}
	})
}

func withArchEnvState(t *testing.T, goarch, amd64, arm64 string) {
	t.Helper()
	resetArchEnvState(t, goarch)
	cfg.Logos = true
	cfg.Goarch = goarch
	settings.SetCurrent(&settings.Settings{Logos: true, GOAMD64: amd64, GOARM64: arm64})
	logosTargetArchEnv = nil
	cfgChangedEnv = nil
}

func withPlainArchEnvState(t *testing.T, goarch string) {
	t.Helper()
	resetArchEnvState(t, goarch)
	cfg.Logos = false
	cfg.Goarch = goarch
	settings.SetCurrent(&settings.Settings{})
	logosTargetArchEnv = nil
	cfgChangedEnv = nil
}

func TestLogosTargetArchEnvPinned(t *testing.T) {
	withArchEnvState(t, "amd64", "v3", "v8.2")
	if got, want := LogosTargetArchEnv(), []string{"GOAMD64=v3"}; !slices.Equal(got, want) {
		t.Fatalf("LogosTargetArchEnv() = %v, want %v", got, want)
	}

	withArchEnvState(t, "arm64", "v3", "v8.2")
	if got, want := LogosTargetArchEnv(), []string{"GOARM64=v8.2"}; !slices.Equal(got, want) {
		t.Fatalf("LogosTargetArchEnv() = %v, want %v", got, want)
	}
}

func TestLogosTargetArchEnvEmpty(t *testing.T) {
	withArchEnvState(t, "amd64", "", "")
	if got := LogosTargetArchEnv(); len(got) != 0 {
		t.Fatalf("LogosTargetArchEnv() = %v, want empty with no levels", got)
	}
}

func TestLogosTargetArchEnvPlainMode(t *testing.T) {
	withPlainArchEnvState(t, "amd64")
	if got := LogosTargetArchEnv(); len(got) != 0 {
		t.Fatalf("LogosTargetArchEnv() = %v in plain mode, want empty", got)
	}
}

func TestLogosSetArchLevelWritesEnv(t *testing.T) {
	resetArchEnvState(t, "amd64")
	logosTargetArchEnv = nil
	cfgChangedEnv = nil
	logosSetArchLevel("GOAMD64", "v3")
	if cfg.GOAMD64 != "v3" || os.Getenv("GOAMD64") != "v3" {
		t.Fatalf("after set: cfg.GOAMD64=%q process=%q, want v3 v3", cfg.GOAMD64, os.Getenv("GOAMD64"))
	}
	logosSetArchLevel("GOAMD64", "v3")
	if n := countEntry(cfgChangedEnv, "GOAMD64=v3"); n != 2 {
		t.Fatalf("cfgChangedEnv=%q, want the level appended twice", cfgChangedEnv)
	}
	if n := countEntry(logosTargetArchEnv, "GOAMD64=v3"); n != 1 {
		t.Fatalf("targetArchEnv=%q, want the level updated in place", logosTargetArchEnv)
	}
}

func countEntry(env []string, want string) int {
	n := 0
	for _, e := range env {
		if e == want {
			n++
		}
	}
	return n
}
