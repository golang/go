// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"runtime"
	"slices"
	"strings"
	"testing"

	"cmd/go/internal/cfg"
	"cmd/go/internal/logos/settings"
)

func testExperimentEnv(t *testing.T, logos bool, raw string, original []string) {
	t.Helper()
	oldLogos, oldRaw, oldEnv := cfg.Logos, cfg.RawGOEXPERIMENT, cfg.OrigEnv
	oldSettings := settings.Current()
	t.Cleanup(func() {
		cfg.Logos, cfg.RawGOEXPERIMENT, cfg.OrigEnv = oldLogos, oldRaw, oldEnv
		settings.SetCurrent(oldSettings)
	})
	cfg.Logos, cfg.RawGOEXPERIMENT, cfg.OrigEnv = logos, raw, original
	settings.SetCurrent(&settings.Settings{})
}

func TestLogosTestExperimentEnv(t *testing.T) {
	for _, tt := range []struct {
		name     string
		logos    bool
		raw      string
		original []string
		want     string
		present  bool
	}{
		{"default", true, "logosarena", nil, "logosarena", true},
		{"explicit-empty", true, "logosarena", []string{"GOEXPERIMENT="}, "logosarena", true},
		{"none-with-arena", true, "none,logosarena", []string{"GOEXPERIMENT=none"}, "none,logosarena", true},
		{"classic-json", true, "nojsonv2,logosarena", []string{"GOEXPERIMENT=nojsonv2"}, "nojsonv2,logosarena", true},
		{"jsonv2", true, "jsonv2,logosarena", []string{"GOEXPERIMENT=jsonv2"}, "jsonv2,logosarena", true},
		{"arena-off", true, "", nil, "", true},
		{"none-arena-off", true, "none", []string{"GOEXPERIMENT=none"}, "none", true},
		{"duplicates", true, "nojsonv2,logosarena", []string{"GOEXPERIMENT=none", "GOEXPERIMENT=nojsonv2"}, "nojsonv2,logosarena", true},
		{"plain-unset", false, "", nil, "", false},
		{"plain-empty", false, "", []string{"GOEXPERIMENT="}, "", true},
		{"plain-none", false, "none", []string{"GOEXPERIMENT=none"}, "none", true},
		{"plain-classic", false, "nojsonv2", []string{"GOEXPERIMENT=nojsonv2"}, "nojsonv2", true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			original := append(slices.Clone(tt.original), "LOGOS_TEST_OTHER=kept")
			testExperimentEnv(t, tt.logos, tt.raw, original)
			env := logosAppendTestEnv(slices.Clone(original))
			var value string
			var present bool
			for _, entry := range env {
				if v, ok := strings.CutPrefix(entry, "GOEXPERIMENT="); ok {
					value, present = v, true
				}
			}
			if value != tt.want || present != tt.present {
				t.Errorf("test GOEXPERIMENT = (%q, %v), want (%q, %v)", value, present, tt.want, tt.present)
			}
			if got, ok := logosInheritedTestEnv("GOEXPERIMENT"); got != tt.want || ok != tt.present {
				t.Errorf("cached GOEXPERIMENT = (%q, %v), want (%q, %v)", got, ok, tt.want, tt.present)
			}
			if got, ok := logosInheritedTestEnv("LOGOS_TEST_OTHER"); got != "kept" || !ok {
				t.Errorf("unrelated inherited variable = (%q, %v), want (kept, true)", got, ok)
			}
			if !tt.logos && !slices.Equal(env, original) {
				t.Errorf("plain go environment changed: %q, want %q", env, original)
			}
		})
	}
}

func TestLogosTestExperimentCache(t *testing.T) {
	testExperimentEnv(t, true, "logosarena", nil)
	first := hashGetenv("GOEXPERIMENT")
	cfg.RawGOEXPERIMENT = "nojsonv2,logosarena"
	if first == hashGetenv("GOEXPERIMENT") {
		t.Fatal("test cache ignores the effective experiment inherited by subprocesses")
	}
	cfg.Logos = false
	first = hashGetenv("GOEXPERIMENT")
	cfg.RawGOEXPERIMENT = "logosarena"
	if first != hashGetenv("GOEXPERIMENT") {
		t.Fatal("plain go hashes an experiment absent from its test environment")
	}
}

func TestLogosTestExperimentEnvNames(t *testing.T) {
	testExperimentEnv(t, true, "logosarena", []string{"GOEXPERIMENT_EXTRA=kept"})
	if got, ok := logosInheritedTestEnv("GOEXPERIMENT_EXTRA"); got != "kept" || !ok {
		t.Fatalf("prefixed name = (%q, %v), want (kept, true)", got, ok)
	}
	got, ok := logosInheritedTestEnv("goexperiment")
	if runtime.GOOS == "windows" {
		if got != "logosarena" || !ok {
			t.Fatalf("Windows name = (%q, %v), want (logosarena, true)", got, ok)
		}
	} else if ok {
		t.Fatalf("case-sensitive name unexpectedly inherited %q", got)
	}
}
