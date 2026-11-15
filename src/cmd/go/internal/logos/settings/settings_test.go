// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package settings

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"cmd/go/internal/logos/config"
)

func logosModule(t *testing.T, toml string) string {
	t.Helper()
	dir := t.TempDir()
	if toml != "" {
		if err := os.WriteFile(filepath.Join(dir, "logos.toml"), []byte(toml), 0o666); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

func logosSrc(t *testing.T, dir string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Join(dir, "pkg"), 0o777); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "pkg", "a.lgo"), []byte("package p\n"), 0o666); err != nil {
		t.Fatal(err)
	}
}

func TestLoadPrecedence(t *testing.T) {
	o0 := logosModule(t, "[build]\nopt_level = \"O0\"\n")
	o1 := logosModule(t, "[build]\nopt_level = \"O1\"\n")
	o2 := logosModule(t, "[build]\nopt_level = \"O2\"\n")
	o3 := logosModule(t, "[build]\nopt_level = \"O3\"\n")
	os2 := logosModule(t, "[build]\nopt_level = \"Os\"\n")
	empty := logosModule(t, "")
	arenaOnFile := logosModule(t, "[build]\narena = true\n")
	broken := logosModule(t, "[build]\narena = maybe\n")
	baseline := logosModule(t, "[build]\ntarget_cpu = \"baseline\"\n")
	det := logosModule(t, "[build]\ntarget_cpu = \"deterministic\"\n")
	native := logosModule(t, "[build]\ntarget_cpu = \"native\"\n")
	kitchen := logosModule(t, "[build]\narena = true\ncgo = \"allow\"\ntarget_cpu = \"baseline\"\nopt_level = \"O3\"\n")

	tests := []struct {
		name       string
		modRoot    string
		flags      Flags
		optLevel   string
		amd64      string
		arm        string
		experiment string
		arena      bool
		pedantic   config.PedanticMode
		module     bool
		wantErr    string
	}{
		{
			name: "flagBeatsFile", modRoot: o3,
			flags: Flags{Logos: true, Opt: "2"}, optLevel: "2",
			amd64: "v3", arm: "v8.2",
			experiment: "logosarena", arena: true, module: true,
		},
		{
			name: "fileLevel", modRoot: o3,
			flags: Flags{Logos: true}, optLevel: "3",
			amd64: "v3", arm: "v8.2",
			experiment: "logosarena", arena: true, module: true,
		},
		{
			name: "fileLevelO0", modRoot: o0,
			flags: Flags{Logos: true}, optLevel: "0",
			amd64: "v3", arm: "v8.2",
			experiment: "logosarena", arena: true, module: true,
		},
		{
			name: "fileLevelO1", modRoot: o1,
			flags: Flags{Logos: true}, optLevel: "1",
			amd64: "v3", arm: "v8.2",
			experiment: "logosarena", arena: true, module: true,
		},
		{
			name: "fileLevelO2", modRoot: o2,
			flags: Flags{Logos: true}, optLevel: "2",
			amd64: "v3", arm: "v8.2",
			experiment: "logosarena", arena: true, module: true,
		},
		{
			name: "fileLevelS", modRoot: os2,
			flags: Flags{Logos: true}, optLevel: "s",
			amd64: "v3", arm: "v8.2",
			experiment: "logosarena", arena: true, module: true,
		},
		{
			name: "flagSBeatsFile", modRoot: o3,
			flags: Flags{Logos: true, Opt: "s"}, optLevel: "s",
			amd64: "v3", arm: "v8.2",
			experiment: "logosarena", arena: true, module: true,
		},
		{
			name: "defaultO2", modRoot: empty,
			flags: Flags{Logos: true}, optLevel: "2",
			amd64: "v3", arm: "v8.2",
			experiment: "logosarena", arena: true, module: true,
		},
		{
			name: "plainGoDefault", modRoot: empty,
			flags: Flags{}, optLevel: "1", experiment: "",
		},
		{
			name: "plainGoReadsNothing", modRoot: kitchen,
			flags: Flags{}, optLevel: "1", experiment: "",
		},
		{
			name: "plainGoOpt", modRoot: empty,
			flags: Flags{Opt: "0"}, optLevel: "0",
		},
		{
			name: "baselineBeatsEnvAMD64", modRoot: baseline,
			flags: Flags{Logos: true, Env: Env{GOAMD64: "v2"}}, optLevel: "2",
			amd64: "v1", arm: "v8.0",
			experiment: "logosarena", arena: true, module: true,
		},
		{
			name: "baselineBeatsEnvARM64", modRoot: baseline,
			flags: Flags{Logos: true, Env: Env{GOARM64: "v8.3"}}, optLevel: "2",
			amd64: "v1", arm: "v8.0",
			experiment: "logosarena", arena: true, module: true,
		},
		{
			name: "fileLevelsWinWhenEnvEmpty", modRoot: baseline,
			flags: Flags{Logos: true}, optLevel: "2",
			amd64: "v1", arm: "v8.0",
			experiment: "logosarena", arena: true, module: true,
		},
		{
			name: "envBeatsFileDeterministic", modRoot: det,
			flags: Flags{Logos: true, Env: Env{GOAMD64: "v4"}}, optLevel: "2",
			amd64: "", arm: "v8.2",
			experiment: "logosarena", arena: true, module: true,
		},
		{
			name: "envBeatsFileNative", modRoot: native,
			flags: Flags{Logos: true, Env: Env{GOAMD64: "v2", GOARM64: "v8.3"}}, optLevel: "2",
			amd64: "", arm: "",
			experiment: "logosarena", arena: true, module: true,
		},
		{
			name: "brokenFileFallbacks", modRoot: broken,
			flags: Flags{Logos: true}, optLevel: "2",
			amd64: "v3", arm: "v8.2", arena: false, module: true,
		},
		{
			name: "invalidOptText", modRoot: empty,
			flags: Flags{Logos: true, Opt: "bad"}, optLevel: "2",
			amd64: "v3", arm: "v8.2",
			experiment: "logosarena", arena: true, module: true,
			wantErr: `logos build: invalid -O level "bad" (want 0, 1, 2, 3, s)`,
		},
		{
			name: "invalidOptPlainGoText", modRoot: empty,
			flags: Flags{Opt: "bad"}, optLevel: "1",
			wantErr: `go build: invalid -O level "bad" (want 0, 1, 2, 3, s)`,
		},
		{
			name: "arenaComposesWithEnv", modRoot: arenaOnFile,
			flags: Flags{Logos: true, Env: Env{GOEXPERIMENT: "nojsonv2"}}, optLevel: "2",
			amd64: "v3", arm: "v8.2",
			experiment: "nojsonv2,logosarena", arena: true, module: true,
		},
		{
			name: "arenaAbsentKeepsEnv", modRoot: empty,
			flags: Flags{Logos: true, Env: Env{GOEXPERIMENT: "jsonv2"}}, optLevel: "2",
			amd64: "v3", arm: "v8.2",
			experiment: "jsonv2,logosarena", arena: true, module: true,
		},
		{
			name: "arenaWithEmptyEnv", modRoot: arenaOnFile,
			flags: Flags{Logos: true}, optLevel: "2",
			amd64: "v3", arm: "v8.2",
			experiment: "logosarena", arena: true, module: true,
		},
		{
			name: "pedanticOn", modRoot: empty,
			flags:    Flags{Logos: true, Pedantic: true, PedanticSet: true},
			optLevel: "2",
			amd64:    "v3", arm: "v8.2",
			experiment: "logosarena", arena: true,
			pedantic: config.PedanticOn, module: true,
		},
		{
			name: "pedanticOff", modRoot: empty,
			flags:    Flags{Logos: true, PedanticSet: true},
			optLevel: "2",
			amd64:    "v3", arm: "v8.2",
			experiment: "logosarena", arena: true,
			pedantic: config.PedanticOff, module: true,
		},
		{
			name: "pedanticUnset", modRoot: empty,
			flags:      Flags{Logos: true},
			optLevel:   "2",
			experiment: "logosarena", arena: true,
			amd64: "v3", arm: "v8.2",
			pedantic: config.PedanticDefault, module: true,
		},
		{
			name: "flagFieldsCopied", modRoot: empty,
			flags:    Flags{Logos: true, JSON: true, WasmLoader: true, LintCmd: true, WasmFeatures: "stack-switching"},
			optLevel: "2",
			amd64:    "v3", arm: "v8.2",
			experiment: "logosarena", arena: true, module: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s, err := Load(tt.modRoot, tt.flags)
			if s == nil {
				t.Fatal("Load returned a nil Settings")
			}
			if tt.wantErr != "" {
				if err == nil || err.Error() != tt.wantErr {
					t.Fatalf("Load error = %v, want %q", err, tt.wantErr)
				}
			} else if err != nil {
				t.Fatalf("Load error = %v, want nil", err)
			}
			if s.OptLevel != tt.optLevel {
				t.Errorf("OptLevel = %q, want %q", s.OptLevel, tt.optLevel)
			}
			if s.GOAMD64 != tt.amd64 || s.GOARM64 != tt.arm {
				t.Errorf("arch levels = (%q, %q), want (%q, %q)", s.GOAMD64, s.GOARM64, tt.amd64, tt.arm)
			}
			if s.Experiment != tt.experiment {
				t.Errorf("Experiment = %q, want %q", s.Experiment, tt.experiment)
			}
			if s.Arena != tt.arena {
				t.Errorf("Arena = %v, want %v", s.Arena, tt.arena)
			}
			if s.Pedantic != tt.pedantic {
				t.Errorf("Pedantic = %v, want %v", s.Pedantic, tt.pedantic)
			}
			if (s.Module != nil) != tt.module {
				t.Errorf("Module = %v, want present %v", s.Module, tt.module)
			}
			if tt.flags.Logos != s.Logos {
				t.Errorf("Logos = %v, want %v", s.Logos, tt.flags.Logos)
			}
			if s.JSON != tt.flags.JSON || s.WasmLoader != tt.flags.WasmLoader || s.LintCmd != tt.flags.LintCmd || s.WasmFeatures != tt.flags.WasmFeatures {
				t.Errorf("flag fields = (%v, %v, %v, %q), want (%v, %v, %v, %q)",
					s.JSON, s.WasmLoader, s.LintCmd, s.WasmFeatures,
					tt.flags.JSON, tt.flags.WasmLoader, tt.flags.LintCmd, tt.flags.WasmFeatures)
			}
		})
	}

	t.Run("brokenFileIsNotALoadError", func(t *testing.T) {
		s, err := Load(broken, Flags{Logos: true})
		if err != nil {
			t.Fatalf("broken logos.toml is not a Load error, got %v", err)
		}
		if s.Module == nil || s.Module.Err == nil {
			t.Fatalf("Module.Err = %v, want the parse error", s.Module)
		}
		if got := s.LintPolicy(s.Module); got != nil {
			t.Errorf("LintPolicy = %q, want nil on a broken file", got)
		}
	})

	t.Run("lintPolicyCarriesPedantic", func(t *testing.T) {
		root := logosModule(t, "[lints]\npedantic = \"allow\"\n")
		s, err := Load(root, Flags{Logos: true, Pedantic: true, PedanticSet: true})
		if err != nil {
			t.Fatalf("Load error = %v", err)
		}
		policy := s.LintPolicy(s.Module)
		if policy == nil {
			t.Fatal("LintPolicy = nil, want the policy JSON")
		}
		var got struct {
			Version int               `json:"version"`
			Groups  map[string]string `json:"groups"`
		}
		if err := json.Unmarshal(policy, &got); err != nil {
			t.Fatalf("unmarshal policy %q: %v", policy, err)
		}
		if got.Version != config.LintPolicyVersion {
			t.Errorf("policy version = %d, want %d", got.Version, config.LintPolicyVersion)
		}
		if got.Groups["pedantic"] != "warn" {
			t.Errorf("policy pedantic = %q, want warn under --pedantic", got.Groups["pedantic"])
		}
	})

	t.Run("lintPolicyPedanticOffForcesAllow", func(t *testing.T) {
		root := logosModule(t, "[lints]\npedantic = \"warn\"\n")
		s, err := Load(root, Flags{Logos: true, PedanticSet: true})
		if err != nil {
			t.Fatalf("Load error = %v", err)
		}
		policy := s.LintPolicy(s.Module)
		if policy == nil {
			t.Fatal("LintPolicy = nil, want the policy JSON")
		}
		var got struct {
			Groups map[string]string `json:"groups"`
		}
		if err := json.Unmarshal(policy, &got); err != nil {
			t.Fatalf("unmarshal policy %q: %v", policy, err)
		}
		if got.Groups["pedantic"] != "allow" {
			t.Errorf("policy pedantic = %q, want allow under --pedantic=false", got.Groups["pedantic"])
		}
	})
}

func TestCurrentZeroAndPublish(t *testing.T) {
	cur := Current()
	if cur == nil {
		t.Fatal("Current = nil before SetCurrent, want the zero Settings")
	}
	if cur.OptLevel != "" || cur.Module != nil || cur.Arena || cur.Cgo.Changed {
		t.Errorf("Current before SetCurrent = %+v, want zero values", cur)
	}
	s := &Settings{OptLevel: "3"}
	SetCurrent(s)
	if Current() != s {
		t.Fatal("Current did not return the published settings")
	}
	SetCurrent(&Settings{})
}

func TestTargetArchEnv(t *testing.T) {
	tests := []struct {
		name       string
		amd64, arm string
		goarch     string
		want       []string
	}{
		{"amd64PinsV3", "v3", "v8.2", "amd64", []string{"GOAMD64=v3"}},
		{"arm64PinsV82", "v3", "v8.2", "arm64", []string{"GOARM64=v8.2"}},
		{"wasmPinsNothing", "v3", "v8.2", "wasm", nil},
		{"unsetContributesNothing", "", "", "amd64", nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := &Settings{GOAMD64: tt.amd64, GOARM64: tt.arm}
			got := s.TargetArchEnv(tt.goarch)
			if strings.Join(got, ",") != strings.Join(tt.want, ",") {
				t.Errorf("TargetArchEnv(%q) = %q, want %q", tt.goarch, got, tt.want)
			}
		})
	}
}

func TestPedanticFlag(t *testing.T) {
	t.Run("recordsValueAndSet", func(t *testing.T) {
		f := &Flags{}
		v := PedanticFlag(f)
		if err := v.Set("true"); err != nil {
			t.Fatalf("Set(\"true\") error = %v", err)
		}
		if !f.Pedantic || !f.PedanticSet {
			t.Errorf("after --pedantic: Pedantic=%v PedanticSet=%v, want true true", f.Pedantic, f.PedanticSet)
		}
		if v.String() != "true" {
			t.Errorf("String = %q, want true", v.String())
		}
	})
	t.Run("falseStillSets", func(t *testing.T) {
		f := &Flags{}
		v := PedanticFlag(f)
		if err := v.Set("false"); err != nil {
			t.Fatalf("Set(\"false\") error = %v", err)
		}
		if f.Pedantic || !f.PedanticSet {
			t.Errorf("after --pedantic=false: Pedantic=%v PedanticSet=%v, want false true", f.Pedantic, f.PedanticSet)
		}
	})
	t.Run("rejectsNonBoolean", func(t *testing.T) {
		f := &Flags{}
		v := PedanticFlag(f)
		if err := v.Set("maybe"); err == nil {
			t.Error("Set(\"maybe\") = nil error, want a parse error")
		}
	})
	t.Run("isBoolFlag", func(t *testing.T) {
		type boolFlag interface {
			IsBoolFlag() bool
		}
		v := PedanticFlag(&Flags{})
		bf, ok := v.(boolFlag)
		if !ok || !bf.IsBoolFlag() {
			t.Error("PedanticFlag value does not behave as a bool flag")
		}
	})
}

func TestWasmFeaturesFlag(t *testing.T) {
	t.Run("acceptsKnownFeature", func(t *testing.T) {
		f := &Flags{}
		v := WasmFeaturesFlag(f)
		if err := v.Set("stack-switching"); err != nil {
			t.Fatalf("Set error = %v", err)
		}
		if f.WasmFeatures != "stack-switching" || v.String() != "stack-switching" {
			t.Errorf("WasmFeatures = %q String = %q, want stack-switching", f.WasmFeatures, v.String())
		}
	})
	t.Run("rejectsUnknownFeature", func(t *testing.T) {
		f := &Flags{}
		v := WasmFeaturesFlag(f)
		err := v.Set("no-such-feature")
		if err == nil {
			t.Fatal("Set(\"no-such-feature\") = nil error, want rejection")
		}
		const want = `unsupported wasm feature "no-such-feature" (want stack-switching)`
		if err.Error() != want {
			t.Errorf("Set error = %q, want %q", err.Error(), want)
		}
		if f.WasmFeatures != "" {
			t.Errorf("WasmFeatures = %q after a rejected value, want empty", f.WasmFeatures)
		}
	})
}

func TestCWDModuleRoot(t *testing.T) {
	t.Run("findsGoModFile", func(t *testing.T) {
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module m\n"), 0o666); err != nil {
			t.Fatal(err)
		}
		sub := filepath.Join(dir, "pkg")
		if err := os.MkdirAll(sub, 0o777); err != nil {
			t.Fatal(err)
		}
		t.Chdir(sub)
		if got := CWDModuleRoot(); got != dir {
			t.Errorf("CWDModuleRoot = %q, want %q", got, dir)
		}
	})
	t.Run("goModDirectoryIsNotARoot", func(t *testing.T) {
		dir := t.TempDir()
		if err := os.MkdirAll(filepath.Join(dir, "go.mod"), 0o777); err != nil {
			t.Fatal(err)
		}
		t.Chdir(dir)
		if got := CWDModuleRoot(); got != "" {
			t.Errorf("CWDModuleRoot = %q, want empty above a go.mod directory", got)
		}
	})
	t.Run("none", func(t *testing.T) {
		t.Chdir(t.TempDir())
		if got := CWDModuleRoot(); got != "" {
			t.Errorf("CWDModuleRoot = %q, want empty with no go.mod", got)
		}
	})
}

func TestLoadRoots(t *testing.T) {
	allowA := logosModule(t, "[build]\ncgo = \"allow\"\n")
	logosSrc(t, allowA)
	forbidA := logosModule(t, "[build]\ncgo = \"forbid\"\n")
	logosSrc(t, forbidA)
	logosNoToml := logosModule(t, "")
	logosSrc(t, logosNoToml)
	goModule := logosModule(t, "")
	arenaOnDir := logosModule(t, "[build]\narena = true\n")
	arenaOffDir := logosModule(t, "[build]\narena = false\n")

	tests := []struct {
		name             string
		modRoot          string
		roots            []string
		env              Env
		enabled, opinion bool
		changed          bool
		arena            bool
	}{
		{
			name: "cgoAllowBeatsEnv", modRoot: allowA, env: Env{CGO_ENABLED: "0"},
			enabled: true, opinion: true, changed: true,
			arena: true,
		},
		{
			name: "cgoForbidFailClosed", modRoot: forbidA,
			opinion: true,
			arena:   true,
		},
		{
			name: "noKeyEnvOnDecides", modRoot: allowA, env: Env{CGO_ENABLED: "1"},
			enabled: true, opinion: true, changed: true,
			arena: true,
		},
		{
			name: "noKeyEnvOffDecides", modRoot: forbidA, env: Env{CGO_ENABLED: "0"},
			opinion: true, changed: true,
			arena: true,
		},
		{
			name: "noKeyNoEnvLogosModuleDisables", modRoot: logosNoToml,
			opinion: true,
			arena:   true,
		},
		{
			name: "goModuleNoOpinion", modRoot: "", roots: []string{goModule},
			arena: true,
		},
		{
			name: "cgoOresAcrossRoots", modRoot: forbidA, roots: []string{forbidA, allowA},
			enabled: true, opinion: true,
			arena: true,
		},
		{
			name: "arenaWhenAnyRootHasIt", modRoot: arenaOffDir, roots: []string{arenaOffDir, arenaOnDir},
			arena: true,
		},
		{
			name: "arenaOffEverywhere", modRoot: arenaOffDir, roots: []string{arenaOffDir},
		},
		{
			name: "rootsOverrideModRoot", modRoot: arenaOnDir, roots: []string{arenaOffDir},
		},
		{
			name:  "noRootsNoModule",
			arena: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s, err := Load(tt.modRoot, Flags{Logos: true, Env: tt.env, Roots: tt.roots})
			if err != nil {
				t.Fatalf("Load error = %v", err)
			}
			if s.Cgo.Enabled != tt.enabled || s.Cgo.Opinion != tt.opinion || s.Cgo.Changed != tt.changed {
				t.Errorf("Cgo = %+v, want enabled=%v opinion=%v changed=%v", s.Cgo, tt.enabled, tt.opinion, tt.changed)
			}
			if s.Arena != tt.arena {
				t.Errorf("Arena = %v, want %v", s.Arena, tt.arena)
			}
		})
	}
}
func TestLoadBuildTablesApplyToGoModule(t *testing.T) {
	root := logosModule(t, "[build]\narena = true\ncgo = \"allow\"\ntarget_cpu = \"baseline\"\nopt_level = \"O3\"\n[lints]\nslop = \"warn\"\n")
	s, err := Load(root, Flags{Logos: true})
	if err != nil {
		t.Fatalf("Load error = %v", err)
	}
	if s.OptLevel != "3" {
		t.Errorf("OptLevel = %q, want %q for a Go module root", s.OptLevel, "3")
	}
	if s.GOAMD64 != "v1" || s.GOARM64 != "v8.0" {
		t.Errorf("arch levels = (%q, %q), want (v1, v8.0) for a Go module root", s.GOAMD64, s.GOARM64)
	}
	if !s.Arena || s.Experiment != "logosarena" {
		t.Errorf("Arena = %v Experiment = %q, want true logosarena for a Go module root", s.Arena, s.Experiment)
	}
	if s.Cgo.Opinion {
		t.Errorf("Cgo.Opinion = true, want no cgo opinion in a Go module")
	}
	policy := s.LintPolicy(s.Module)
	if policy == nil {
		t.Fatal("LintPolicy = nil, want the lints table for a Go module root")
	}
	var got struct {
		Groups map[string]string `json:"groups"`
	}
	if err := json.Unmarshal(policy, &got); err != nil {
		t.Fatalf("unmarshal policy %q: %v", policy, err)
	}
	if got.Groups["slop"] != "warn" {
		t.Errorf("policy slop = %q, want warn for a Go module root", got.Groups["slop"])
	}
}
