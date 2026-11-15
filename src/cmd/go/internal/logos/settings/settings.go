// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package settings

import (
	"cmp"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync/atomic"

	"cmd/go/internal/logos/config"
)

// Env is the user environment observed at process start (cfg takes it
// before main.invoke injects MkEnv defaults, covering the GOENV file).
type Env struct{ GOEXPERIMENT, CGO_ENABLED, GOAMD64, GOARM64 string }

// Flags is every input of Load that is not a logos.toml.
type Flags struct {
	Logos        bool
	Env          Env
	Roots        []string
	Opt          string
	Pedantic     bool
	PedanticSet  bool
	JSON         bool
	WasmLoader   bool
	WasmFeatures string
	LintCmd      bool
}

// CommandLine is the storage the flag-registration hook lines bind to.
var CommandLine Flags

// Cgo is the build-wide cgo decision: Enabled and Opinion OR across the
// main module roots, and Changed tracks an explicit user CGO_ENABLED.
type Cgo struct{ Enabled, Opinion, Changed bool }

// Settings is the resolved logos build state: every decision Load derives
// from the flags, the environment, and the main module roots' logos.toml.
type Settings struct {
	Logos                     bool
	Module                    *Module
	OptLevel                  string
	Gcflags                   []string
	GOAMD64, GOARM64          string
	Cgo                       Cgo
	Arena                     bool
	Experiment                string
	Pedantic                  config.PedanticMode
	JSON, WasmLoader, LintCmd bool
	WasmFeatures              string
}

// Load resolves the settings: the -O level from the flag first, the
// logos.toml of each main module root second. Plain go reads no
// logos.toml and keeps the upstream opt level. Its only error is an
// invalid -O value, and it still returns a non-nil Settings whose
// file-derived fields are valid, so the caller applies the environment
// before reporting the error. A broken logos.toml is not an error of
// Load: it is Settings.Module.Err, and every field takes the broken-file
// fallbacks.
func Load(modRoot string, flags Flags) (*Settings, error) {
	s := &Settings{
		Logos:        flags.Logos,
		JSON:         flags.JSON,
		WasmLoader:   flags.WasmLoader,
		LintCmd:      flags.LintCmd,
		WasmFeatures: flags.WasmFeatures,
		Pedantic:     pedanticMode(flags),
	}
	var err error
	level := ""
	if flags.Opt != "" {
		if validOptLevel(flags.Opt) {
			level = flags.Opt
		} else {
			prog := "go"
			if flags.Logos {
				prog = "logos"
			}
			err = fmt.Errorf("%s build: invalid -O level %q (want 0, 1, 2, 3, s)", prog, flags.Opt)
		}
	}
	fallback := "1"
	var m *Module
	var f *config.File
	if flags.Logos {
		m = ModuleAt(modRoot)
		f = m.File
		if f == nil {
			f = ModuleAt("").File
		}
		fallback = strings.TrimPrefix(f.Build.OptLevel, "O")
	}
	s.OptLevel = cmp.Or(level, fallback)
	s.Gcflags = flagsFor(s.OptLevel)
	if !flags.Logos {
		s.Experiment = flags.Env.GOEXPERIMENT
		return s, err
	}
	s.Module = m
	s.GOAMD64, s.GOARM64 = archLevels(f, flags.Env)
	roots := flags.Roots
	if len(roots) == 0 && modRoot != "" {
		roots = []string{modRoot}
	}
	s.Cgo = cgoDecision(roots, flags.Env.CGO_ENABLED)
	s.Arena = anyRootArena(roots)
	if s.Arena {
		s.Experiment = WithArena(flags.Env.GOEXPERIMENT)
	} else {
		s.Experiment = flags.Env.GOEXPERIMENT
	}
	return s, err
}

var zeroSettings Settings

var current atomic.Pointer[Settings]

// Current returns the settings published by SetCurrent, or the zero
// Settings before it.
func Current() *Settings {
	if s := current.Load(); s != nil {
		return s
	}
	return &zeroSettings
}

// SetCurrent publishes the process-wide settings; work BuildInit calls it
// once.
func SetCurrent(s *Settings) {
	current.Store(s)
}

// CWDModuleRoot reports the module root above the working directory by
// walking up to the nearest go.mod file, or "" when there is none. It
// runs after the -C chdir, so -C targets resolve to the target module.
func CWDModuleRoot() string {
	wd, err := os.Getwd()
	if err != nil {
		return ""
	}
	dir := wd
	for {
		if info, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil && !info.IsDir() {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return ""
		}
		dir = parent
	}
}

// LintPolicy returns the -logos.cfg policy bytes of m in this build's
// pedantic state: nil on a broken file or a marshal error.
func (s *Settings) LintPolicy(m *Module) []byte {
	if m == nil || m.Err != nil || m.File == nil {
		return nil
	}
	policy, err := m.File.Lints.PolicyJSON(s.Pedantic)
	if err != nil {
		return nil
	}
	return policy
}

// TargetArchEnv returns the KEY=VALUE pairs of the architecture levels
// this settings pins, for test binary environments: an unset level and a
// level of another architecture contribute nothing.
func (s *Settings) TargetArchEnv(goarch string) []string {
	prefix := "GO" + strings.ToUpper(goarch) + "="
	var env []string
	for _, level := range []struct{ key, val string }{{"GOAMD64", s.GOAMD64}, {"GOARM64", s.GOARM64}} {
		if level.val == "" {
			continue
		}
		if entry := level.key + "=" + level.val; strings.HasPrefix(entry, prefix) {
			env = append(env, entry)
		}
	}
	return env
}

// PedanticFlag returns the --pedantic flag value storing into f: a
// boolean that also records that it was given, so --pedantic=false
// overrides a logos.toml pedantic key and an absent flag does not.
func PedanticFlag(f *Flags) flag.Value {
	return pedanticFlag{f}
}

// WasmFeaturesFlag returns the --wasm-features flag value storing into f:
// a comma-separated list validated against the supported features where
// the flag is parsed.
func WasmFeaturesFlag(f *Flags) flag.Value {
	return wasmFeaturesFlag{f}
}

var wasmFeatureNames = []string{"stack-switching"}

type pedanticFlag struct{ f *Flags }

func (p pedanticFlag) IsBoolFlag() bool { return true }

func (p pedanticFlag) String() string { return strconv.FormatBool(p.f.Pedantic) }

func (p pedanticFlag) Set(s string) error {
	v, err := strconv.ParseBool(s)
	if err != nil {
		return err
	}
	p.f.Pedantic, p.f.PedanticSet = v, true
	return nil
}

type wasmFeaturesFlag struct{ f *Flags }

func (w wasmFeaturesFlag) String() string { return w.f.WasmFeatures }

func (w wasmFeaturesFlag) Set(s string) error {
	for f := range strings.SplitSeq(s, ",") {
		if !slices.Contains(wasmFeatureNames, f) {
			return fmt.Errorf("unsupported wasm feature %q (want %s)", f, strings.Join(wasmFeatureNames, ", "))
		}
	}
	w.f.WasmFeatures = s
	return nil
}

func pedanticMode(flags Flags) config.PedanticMode {
	switch {
	case !flags.PedanticSet:
		return config.PedanticDefault
	case flags.Pedantic:
		return config.PedanticOn
	}
	return config.PedanticOff
}

func archLevels(f *config.File, env Env) (amd64, arm64 string) {
	amd64, arm64 = config.ArchLevels(f.Build.TargetCPU)
	if f.Build.TargetCPU == "baseline" {
		return amd64, arm64
	}
	if env.GOAMD64 != "" {
		amd64 = ""
	}
	if env.GOARM64 != "" {
		arm64 = ""
	}
	return amd64, arm64
}

func cgoDecision(roots []string, cgoEnv string) (c Cgo) {
	c.Changed = cgoEnv == "0" || cgoEnv == "1"
	for _, root := range roots {
		enabled, opinion := ModuleAt(root).CgoOpinion(cgoEnv)
		if opinion {
			c.Opinion = true
			c.Enabled = c.Enabled || enabled
		}
	}
	return c
}

func anyRootArena(roots []string) bool {
	if len(roots) == 0 {
		return true
	}
	for _, root := range roots {
		m := ModuleAt(root)
		if m.Err == nil && (m.File == nil || m.File.Build.Arena == nil || *m.File.Build.Arena) {
			return true
		}
	}
	return false
}

var optLevelFlags = map[string][]string{
	"0": {"-N", "-l", "-d=variablemakethreshold=32,variablemakeundern=1"},
	"1": nil,
	"2": {"-d=inlbudget=120,variablemakethreshold=256"},
	"3": {"-d=inlbudget=400,inlrecursivebudget=80,variablemakethreshold=4096"},
	"s": {"-d=inlbudget=15"},
}

func validOptLevel(level string) bool {
	_, ok := optLevelFlags[level]
	return ok
}

func flagsFor(level string) []string {
	return optLevelFlags[level]
}
