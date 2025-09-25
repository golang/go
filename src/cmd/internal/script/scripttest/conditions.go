// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package scripttest

import (
	"cmd/internal/script"
	"fmt"
	"internal/buildcfg"
	"internal/platform"
	"internal/testenv"
	"runtime"
	"strings"
	"testing"
)

// AddToolChainScriptConditions accepts a [script.Cond] map and adds into it a
// set of commonly used conditions for doing toolchains testing,
// including whether the platform supports cgo, a buildmode condition,
// support for GOEXPERIMENT testing, etc. Callers must also pass in
// current GOHOSTOOS/GOHOSTARCH settings, since some of the conditions
// introduced can be influenced by them.
func AddToolChainScriptConditions(t *testing.T, conds map[string]script.Cond, goHostOS, goHostArch string) {
	add := func(name string, cond script.Cond) {
		if _, ok := conds[name]; ok {
			t.Fatalf("condition %q is already registered", name)
		}
		conds[name] = cond
	}

	lazyBool := func(summary string, f func() bool) script.Cond {
		return script.OnceCondition(summary, func() (bool, error) { return f(), nil })
	}

	add("asan", sysCondition("-asan", platform.ASanSupported, true, goHostOS, goHostArch))
	add("buildmode", script.PrefixCondition("go supports -buildmode=<suffix>", hasBuildmode))
	add("cgo", script.BoolCondition("host CGO_ENABLED", testenv.HasCGO()))
	add("cgolinkext", script.Condition("platform requires external linking for cgo", cgoLinkExt))
	add("cross", script.BoolCondition("cmd/go GOOS/GOARCH != GOHOSTOS/GOHOSTARCH", goHostOS != runtime.GOOS || goHostArch != runtime.GOARCH))
	add("fuzz", sysCondition("-fuzz", platform.FuzzSupported, false, goHostOS, goHostArch))
	add("fuzz-instrumented", sysCondition("-fuzz with instrumentation", platform.FuzzInstrumented, false, goHostOS, goHostArch))
	add("GODEBUG", script.PrefixCondition("GODEBUG contains <suffix>", hasGodebug))
	add("GOEXPERIMENT", script.PrefixCondition("GOEXPERIMENT <suffix> is enabled", hasGoexperiment))
	add("go-builder", script.BoolCondition("GO_BUILDER_NAME is non-empty", testenv.Builder() != ""))
	add("link", lazyBool("testenv.HasLink()", testenv.HasLink))
	add("msan", sysCondition("-msan", platform.MSanSupported, true, goHostOS, goHostArch))
	add("mustlinkext", script.Condition("platform always requires external linking", mustLinkExt))
	add("pielinkext", script.Condition("platform requires external linking for PIE", pieLinkExt))
	add("race", sysCondition("-race", platform.RaceDetectorSupported, true, goHostOS, goHostArch))
	add("symlink", lazyBool("testenv.HasSymlink()", testenv.HasSymlink))
}

func sysCondition(flag string, f func(goos, goarch string) bool, needsCgo bool, goHostOS, goHostArch string) script.Cond {
	return script.Condition(
		"GOOS/GOARCH supports "+flag,
		func(s *script.State) (bool, error) {
			GOOS, _ := s.LookupEnv("GOOS")
			GOARCH, _ := s.LookupEnv("GOARCH")
			cross := goHostOS != GOOS || goHostArch != GOARCH
			return (!needsCgo || (testenv.HasCGO() && !cross)) && f(GOOS, GOARCH), nil
		})
}

func hasBuildmode(s *script.State, mode string) (bool, error) {
	GOOS, _ := s.LookupEnv("GOOS")
	GOARCH, _ := s.LookupEnv("GOARCH")
	return platform.BuildModeSupported(runtime.Compiler, mode, GOOS, GOARCH), nil
}

func cgoLinkExt(s *script.State) (bool, error) {
	GOOS, _ := s.LookupEnv("GOOS")
	GOARCH, _ := s.LookupEnv("GOARCH")
	return platform.MustLinkExternal(GOOS, GOARCH, true), nil
}

func mustLinkExt(s *script.State) (bool, error) {
	GOOS, _ := s.LookupEnv("GOOS")
	GOARCH, _ := s.LookupEnv("GOARCH")
	return platform.MustLinkExternal(GOOS, GOARCH, false), nil
}

func pieLinkExt(s *script.State) (bool, error) {
	GOOS, _ := s.LookupEnv("GOOS")
	GOARCH, _ := s.LookupEnv("GOARCH")
	return !platform.InternalLinkPIESupported(GOOS, GOARCH), nil
}

func hasGodebug(s *script.State, value string) (bool, error) {
	godebug, _ := s.LookupEnv("GODEBUG")
	for p := range strings.SplitSeq(godebug, ",") {
		if strings.TrimSpace(p) == value {
			return true, nil
		}
	}
	return false, nil
}

func hasGoexperiment(s *script.State, value string) (bool, error) {
	GOOS, _ := s.LookupEnv("GOOS")
	GOARCH, _ := s.LookupEnv("GOARCH")
	goexp, _ := s.LookupEnv("GOEXPERIMENT")
	flags, err := buildcfg.ParseGOEXPERIMENT(GOOS, GOARCH, goexp)
	if err != nil {
		return false, err
	}
	for _, exp := range flags.All() {
		if value == exp {
			return true, nil
		}
		if strings.TrimPrefix(value, "no") == strings.TrimPrefix(exp, "no") {
			return false, nil
		}
	}
	return false, fmt.Errorf("unrecognized GOEXPERIMENT %q", value)
}
