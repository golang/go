// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main_test

import (
	"cmd/go/internal/cfg"
	"cmd/go/internal/script"
	"cmd/go/internal/script/scripttest"
	"errors"
	"fmt"
	"internal/buildcfg"
	"internal/platform"
	"internal/testenv"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"strings"
	"sync"
)

func scriptConditions() map[string]script.Cond {
	conds := scripttest.DefaultConds()

	add := func(name string, cond script.Cond) {
		if _, ok := conds[name]; ok {
			panic(fmt.Sprintf("condition %q is already registered", name))
		}
		conds[name] = cond
	}

	lazyBool := func(summary string, f func() bool) script.Cond {
		return script.OnceCondition(summary, func() (bool, error) { return f(), nil })
	}

	add("abscc", script.Condition("default $CC path is absolute and exists", defaultCCIsAbsolute))
	add("asan", sysCondition("-asan", platform.ASanSupported, true))
	add("buildmode", script.PrefixCondition("go supports -buildmode=<suffix>", hasBuildmode))
	add("case-sensitive", script.OnceCondition("$WORK filesystem is case-sensitive", isCaseSensitive))
	add("cc", script.PrefixCondition("go env CC = <suffix> (ignoring the go/env file)", ccIs))
	add("cgo", script.BoolCondition("host CGO_ENABLED", testenv.HasCGO()))
	add("cgolinkext", script.Condition("platform requires external linking for cgo", cgoLinkExt))
	add("cross", script.BoolCondition("cmd/go GOOS/GOARCH != GOHOSTOS/GOHOSTARCH", goHostOS != runtime.GOOS || goHostArch != runtime.GOARCH))
	add("fuzz", sysCondition("-fuzz", platform.FuzzSupported, false))
	add("fuzz-instrumented", sysCondition("-fuzz with instrumentation", platform.FuzzInstrumented, false))
	add("git", lazyBool("the 'git' executable exists and provides the standard CLI", hasWorkingGit))
	add("GODEBUG", script.PrefixCondition("GODEBUG contains <suffix>", hasGodebug))
	add("GOEXPERIMENT", script.PrefixCondition("GOEXPERIMENT <suffix> is enabled", hasGoexperiment))
	add("go-builder", script.BoolCondition("GO_BUILDER_NAME is non-empty", testenv.Builder() != ""))
	add("link", lazyBool("testenv.HasLink()", testenv.HasLink))
	add("mismatched-goroot", script.Condition("test's GOROOT_FINAL does not match the real GOROOT", isMismatchedGoroot))
	add("msan", sysCondition("-msan", platform.MSanSupported, true))
	add("mustlinkext", script.Condition("platform always requires external linking", mustLinkExt))
	add("net", script.PrefixCondition("can connect to external network host <suffix>", hasNet))
	add("pielinkext", script.Condition("platform requires external linking for PIE", pieLinkExt))
	add("race", sysCondition("-race", platform.RaceDetectorSupported, true))
	add("symlink", lazyBool("testenv.HasSymlink()", testenv.HasSymlink))
	add("trimpath", script.OnceCondition("test binary was built with -trimpath", isTrimpath))

	return conds
}

func defaultCCIsAbsolute(s *script.State) (bool, error) {
	GOOS, _ := s.LookupEnv("GOOS")
	GOARCH, _ := s.LookupEnv("GOARCH")
	defaultCC := cfg.DefaultCC(GOOS, GOARCH)
	if filepath.IsAbs(defaultCC) {
		if _, err := exec.LookPath(defaultCC); err == nil {
			return true, nil
		}
	}
	return false, nil
}

func ccIs(s *script.State, want string) (bool, error) {
	CC, _ := s.LookupEnv("CC")
	if CC != "" {
		return CC == want, nil
	}
	GOOS, _ := s.LookupEnv("GOOS")
	GOARCH, _ := s.LookupEnv("GOARCH")
	return cfg.DefaultCC(GOOS, GOARCH) == want, nil
}

func isMismatchedGoroot(s *script.State) (bool, error) {
	gorootFinal, _ := s.LookupEnv("GOROOT_FINAL")
	if gorootFinal == "" {
		gorootFinal, _ = s.LookupEnv("GOROOT")
	}
	return gorootFinal != testGOROOT, nil
}

func sysCondition(flag string, f func(goos, goarch string) bool, needsCgo bool) script.Cond {
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

var scriptNetEnabled sync.Map // testing.TB → already enabled

func hasNet(s *script.State, host string) (bool, error) {
	if !testenv.HasExternalNetwork() {
		return false, nil
	}

	// TODO(bcmills): Add a flag or environment variable to allow skipping tests
	// for specific hosts and/or skipping all net tests except for specific hosts.

	t, ok := tbFromContext(s.Context())
	if !ok {
		return false, errors.New("script Context unexpectedly missing testing.TB key")
	}

	if netTestSem != nil {
		// When the number of external network connections is limited, we limit the
		// number of net tests that can run concurrently so that the overall number
		// of network connections won't exceed the limit.
		_, dup := scriptNetEnabled.LoadOrStore(t, true)
		if !dup {
			// Acquire a net token for this test until the test completes.
			netTestSem <- struct{}{}
			t.Cleanup(func() {
				<-netTestSem
				scriptNetEnabled.Delete(t)
			})
		}
	}

	// Since we have confirmed that the network is available,
	// allow cmd/go to use it.
	s.Setenv("TESTGONETWORK", "")
	return true, nil
}

func hasGodebug(s *script.State, value string) (bool, error) {
	godebug, _ := s.LookupEnv("GODEBUG")
	for _, p := range strings.Split(godebug, ",") {
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

func isCaseSensitive() (bool, error) {
	tmpdir, err := os.MkdirTemp(testTmpDir, "case-sensitive")
	if err != nil {
		return false, fmt.Errorf("failed to create directory to determine case-sensitivity: %w", err)
	}
	defer os.RemoveAll(tmpdir)

	fcap := filepath.Join(tmpdir, "FILE")
	if err := os.WriteFile(fcap, []byte{}, 0644); err != nil {
		return false, fmt.Errorf("error writing file to determine case-sensitivity: %w", err)
	}

	flow := filepath.Join(tmpdir, "file")
	_, err = os.ReadFile(flow)
	switch {
	case err == nil:
		return false, nil
	case os.IsNotExist(err):
		return true, nil
	default:
		return false, fmt.Errorf("unexpected error reading file when determining case-sensitivity: %w", err)
	}
}

func isTrimpath() (bool, error) {
	info, _ := debug.ReadBuildInfo()
	if info == nil {
		return false, errors.New("missing build info")
	}

	for _, s := range info.Settings {
		if s.Key == "-trimpath" && s.Value == "true" {
			return true, nil
		}
	}
	return false, nil
}

func hasWorkingGit() bool {
	if runtime.GOOS == "plan9" {
		// The Git command is usually not the real Git on Plan 9.
		// See https://golang.org/issues/29640.
		return false
	}
	_, err := exec.LookPath("git")
	return err == nil
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
