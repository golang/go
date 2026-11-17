// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main_test

import (
	"cmd/go/internal/cfg"
	"cmd/go/internal/logos/config"
	"cmd/internal/script"
	"cmd/internal/script/scripttest"
	"errors"
	"fmt"
	"internal/buildcfg"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"runtime/debug"
	"strings"
	"testing"

	"golang.org/x/mod/semver"
)

func scriptConditions(t *testing.T) map[string]script.Cond {
	conds := scripttest.DefaultConds()

	scripttest.AddToolChainScriptConditions(t, conds, goHostOS, goHostArch)

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
	add("case-sensitive", script.OnceCondition("$WORK filesystem is case-sensitive", isCaseSensitive))
	add("cc", script.PrefixCondition("go env CC = <suffix> (ignoring the go/env file)", ccIs))
	add("git", lazyBool("the 'git' executable exists and provides the standard CLI", hasWorkingGit))
	add("git-sha256", script.OnceCondition("the local 'git' version is recent enough to support sha256 object/commit hashes", gitSupportsSHA256))
	add("host-goamd64", script.CachedCondition("host supports GOAMD64=<suffix>", func(level string) (bool, error) {
		switch level {
		case "v1", "v2", "v3", "v4":
		default:
			return false, fmt.Errorf("unsupported GOAMD64 level %q", level)
		}
		if runtime.GOARCH != "amd64" {
			return false, nil
		}
		host, _ := config.ArchLevels("native")
		return host >= level, nil
	}))
	add("trimpath", script.OnceCondition("test binary was built with -trimpath", isTrimpath))
	add("default-cgo", lazyBool("when CGO_ENABLED=1|0 was set in make.bash", defaultCgo))
	add("wasmfx", script.OnceCondition("node runs WebAssembly stack switching (cont.new/resume)", hasWasmFx))
	add("expstamp", script.OnceCondition("installed go driver was built with experiments and reports a -X: stamp", forkDriverStamped))

	return conds
}

// forkDriverStamped reports whether the installed fork go driver was built
// with experiments baked in: upstream stamps such a toolchain's version with
// a -X: suffix, so plain-version assertions must tolerate it.
func forkDriverStamped() (bool, error) {
	out, err := exec.Command(filepath.Join(testGOROOT, "bin", "go"+cfg.ExeSuffix), "version").Output()
	if err != nil {
		return false, fmt.Errorf("test goroot driver: %w", err)
	}
	return strings.Contains(string(out), "-X:"), nil
}

// hasWasmFx reports whether node in PATH runs WebAssembly stack switching:
// node must accept --experimental-wasm-wasmfx and compile a module with a
// cont type and a cont.new instruction. When LOGOS_REQUIRE_WASMFX is set,
// a missing engine fails the test instead of skipping it, so required proof
// never passes as a skip.
func hasWasmFx() (bool, error) {
	node, err := exec.LookPath("node")
	if err != nil {
		return requireWasmFx("node not in PATH")
	}
	probe := `const bytes = new Uint8Array([0,97,115,109,1,0,0,0,1,9,3,96,0,0,96,0,0,93,1,3,2,1,0,10,9,1,7,0,208,1,224,2,26,11]); try { new WebAssembly.Module(bytes); } catch (e) { process.exit(1); }`
	if err := exec.Command(node, "--experimental-wasm-wasmfx", "-e", probe).Run(); err != nil {
		return requireWasmFx("node lacks --experimental-wasm-wasmfx stack-switching instructions")
	}
	return true, nil
}

func requireWasmFx(reason string) (bool, error) {
	if os.Getenv("LOGOS_REQUIRE_WASMFX") != "" {
		return false, errors.New("LOGOS_REQUIRE_WASMFX=1 but " + reason)
	}
	return false, nil
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

// Capture the major, minor and (optionally) patch version, but ignore anything later
var gitVersLineExtract = regexp.MustCompile(`git version\s+(\d+\.\d+(?:\.\d+)?)`)

func gitVersion() (string, error) {
	gitOut, runErr := exec.Command("git", "version").CombinedOutput()
	if runErr != nil {
		return "v0", fmt.Errorf("failed to execute git version: %w", runErr)
	}
	matches := gitVersLineExtract.FindSubmatch(gitOut)
	if len(matches) < 2 {
		return "v0", fmt.Errorf("git version extraction regexp did not match version line: %q", gitOut)
	}
	return "v" + string(matches[1]), nil
}

func hasAtLeastGitVersion(minVers string) (bool, error) {
	gitVers, gitVersErr := gitVersion()
	if gitVersErr != nil {
		return false, gitVersErr
	}
	return semver.Compare(minVers, gitVers) <= 0, nil
}

func gitSupportsSHA256() (bool, error) {
	return hasAtLeastGitVersion("v2.29")
}

func defaultCgo() bool {
	return buildcfg.DefaultCGO_ENABLED == "1" || buildcfg.DefaultCGO_ENABLED == "0"
}
